import torch
import torch.nn as nn
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerateOutput
from models.llama import LlamaForCausalLM
from models.retriever import Retriever
from typing import List, Optional, Union, Tuple, Callable
import os
from transformers import PreTrainedModel
import collections
from models.config import LlamaCLConfig
import re
import time
from copy import deepcopy
from transformers.modeling_utils \
import (SAFE_WEIGHTS_INDEX_NAME,
WEIGHTS_INDEX_NAME, 
SAFE_WEIGHTS_NAME, PreTrainedModel, 
logging, 
_add_variant, 
WEIGHTS_NAME, 
is_safetensors_available, 
unwrap_model, 
get_parameter_dtype, 
custom_object_save, 
id_tensor_storage, 
safe_save_file, shard_checkpoint)
IS_SAGEMAKER_MP_POST_1_10 = False
import json
import warnings
logger = logging.get_logger(__name__)


class ScalableLM(PreTrainedModel):
    config_class = LlamaCLConfig
    _keys_to_ignore_on_save = ['model', 'retriever.bert', 'src_weight_offset']
    _keys_to_ignore_on_load_missing = ['model', 'retriever.bert']
    _keys_to_ignore_on_load_unexpected = ['src_weight_offset']

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.task_list = config.task_pool_index_range.keys()

        self.model = LlamaForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16)
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        self.retriever = Retriever(config)
        if config.retriever_state_dict is not None:
            self.retriever.keys = nn.parameter.Parameter(torch.load(config.retriever_state_dict)['keys'], requires_grad=False)

        self.src_weight_offset = None
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        task: Optional[str] = None,
        raw_text = None,
    ):  
        if self.training:
            # self.restore_other_task_weight_offset(task)
            pool_mask = self.get_task_mask(task)
        elif not self.config.disable_task_id:
            pool_mask = self.get_task_mask(task)
                
        retriever_outputs = self.retriever(
            inputs=raw_text,
            pool_mask=pool_mask,
        )
        weight_offset = retriever_outputs['weight_offset']
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            weight_offset=weight_offset,
        )
        return outputs
    
    def generate(self, raw_text=None, task=None, **kwargs) -> GenerateOutput | torch.LongTensor:
        pool_mask = None
        if task is not None and not self.config.disable_task_id:
            pool_mask = self.get_task_mask(task)
        retriever_outputs = self.retriever(
            inputs=raw_text,
            pool_mask=pool_mask,
        )
        weight_offset = retriever_outputs['weight_offset']
        return self.model.generate(weight_offset=weight_offset, **kwargs)
    
    def update_task(self):
        self.src_weight_offset = deepcopy(self.retriever.weight_offset)
        self.src_weight_offset.requires_grad = False

    def restore_other_task_weight_offset(self, task):
        assert self.training, "Only need to restore when training"
        assert self.src_weight_offset is not None, "self.src_weight_offset is None"
        l_idx, r_idx = self.config.task_pool_index_range[task]
        self.retriever.weight_offset.requires_grad = False
        self.retriever.weight_offset[:, :l_idx, ...] = self.src_weight_offset[:, :l_idx, ...]
        self.retriever.weight_offset[:, r_idx:, ...] = self.src_weight_offset[:, r_idx:, ...]
        self.retriever.weight_offset.requires_grad = True


    # Filter out the tasks not in the test list
    def get_task_list_mask(self):
        mask = torch.zeros(self.config.pool_size, dtype=torch.int, device=self.retriever.keys.device)
        for task in self.config.task_list:
            l_idx, r_idx = self.config.task_pool_index_range[task]
            mask[l_idx: r_idx] = 1
        return mask
    
    # If know the task id when inference. Not use when in the inference withou task_id setting
    def get_task_mask(self, task):
        assert not self.config.disable_task_id, "Can not use task_id"
        mask = torch.zeros(self.config.pool_size, dtype=torch.int, device=self.retriever.keys.device)
        l_idx, r_idx = self.config.task_pool_index_range[task]
        mask[l_idx: r_idx] = 1
        return mask
    
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "10GB",
        safe_serialization: bool = False,
        variant: Optional[str] = None,
        **kwargs,
    ):
        if getattr(self, "is_loaded_in_8bit", False) and getattr(self, "is_8bit_serializable", False):
            warnings.warn(
                "You are calling `save_pretrained` to a 8-bit converted model you may likely encounter unexepected"
                " behaviors. If you want to save 8-bit models, make sure to have `bitsandbytes>0.37.2` installed.",
                UserWarning,
            )

        if getattr(self, "is_loaded_in_4bit", False):
            raise NotImplementedError(
                "You are calling `save_pretrained` on a 4-bit converted model. This is currently not supported"
            )

        if "save_config" in kwargs:
            warnings.warn(
                "`save_config` is deprecated and will be removed in v5 of Transformers. Use `is_main_process` instead."
            )
            is_main_process = kwargs.pop("save_config")
        if safe_serialization and not is_safetensors_available():
            raise ImportError("`safe_serialization` requires the `safetensors library: `pip install safetensors`.")

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        # Only save the model itself if we are using distributed training
        model_to_save = unwrap_model(self)

        # save the string version of dtype to the config, e.g. convert torch.float32 => "float32"
        # we currently don't use this setting automatically, but may start to use with v5
        dtype = get_parameter_dtype(model_to_save)
        model_to_save.config.torch_dtype = str(dtype).split(".")[1]

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self.config)

        # Save the config
        if is_main_process:
            model_to_save.config.save_pretrained(save_directory)
            if self.can_generate():
                model_to_save.generation_config.save_pretrained(save_directory)

        # Save the model
        if state_dict is None:
            state_dict = model_to_save.state_dict()

        # Handle the case where some state_dict keys shouldn't be saved
        if self._keys_to_ignore_on_save is not None:
            keys = list(state_dict.keys())
            for key in keys:
                for ignore_keys in self._keys_to_ignore_on_save:
                    if ignore_keys in key:
                        del state_dict[key]    
                        break
        if safe_serialization:
            # Safetensors does not allow tensor aliasing.
            # We're going to remove aliases before saving
            ptrs = collections.defaultdict(list)
            for name, tensor in state_dict.items():
                ptrs[id_tensor_storage(tensor)].append(name)

            # These are all the pointers of shared tensors.
            shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}
            warn_names = set()
            for names in shared_ptrs.values():
                # Removing the keys which are declared as known duplicates on
                # load. This allows to make sure the name which is kept is consistent.
                if self._keys_to_ignore_on_load_missing is not None:
                    found = 0
                    for name in sorted(names):
                        matches_pattern = any(re.search(pat, name) for pat in self._keys_to_ignore_on_load_missing)
                        if matches_pattern and name in state_dict:
                            found += 1
                            if found < len(names):
                                del state_dict[name]

                # When not all duplicates have been cleaned, still remove those keys, but put a clear warning.
                # If the link between tensors was done at runtime then `from_pretrained` will not get
                # the key back leading to random tensor. A proper warning will be shown
                # during reload (if applicable), but since the file is not necessarily compatible with
                # the config, better show a proper warning.
                found = 0
                for name in names:
                    if name in state_dict:
                        found += 1
                        if found > 1:
                            del state_dict[name]
                            warn_names.add(name)
            if len(warn_names) > 0:
                logger.warning_once(
                    f"Removed shared tensor {warn_names} while saving. This should be OK, but check by verifying that you don't receive any warning while reloading",
                )

        # Shard the model if it is too big.
        weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
        weights_name = _add_variant(weights_name, variant)

        shards, index = shard_checkpoint(state_dict, max_shard_size=max_shard_size, weights_name=weights_name)

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            weights_no_suffix = weights_name.replace(".bin", "").replace(".safetensors", "")

            # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
            filename_no_suffix = filename.replace(".bin", "").replace(".safetensors", "")
            reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

            if (
                filename.startswith(weights_no_suffix)
                and os.path.isfile(full_filename)
                and filename not in shards.keys()
                and is_main_process
                and reg.fullmatch(filename_no_suffix) is not None
            ):
                os.remove(full_filename)

        # Save the model
        for shard_file, shard in shards.items():
            if safe_serialization:
                # At some point we will need to deal better with save_function (used for TPU and other distributed
                # joyfulness), but for now this enough.
                safe_save_file(shard, os.path.join(save_directory, shard_file), metadata={"format": "pt"})
            else:
                save_function(shard, os.path.join(save_directory, shard_file))

        if index is None:
            path_to_weights = os.path.join(save_directory, _add_variant(WEIGHTS_NAME, variant))
            logger.info(f"Model weights saved in {path_to_weights}")
        else:
            save_index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_directory, _add_variant(save_index_file, variant))
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            logger.info(
                f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("use_auth_token"),
            )