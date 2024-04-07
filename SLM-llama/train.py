from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerBase
import numpy as np
from transformers import TrainingArguments, Trainer
from models.config import LlamaCLConfig
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from models.slm import ScalableLM
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Union, List, Any, Dict
from dataset.dataset import get_dataset_by_name, IGNORE_INDEX
from datasets.utils.logging import disable_progress_bar
import os
import re
import argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class DataCollatorWithPadding:

    eos_token_id: PreTrainedTokenizerBase
    task: str = "finance"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = dict(task=self.task, 
                     raw_text=[feature.pop('raw_text') for feature in features])  # num_labels can not use
        label_key = 'labels' if 'labels' in features else 'label'
        input_ids, attention_mask, labels = tuple([torch.tensor(feature[key]) for feature in features] for key in ['input_ids', 'attention_mask', label_key])
        input_ids = nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.eos_token_id
        )
        attention_mask = nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels = nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        batch.update({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })
        return batch
    


def train_task(
    task,
    model, 
    tokenizer,
    epoch, 
    tqdm=False,
    lr=1e-3, 
    batch_size=8, 
    deepspeed=None,
    output_dir=None):
    
    dataset = get_dataset_by_name(task, tokenizer=tokenizer)
    dataset.set_format(columns=['input_ids', 'attention_mask', 'label', 'raw_text'])
    data_collator = DataCollatorWithPadding(
                        eos_token_id=tokenizer.eos_token_id, 
                        task=task)

    training_args = TrainingArguments(
        output_dir=output_dir,
        optim='adamw_torch',
        learning_rate=lr,
        report_to=None,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epoch,
        weight_decay=0.01,
        save_strategy="epoch",
        warmup_ratio=0.03,
        logging_steps=10,
        remove_unused_columns=False,
        ddp_find_unused_parameters=True,
        gradient_accumulation_steps =8,
        bf16=True,
        save_total_limit=1,
        disable_tqdm=not tqdm,
        deepspeed=deepspeed,
    )
    training_args = training_args.set_lr_scheduler(name="cosine", warmup_ratio=0.03)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="history", type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--output_dir', default="./outputs/all", type=str)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--deepspeed', default="./default_offload_opt_param.json", type=str)
    parser.add_argument('--tqdm', default=False, action='store_true')
    args = parser.parse_args()
    if not args.tqdm:
        disable_progress_bar()
    
    num_workers = args.num_workers
    output_dir = args.output_dir
    
    if args.model_path is None:
        config = LlamaCLConfig()
        model = ScalableLM(config)
    else:
        print("Load checkpoint from: {}".format(args.model_path))
        config = LlamaCLConfig()
        model = ScalableLM(config)
        model.load_state_dict(torch.load(args.model_path), strict=False)
            
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # if config.disable_task_id:
    #     model.update_task(args.task)
    train_task(task=args.task, 
                model=model, 
                tokenizer=tokenizer,
                epoch=args.epoch, 
                lr=args.lr, 
                tqdm=args.tqdm,
                batch_size=args.batch_size,
                deepspeed=args.deepspeed,
                output_dir=os.path.join(output_dir, args.task))
    
        