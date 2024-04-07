import torch
import json
import random
import os
from models.retriever import Config, Retriever
import torch.nn as nn
from einops import rearrange
import tqdm
from dataset.dataset import load_datasets_to_raw_text, load_json_to_raw_text


def save_config(config, path):
    dir_path = os.path.split(path)[0]
    assert os.path.exists(dir_path), "{} path doesn't exist!".format(dir_path)
    with open(path, "w") as f:
        json.dump(
            config.__dict__,
            f,
            indent=4
        )


def conver_json_to_config(config_json):
    config = Config()
    keys = config.__dict__.keys()
    for key, value in config_json.items():
        if key in keys:
            setattr(config, key, value)
    return config


def load_config(path):
    with open(path) as f:
        config_json = json.load(f)
    return conver_json_to_config(config_json)


def merge_retriever(
    config_paths,
    state_dict_paths,
    new_config_path,
    new_state_dict_path,
    config_same_attr=['weight_topk', 'groups', 'similarity_type', ],
    config_merge_attr=['pool_size'],
):
    configs = []
    for config_path in config_paths:
        configs.append(load_config(config_path))
    final_config = configs.pop(0)
    
    for key in config_same_attr:
        for config in configs:
            assert getattr(config, key) == getattr(final_config, key), "config: {} conflication!".format(key)

    current_size = final_config.pool_size
    for key in config_merge_attr:
        for config in configs:
            setattr(final_config, key, getattr(final_config, key) + getattr(config, key))
            for k, v in config.task_pool_index_range.items():
                assert k not in final_config.task_pool_index_range, "{} is in multi pool".format(k)
                final_config.task_pool_index_range[k] = [v[0] + current_size, v[1] + current_size]
            current_size += config.pool_size
    save_config(final_config, new_config_path)

    state_dicts = []
    for state_dict_path in state_dict_paths:
        state_dicts.append(torch.load(state_dict_path))
    final_state_dict = state_dicts.pop(0)

    for state_dict in state_dicts:
        final_state_dict['keys'] = torch.cat([final_state_dict['keys'], state_dict['keys']], dim=2)
    torch.save(final_state_dict, new_state_dict_path)


def group_avg_initialization(
    config_path,
    data_path,
    model_save_path,
    batch_size=64,
    device = "cuda"
):
    config = load_config(config_path)
    model = Retriever(config)
    model = model.to(device)
    
    if os.path.splitext(data_path) == '.json':
        data_list = load_json_to_raw_text(data_path)
    else:
        data_list = load_datasets_to_raw_text(data_path)
    random.shuffle(data_list)

    keys = []
    for pool_idx in range(config.pool_size):
        print("Start to initialize the {} key".format(pool_idx))
        sub_data_list = data_list[pool_idx::config.pool_size]
        batch_index = list(range(0, len(sub_data_list), batch_size))
        batch_index.append(len(sub_data_list))

        avg_vector = torch.zeros(model.model_hidden_size).to(device)
        for idx in tqdm.tqdm(range(len(batch_index) - 1)):
            inputs = sub_data_list[batch_index[idx] : batch_index[idx + 1]]
            embeddings = model.encode(inputs)
            embeddings = embeddings.mean(dim=0)
            avg_vector = avg_vector + embeddings
        avg_vector = avg_vector / (len(batch_index) - 1)
        avg_vector = rearrange(avg_vector, "(g c) -> 1 g c", g=config.groups)
        keys.append(avg_vector)
    keys = torch.stack(keys, dim=2)
    model.set_keys(keys)

    model = model.to("cpu")
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()    
    else:
        state_dict = model.state_dict()
    keys = list(state_dict.keys())
    for key in keys:
        if "model." in key:
            state_dict.pop(key)
    torch.save(state_dict, model_save_path)
