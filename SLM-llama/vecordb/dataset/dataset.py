from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union, List
import torch
import transformers
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
import json
import pandas as pd
from datasets import load_dataset
import datasets
import numpy as np
import os
import io


PROMPT_DICT = {
    "prompt_input": (
        "#Instruction:\n{instruction}\n#Input:\n{input}\n# Response:"
    ),
    "prompt_no_input": (
        "#Instruction:\n{instruction}\n# Response:"
    ),
}

TASK_TO_DATANAME = {
    "medical": "LinhDuong/chatdoctor-200k",
}

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def load_json_to_raw_text(data_path):
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    data_list = []
    list_data_dict = jload(data_path)
    for example in list_data_dict:
        data_list.append(
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        )
    return data_list


def load_datasets_to_raw_text(data_name,
                    split='train'):
    
    if data_name in TASK_TO_DATANAME.keys():
        data_name = TASK_TO_DATANAME[data_name]
    dataset = load_dataset(data_name, split=split)
    raw_texts = []
    print("Start to extract the raw text of {}".format(data_name))
    for data in dataset:
        raw_text = data['instruction'] + " ### " + data['input'] if 'instruction' in data.keys() else \
                   data['input']
        raw_texts.append(raw_text)
    return raw_texts


class RetriverDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_paths):
        super(RetriverDataset, self).__init__()
        if isinstance(data_paths, str):
            data_paths = [data_paths]

        self.data_list = []
        for data_path in data_paths:
            if os.path.splitext(data_path)[1] == '.json':
                self.data_list += load_json_to_raw_text(data_path)
            else:
                self.data_list += load_datasets_to_raw_text(data_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        return self.data_list[i]
    

class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, instances: Sequence[str]):
        return self.tokenizer(instances, padding=True, truncation=True, return_tensors='pt')