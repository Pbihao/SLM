from typing import Any
from datasets import load_dataset
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
from copy import deepcopy
import torch
import math
import os
import re
import numpy as np


TASK_TO_DATANAME = {
    "medical": "LinhDuong/chatdoctor-200k",
    "finance": "AdiOO7/llama-2-finance"
}


PROMPT_TEMPLATE = {
'with_sys': """[INST] <<SYS>>
{instruction}
<</SYS>>

{input} [/INST] """,

'without_sys': """[INST] {input} [/INST] """
}


IGNORE_INDEX = -100
IGNORE_INDEX_LIST = [IGNORE_INDEX] * 2048


def extract_from_finance(example):
    text = example['text']
    m = re.search('### Instruction:(.+?)### Human:', text)
    instruction = m.group(1).strip() if m else None
    m = re.search('### Human:(.+?)### Assistant:', text)
    input = m.group(1).strip() if m else None
    output = text.split("### Assistant:")[1].strip()
    return dict(
        instruction=instruction,
        input=input,
        output=output
    )

def extract_from_history(example):
    question = example['question']
    choices = example['choices']
    output = example['answer']
    input="Question:{}\n Choices:{}\n".format(question, str(choices))
    output = chr(ord('@')+int(output)+1)
    return dict(
        input=input,
        output=output
    )

@dataclass
class Preprocess:
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 512
    task:str = 'finance'

    def __call__(self, example) -> Any:
        if 'finance' in self.task:
            example = extract_from_finance(example)
        elif 'history' in self.task:
            example = extract_from_history(example)
        source = PROMPT_TEMPLATE['with_sys'].format_map(example) if 'instruction' in example.keys() else\
                 PROMPT_TEMPLATE['without_sys'].format_map(example)
        raw_text = example['instruction'] + " ### " + example['input'] if 'instruction' in example.keys() else \
                   example['input']
        target = example['output'] + "</s>"
        source = self.tokenizer(
            source,
            max_length=self.max_length,
            truncation=True,
        )
        target = self.tokenizer(
            target,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens =False,
        )
        source_length = len(source['input_ids'])
        input_ids = source['input_ids'] + target['input_ids']
        attention_mask = source['attention_mask'] + target['attention_mask']
        label = deepcopy(input_ids)
        label[:source_length] = IGNORE_INDEX_LIST[:source_length]
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            label=label,
            raw_text=raw_text
        )

        
@dataclass
class PreprocessTest:
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 512
    task:str = 'finance'

    def __call__(self, example) -> Any:
        if 'finance' in self.task:
            example = extract_from_finance(example)
        elif 'history' in self.task:
            example = extract_from_history(example)
        source = PROMPT_TEMPLATE['with_sys'].format_map(example) if 'instruction' in example.keys() else\
                 PROMPT_TEMPLATE['without_sys'].format_map(example)
        raw_text = example['instruction'] + " ### " + example['input'] if 'instruction' in example.keys() else \
                   example['input']
        target = example['output'] + "</s>"
        source = self.tokenizer(
            source,
            max_length=self.max_length,
            truncation=True,
        )
        target = self.tokenizer(
            target,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens =False,
        )
        input_ids = source['input_ids']
        attention_mask = source['attention_mask']
        label = target['input_ids']
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            label=label,
            raw_text=raw_text
        )
    




def get_dataset_by_name(task, tokenizer, num_proc=4, split='train', dive=None):
    # ===================== LOAD DATA ==========================
    if "history" in task:
        dataset = load_dataset("Stevross/mmlu", "high_school_european_history", split="auxiliary_train")
    else:
        data_name = TASK_TO_DATANAME[task]
        dataset = load_dataset(data_name, split="train")
    
    # ===================== SELECT TRAIN INDEX ============================
    if os.path.exists("dataset/index/{}/{}.npy".format(split, task)):
        print("============================= LOAD {} DATA LIST ====================================".format(split.upper()))
        dataset = dataset.select(np.load("dataset/index/{}/{}.npy".format(split, task)))

        # ================== dive ===========================
        if dive is not None:
            idx = np.arange(len(dataset))
            per_num = int(math.ceil(len(dataset) / 8))
            print(" =================== form {} to {} ===========================".format(per_num * dive, min(per_num * (dive + 1), len(dataset))))
            idx = idx[per_num * dive: min(per_num * (dive + 1), len(dataset))]
            dataset = dataset.select(idx)

    # ===================== GENERATE RAW TEXT =============================
    if split == 'train':
        dataset = dataset.map(Preprocess(tokenizer=tokenizer, task=task), batched=False, num_proc=num_proc)
    else:
        dataset = dataset.map(PreprocessTest(tokenizer=tokenizer, task=task), batched=False, num_proc=num_proc)
    return dataset