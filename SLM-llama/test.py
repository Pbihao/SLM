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
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import re
import argparse
from evaluate import load
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class DataCollatorWithPadding:

    eos_token_id: PreTrainedTokenizerBase
    task: str = "finance"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = dict(raw_text=[feature.pop('raw_text') for feature in features])  # num_labels can not use
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
    

def test_task(
    task,
    model, 
    tokenizer,
    batch_size=1, 
    output_dir=None):
    assert batch_size==1, "batch_size can only be 1"
    dataset = get_dataset_by_name(task, tokenizer=tokenizer, split='test')
    dataset.set_format(columns=['input_ids', 'attention_mask', 'label', 'raw_text'])
    data_collator = DataCollatorWithPadding(
                        eos_token_id=tokenizer.eos_token_id, 
                        task=task)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=data_collator)

    model = model.cuda().eval()
    results = []
    for data in tqdm(data_loader):
        data['input_ids'] = data['input_ids'].to("cuda")
        data['attention_mask'] = data['attention_mask'].to("cuda")
        data['labels'] = data['labels'].to("cuda")
        output = model.generate(
            input_ids=data['input_ids'],
            raw_text=data['raw_text'],
            attention_mask=data['attention_mask'],
            max_length=1024 if task == "medical" else 600
        )[0]

        label = data['labels'][0]
        output = output[data['input_ids'].shape[1]:]
        label = tokenizer.decode(label, skip_special_tokens=True)
        output = tokenizer.decode(output, skip_special_tokens=True)
        results.append(dict(label=label, output=output))

    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        path = os.path.join(output_dir, "result.json")
        if os.path.exists(path):
            print(" ====================== File has existed... =========================")
        else:
            print(" ====================== Save task: {} results to: {} =========================".format(task, path))
            with open(path, "w") as f:
                json.dump(results, f)

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="finance", type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--output_dir', default="./tmp/results", type=str)
    parser.add_argument('--model_path', default=None, type=str)
    args = parser.parse_args()

    assert args.model_path is not None, "model_path should not be None"

    num_workers = args.num_workers
    output_dir = args.output_dir
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    print("Load checkpoint from: {}".format(args.model_path))
    config = LlamaCLConfig()
    model = ScalableLM(config)
    model.load_state_dict(torch.load(args.model_path), strict=False)
    test_task(task=args.task, 
            model=model, 
            tokenizer=tokenizer,
            batch_size=1,
            output_dir=os.path.join(output_dir, args.task))
    
        