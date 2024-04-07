import os
from models.retriever import Retriever
import torch
from transformers import AutoTokenizer
import torch.nn as nn
import json
import random
import tqdm
from dataset.dataset import load_datasets_to_raw_text, load_json_to_raw_text
from models.retriever import Retriever, Config
from models.utils import load_config
cos = nn.CosineSimilarity(dim=-1)

datas_path = [
    "amazon",
    "ag",
    "yelp",
    "yahoo",
    "dbpedia"
]
tokenizer_path = 'sentence-transformers/nli-roberta-base-v2'
config_path = '/dataset/zhuotaotian/bhpeng/vectordb/output/config_6*20_all.json'
model_path = "/dataset/zhuotaotian/bhpeng/vectordb/output/model_6*20_all.pth"
save_path = "/dataset/zhuotaotian/bhpeng/vectordb/output/record.json"
batch_size = 64

print("================ LOAD MODEL ===========")
config = load_config(config_path)
pool_size = config.pool_size
groups = config.groups
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = Retriever(config)
model.load_state_dict(torch.load(model_path), strict=False)
model = model.cuda()
model = model.eval()
print("================ FINISH LOADING MODEL ===========")

record = dict()
for data_path in datas_path:
    record[data_path] = []
    if os.path.splitext(data_path)[1] == '.json':
        data_list = load_json_to_raw_text(data_path)
        data_name = os.path.split(data_path)[1]
        data_name = os.path.splitext(data_name)[0]
    else:
        data_list = load_datasets_to_raw_text(data_path, split='test')
        data_name = data_path
    batched_data = []
    for i, text in enumerate(data_list):
        if i % batch_size == 0:
            batched_data.append([])
        batched_data[-1].append(text)
    

    print("="*30, data_name, "="*30)
    for texts in tqdm.tqdm(batched_data):
        idxs = model(texts, return_topk_index=True)['topk_index']
        for g in range(groups):
            idx = idxs[:, g, :]
            idx = torch.flatten(idx)
            for i in idx:
                record[data_path].append({
                    "group": g,
                    "index": int(i),
                    "label": data_path
                })

with open(save_path, 'w') as f:
    json.dump(record, f)