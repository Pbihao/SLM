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
    # "/newdata/bohaopeng/research/llm/vectordb/data/alpaca_data.json",
    # "/newdata/bohaopeng/research/llm/vectordb/data/medical_76K.json",
    # "/newdata/bohaopeng/research/llm/vectordb/data/law_train_43k.json",
    "amazon",
    "ag_news",
    "yelp_review_full",
    "yahoo_answers_topics",
    "dbpedia_14"
]
tokenizer_path = 'sentence-transformers/nli-roberta-base-v2'
config_path = '/newdata/bohaopeng/research/llm/vectordb/output/CL/config_6*60_all_or.json'
model_path = "/newdata/bohaopeng/research/llm/vectordb/output/CL/model_6*60_all_or.pth"
save_path = "/newdata/bohaopeng/research/llm/vectordb/output/CL/results_6*60_all_or.txt"
batch_size = 64

config = load_config(config_path)
pool_size = config.pool_size
groups = config.groups
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = Retriever(config)
model.load_state_dict(torch.load(model_path), strict=False)
model = model.cuda()

fw = open(save_path, "w")
for data_path in datas_path:
    if os.path.splitext(data_path)[1] == '.json':
        data_list = load_json_to_raw_text(data_path)
        data_name = os.path.split(data_path)[1]
        data_name = os.path.splitext(data_name)[0]
    else:
        data_list = load_datasets_to_raw_text(data_path)
        data_name = data_path
    batched_data = []
    for i, text in enumerate(data_list):
        if i % batch_size == 0:
            batched_data.append([])
        batched_data[-1].append(text)
    

    print("="*30, data_name, "="*30)
    count_all = torch.zeros(groups * pool_size, dtype=torch.long)
    count_split = torch.zeros(groups, pool_size)

    for texts in tqdm.tqdm(batched_data):
        idxs = model(texts, return_topk_index=True)['topk_index']
        for g in range(groups):
            idx = idxs[:, g, :]
            idx = torch.flatten(idx)
            for i in idx:
                count_split[g][i] += 1
                count_all[i + g * pool_size] += 1
    
    fw.write("\n\n{}\n".format(data_name))
    for i in range(groups * pool_size):
        if i % pool_size == 0:
            fw.write("\n".format(count_all[i]))
        fw.write("{}\n".format(count_all[i]))
        
fw.close()