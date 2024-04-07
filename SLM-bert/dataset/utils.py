# ===================================================================================================================================================================================================
# For a fair comparison, we construct the dataset for bert backbone folllowing IDBR on Setting(FULL)
# More detail can refer to: https://aclanthology.org/2021.naacl-main.218.pdf
# And the dataset download can refer to: https://github.com/SALT-NLP/IDBR (https://drive.google.com/file/d/1rWcgnVcNpwxmBI3c5ovNx-E8XKOEL77S/view)
# Please notice that in IDBR using one head for bert classifier, which means that each adding new domain task needs to retrain the whole classifier, or requires to know all label offset at once.
# In our code, we use the label num as keys to provide as little prior information as possible in bert. 
# In t5, we following previous work to map original labels to words.
# ===================================================================================================================================================================================================

import pandas as pd
import datasets
from datasets import load_dataset
import numpy as np

DATA_NAME_TO_KEYS = {
    "ag_news": ("text", ),
    "yelp_review_full": ("text", ),
    "yahoo_answers_topics": ("question_content", "question_content", "best_answer"),
    "dbpedia_14": ("title", "content"),

    "amazon": ("content", ),
    "ag": ("content", ),
    "dbpedia": ("content", ),
    "yahoo": ("content", ),
    "yelp": ("content", ),
}
        

def get_dataset_by_name(data_name, num_proc=4, k_samples=-1, seed=0):
    # ===================== LOAD DATA ==========================
    if data_name in ['amazon', 'dbpedia', 'ag', 'yahoo', 'yelp']:
        dataset = datasets.DatasetDict()
        for split in ['train', 'test']:
            df = pd.read_csv('data/src/data/'+data_name+'/'+split+'.csv', header=None)
            df = df.rename(columns={0: "label", 1: "title", 2: "content"})
            df['label'] = df['label'] - 1
            dataset[split] = datasets.Dataset.from_pandas(df)
    else:
        dataset = load_dataset(data_name)

    if data_name == "yahoo_answers_topics":
        for split in ['train', 'test']: 
            good_id = np.load('data/src/data/yahoo/good_id_yahoo_{}2.npy'.format(split))
            dataset[split] = dataset[split].select(good_id)
        dataset = dataset.rename_column("topic", "label")
        
    # ===================== RANDOM SELECT K SAMPLES =======================
    if k_samples != -1:
        for split in ['train', 'test']:
            if split == 'test':
                k = int(k_samples * 0.15) if data_name not in ["dbpedia_14", "sst2"] else int(k_samples * 0.1)
            else:
                k = k_samples

            idx_random = np.array([], dtype='int64')
            for label in set(dataset[split]['label']):
                idx = np.where(np.array(dataset[split]['label']) == label)[0]
                idx = np.random.choice(idx, min(k, idx.shape[0]), replace=False)
                idx_random = np.concatenate([idx_random, idx])
            np.random.seed(seed)
            np.random.shuffle(idx_random)
            dataset[split] = dataset[split].select(idx_random)

    # ===================== GENERATE RAW TEXT =============================
    dataset = dataset.map(lambda x: dict(raw_text="[SEP]".join([x[key] for key in DATA_NAME_TO_KEYS[data_name]])), batched=False, num_proc=num_proc)
    return dataset
