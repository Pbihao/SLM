# ------------------------- pool: 4, groups: 6 -------------------------------------------
sc create --name g3-p4-1 --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 4 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/vectordb && \
python train.py -g 4 --epoch 3   --lr_step_size 2   \
--lr  0.0005    --batch_size  4     \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/vectordb/output-g3-p4/model_3*4_ag_news.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/vectordb/output-g3-p4/config_3*4_ag_news.json  \
--data_paths  ag"

sc create --name g3-p4-2 --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 4 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/vectordb && \
python train.py -g 4 --epoch 3   --lr_step_size 2   \
--lr  0.0005    --batch_size  4     \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/vectordb/output-g3-p4/model_3*4_amazon.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/vectordb/output-g3-p4/config_3*4_amazon.json  \
--data_paths  amazon"

sc create --name g3-p4-3 --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 4 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/vectordb && \
python train.py -g 4 --epoch 3   --lr_step_size 2   \
--lr  0.0005    --batch_size  4     \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/vectordb/output-g3-p4/model_3*4_dbpedia.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/vectordb/output-g3-p4/config_3*4_dbpedia.json  \
--data_paths  dbpedia"

sc create --name g3-p4-4 --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 4 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/vectordb && \
python train.py -g 4 --epoch 3   --lr_step_size 2   \
--lr  0.0005    --batch_size  4     \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/vectordb/output-g3-p4/model_3*4_yahoo.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/vectordb/output-g3-p4/config_3*4_yahoo.json  \
--data_paths  yahoo"

sc create --name g3-p4-5 --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 4 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/vectordb && \
python train.py -g 4 --epoch 3   --lr_step_size 2   \
--lr  0.0005    --batch_size  4     \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/vectordb/output-g3-p4/model_3*4_yelp.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/vectordb/output-g3-p4/config_3*4_yelp.json  \
--data_paths  yelp"



# ------------------------- pool: 24, groups: 1 ------------------------------------------
sc create --name g1-p24-1 --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 6 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/vectordb && \
python train.py -g 6 --epoch 3   --lr_step_size 2   \
--lr  0.0005    --batch_size  4     \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/vectordb/output-group-1-nomask/model_1*24_ag_news.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/vectordb/output-group-1-nomask/config_1*24_ag_news.json  \
--data_paths  ag"

sc create --name g1-p24-2 --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 6 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/vectordb && \
python train.py -g 6 --epoch 3   --lr_step_size 2   \
--lr  0.0005    --batch_size  4     \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/vectordb/output-group-1-nomask/model_1*24_amazon.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/vectordb/output-group-1-nomask/config_1*24_amazon.json  \
--data_paths  amazon"

sc create --name g1-p24-3 --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 6 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/vectordb && \
python train.py -g 6 --epoch 3   --lr_step_size 2   \
--lr  0.0005    --batch_size  4     \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/vectordb/output-group-1-nomask/model_1*24_dbpedia.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/vectordb/output-group-1-nomask/config_1*24_dbpedia.json  \
--data_paths  dbpedia"

sc create --name g1-p24-4 --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 6 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/vectordb && \
python train.py -g 6 --epoch 3   --lr_step_size 2   \
--lr  0.0005    --batch_size  4     \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/vectordb/output-group-1-nomask/model_1*24_yahoo.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/vectordb/output-group-1-nomask/config_1*24_yahoo.json  \
--data_paths  yahoo"

sc create --name g1-p24-5 --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 6 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/vectordb && \
python train.py -g 6 --epoch 3   --lr_step_size 2   \
--lr  0.0005    --batch_size  4     \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/vectordb/output-group-1-nomask/model_1*24_yelp.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/vectordb/output-group-1-nomask/config_1*24_yelp.json  \
--data_paths  yelp"

# ------------------------- pool: 4, groups: 6 ------------------------------------------
sc create --name nom-g6-p4-1 --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 2 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/vectordb && \
python train.py -g 2 --epoch 3   --lr_step_size 2   \
--lr  0.0005    --batch_size  4     \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/vectordb/output-group/model_6*4_ag_news.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/vectordb/output-group/config_6*4_ag_news.json  \
--data_paths  ag"

sc create --name nom-g6-p4-2 --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 2 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/vectordb && \
python train.py -g 2 --epoch 3   --lr_step_size 2   \
--lr  0.0005    --batch_size  4     \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/vectordb/output-group/model_6*4_amazon.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/vectordb/output-group/config_6*4_amazon.json  \
--data_paths  amazon"

sc create --name nom-g6-p4-3 --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 2 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/vectordb && \
python train.py -g 2 --epoch 3   --lr_step_size 2   \
--lr  0.0005    --batch_size  4     \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/vectordb/output-group/model_6*4_dbpedia.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/vectordb/output-group/config_6*4_dbpedia.json  \
--data_paths  dbpedia"

sc create --name nom-g6-p4-4 --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 2 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/vectordb && \
python train.py -g 2 --epoch 3   --lr_step_size 2   \
--lr  0.0005    --batch_size  4     \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/vectordb/output-group/model_6*4_yahoo.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/vectordb/output-group/config_6*4_yahoo.json  \
--data_paths  yahoo"

sc create --name nom-g6-p4-5 --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 2 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/vectordb && \
python train.py -g 2 --epoch 3   --lr_step_size 2   \
--lr  0.0005    --batch_size  4     \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/vectordb/output-group/model_6*4_yelp.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/vectordb/output-group/config_6*4_yelp.json  \
--data_paths  yelp"



python train.py -g 2 --epoch 3   --lr_step_size 2   \
--lr  0.001    --batch_size  4     \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/vectordb/output/model_6*4_amazon.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/vectordb/output/config_6*4_amazon.json  \
--data_paths  amazon


merge_retriever(
    [
        "/dataset/zhuotaotian/bhpeng/vectordb/output/config_6*4_ag_news.json",
        "/dataset/zhuotaotian/bhpeng/vectordb/output/config_6*4_amazon.json",
        "/dataset/zhuotaotian/bhpeng/vectordb/output/config_6*4_dbpedia.json",
        "/dataset/zhuotaotian/bhpeng/vectordb/output/config_6*4_yahoo.json",
        "/dataset/zhuotaotian/bhpeng/vectordb/output/config_6*4_yelp.json",
    ],
    [
        "/dataset/zhuotaotian/bhpeng/vectordb/output/model_6*4_ag_news.pth",
        "/dataset/zhuotaotian/bhpeng/vectordb/output/model_6*4_amazon.pth",
        "/dataset/zhuotaotian/bhpeng/vectordb/output/model_6*4_dbpedia.pth",
        "/dataset/zhuotaotian/bhpeng/vectordb/output/model_6*4_yahoo.pth",
        "/dataset/zhuotaotian/bhpeng/vectordb/output/model_6*4_yelp.pth",
    ],
    "/dataset/zhuotaotian/bhpeng/vectordb/output/config_6*20_all.json",
    "/dataset/zhuotaotian/bhpeng/vectordb/output/model_6*20_all.pth",
    config_same_attr=['weight_topk', 'groups', 'similarity_type', ],
    config_merge_attr=['pool_size'],
)

merge_retriever(
    [
        "/dataset/zhuotaotian/bhpeng/vectordb/output-group-1-nomask/config_1*24_ag_news.json",
        "/dataset/zhuotaotian/bhpeng/vectordb/output-group-1-nomask/config_1*24_amazon.json",
        "/dataset/zhuotaotian/bhpeng/vectordb/output-group-1-nomask/config_1*24_dbpedia.json",
        "/dataset/zhuotaotian/bhpeng/vectordb/output-group-1-nomask/config_1*24_yahoo.json",
        "/dataset/zhuotaotian/bhpeng/vectordb/output-group-1-nomask/config_1*24_yelp.json",
    ],
    [
        "/dataset/zhuotaotian/bhpeng/vectordb/output-group-1-nomask/model_1*24_ag_news.pth",
        "/dataset/zhuotaotian/bhpeng/vectordb/output-group-1-nomask/model_1*24_amazon.pth",
        "/dataset/zhuotaotian/bhpeng/vectordb/output-group-1-nomask/model_1*24_dbpedia.pth",
        "/dataset/zhuotaotian/bhpeng/vectordb/output-group-1-nomask/model_1*24_yahoo.pth",
        "/dataset/zhuotaotian/bhpeng/vectordb/output-group-1-nomask/model_1*24_yelp.pth",
    ],
    "/dataset/zhuotaotian/bhpeng/vectordb/output-group-1-nomask/config_1*120_all.json",
    "/dataset/zhuotaotian/bhpeng/vectordb/output-group-1-nomask/model_1*120_all.pth",
    config_same_attr=['weight_topk', 'groups', 'similarity_type', ],
    config_merge_attr=['pool_size'],
)

merge_retriever(
    [
        "/dataset/zhuotaotian/bhpeng/vectordb/output-g3-p4/config_3*4_ag_news.json",
        "/dataset/zhuotaotian/bhpeng/vectordb/output-g3-p4/config_3*4_amazon.json",
        "/dataset/zhuotaotian/bhpeng/vectordb/output-g3-p4/config_3*4_dbpedia.json",
        "/dataset/zhuotaotian/bhpeng/vectordb/output-g3-p4/config_3*4_yahoo.json",
        "/dataset/zhuotaotian/bhpeng/vectordb/output-g3-p4/config_3*4_yelp.json",
    ],
    [
        "/dataset/zhuotaotian/bhpeng/vectordb/output-g3-p4/model_3*4_ag_news.pth",
        "/dataset/zhuotaotian/bhpeng/vectordb/output-g3-p4/model_3*4_amazon.pth",
        "/dataset/zhuotaotian/bhpeng/vectordb/output-g3-p4/model_3*4_dbpedia.pth",
        "/dataset/zhuotaotian/bhpeng/vectordb/output-g3-p4/model_3*4_yahoo.pth",
        "/dataset/zhuotaotian/bhpeng/vectordb/output-g3-p4/model_3*4_yelp.pth",
    ],
    "/dataset/zhuotaotian/bhpeng/vectordb/output-group/config_3*20_all.json",
    "/dataset/zhuotaotian/bhpeng/vectordb/output-group/model_3*20_all.pth",
    config_same_attr=['weight_topk', 'groups', 'similarity_type', ],
    config_merge_attr=['pool_size'],
)

2: 8.7
1: 10.2
3: 10.5
4: 14.3



import torch
data = torch.load("/dataset/zhuotaotian/bhpeng/SLM-weight-ab/outputs/all/amazon/checkpoint-17970/pytorch_model.bin")

yelp = torch.load("/dataset/zhuotaotian/bhpeng/SLM-weight-ab/outputs/task-13/yelp/checkpoint-17970/pytorch_model.bin")

dbpedia = torch.load("/dataset/zhuotaotian/bhpeng/SLM-weight-ab/outputs/task-12/dbpedia/checkpoint-14376/pytorch_model.bin")

yahoo = torch.load("/dataset/zhuotaotian/bhpeng/SLM-weight-ab/outputs/task-11/yahoo/checkpoint-17970/pytorch_model.bin")

ag_news = torch.load("/dataset/zhuotaotian/bhpeng/SLM-weight-ab/outputs/task-10/ag_news/checkpoint-17970/pytorch_model.bin")

amazon = torch.load("/dataset/zhuotaotian/bhpeng/SLM-weight-ab/outputs/task-9/amazon/checkpoint-17970/pytorch_model.bin")


data["model.classifier.5.weight"] = yelp['model.classifier.5.weight']
data["model.classifier.5.bias"] = yelp['model.classifier.5.bias']

data["model.classifier.4.weight"] = ag_news['model.classifier.4.weight']
data["model.classifier.4.bias"] = ag_news['model.classifier.4.bias']

data["model.classifier.10.weight"] = yahoo['model.classifier.10.weight']
data["model.classifier.10.bias"] = yahoo['model.classifier.10.bias']

data["model.classifier.14.weight"] = dbpedia['model.classifier.14.weight']
data["model.classifier.14.bias"] = dbpedia['model.classifier.14.bias']

data['retriever.weight_offset'] = yelp['retriever.weight_offset']
data['retriever.weight_offset'][0:4, ...] = ag_news['retriever.weight_offset'][0:4, ...]
data['retriever.weight_offset'][8:12, ...] = dbpedia['retriever.weight_offset'][8:12, ...]
data['retriever.weight_offset'][12:16, ...] = yahoo['retriever.weight_offset'][12:16, ...]

torch.save(data, "/dataset/zhuotaotian/bhpeng/SLM-weight-ab/outputs/all/amazon/checkpoint-17970/pytorch_model.bin")





sc delete amazon
sc delete ag-news
sc delete dbpedia
sc delete yahoo
sc delete yelp

sc create --name amazon --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 4 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb && \
python train.py \
-g 4 --epoch 1   --lr_step_size 2   \
--lr  0.0005    --batch_size  4     \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/model_1*1_amazon.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/config_1*1_amazon.json  \
--data_paths  amazon"

sc create --name ag-news --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 4 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb && \
python train.py \
-g 4 --epoch 1   --lr_step_size 2   \
--lr  0.0005    --batch_size  4     \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/model_1*1_ag_news.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/config_1*1_ag_news.json  \
--data_paths  ag"

sc create --name dbpedia --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 4 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb && \
python train.py \
-g 4 --epoch 1   --lr_step_size 2   \
--lr  0.0005    --batch_size  4     \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/model_1*1_dbpedia.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/config_1*1_dbpedia.json  \
--data_paths  dbpedia"

sc create --name yahoo --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 4 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb && \
python train.py \
-g 4 --epoch 1   --lr_step_size 2   \
--lr  0.0005    --batch_size  4     \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/model_1*1_yahoo.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/config_1*1_yahoo.json  \
--data_paths  yahoo"

sc create --name yelp --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 4 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb && \
python train.py \
-g 4 --epoch 1   --lr_step_size 2   \
--lr  0.0005    --batch_size  4     \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/model_1*1_yelp.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/config_1*1_yelp.json  \
--data_paths  yelp"
