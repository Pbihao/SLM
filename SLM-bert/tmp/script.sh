   # ++++++++++++++++++++++++ Train g-1 t-1 p-1 keys ++++++++++++++++++++++++++++++++++++

sc create --name amazon --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 1 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb && \
python train.py \
-g 1 --epoch 3   --lr_step_size 2   \
--lr  0.001    --batch_size  4     \
--log_interval 30 \ 
--save_path  /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/model_1*1_amazon.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/config_1*1_amazon.json  \
--data_paths  amazon"

sc create --name ag-news --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 1 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb && \
python train.py \
-g 1 --epoch 3   --lr_step_size 2   \
--lr  0.001    --batch_size  4     \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/model_1*1_ag_news.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/config_1*1_ag_news.json  \
--data_paths  ag"

sc create --name dbpedia --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 1 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb && \
python train.py \
-g 1 --epoch 3   --lr_step_size 2   \
--lr  0.001    --batch_size  4     \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/model_1*1_dbpedia.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/config_1*1_dbpedia.json  \
--data_paths  dbpedia"

sc create --name yahoo --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 1 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb && \
python train.py \
-g 1 --epoch 3   --lr_step_size 2   \
--lr  0.001    --batch_size  4     \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/model_1*1_yahoo.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/config_1*1_yahoo.json  \
--data_paths  yahoo"

sc create --name yelp --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 1 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb && \
python train.py \
-g 1 --epoch 3   --lr_step_size 2   \
--lr  0.001    --batch_size  4     \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/model_1*1_yelp.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/config_1*1_yelp.json  \
--data_paths  yelp"





merge_retriever(
    [
        "/dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/config_1*1_ag_news.json",
        "/dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/config_1*1_amazon.json",
        "/dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/config_1*1_dbpedia.json",
        "/dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/config_1*1_yahoo.json",
        "/dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/config_1*1_yelp.json",
    ],
    [
        "/dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/model_6*4_ag_news.pth",
        "/dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/model_6*4_amazon.pth",
        "/dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/model_6*4_dbpedia.pth",
        "/dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/model_6*4_yahoo.pth",
        "/dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/model_6*4_yelp.pth",
    ],
    "/dataset/zhuotaotian/bhpeng/vectordb/output/config_6*20_all.json",
    "/dataset/zhuotaotian/bhpeng/vectordb/output/model_6*20_all.pth",
    config_same_attr=['weight_topk', 'groups', 'similarity_type', ],
    config_merge_attr=['pool_size'],
)








sc create --name task-9 --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 4 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/SLM-weight-ab && \
torchrun \
--nproc-per-node=4 \
--master-port 4325 \
train.py \
--lr 0.0002 \
--batch_size 8 \
--task_id 9 \
--output_dir ./outputs/task-9 "

sc create --name task-10 --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 4 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/SLM-weight-ab && \
torchrun \
--nproc-per-node=4 \
--master-port 4325 \
train.py \
--lr 0.0002 \
--batch_size 8 \
--task_id 10 \
--output_dir ./outputs/task-10 "

sc create --name task-11 --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 4 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/SLM-weight-ab && \
torchrun \
--nproc-per-node=4 \
--master-port 4325 \
train.py \
--lr 0.0002 \
--batch_size 8 \
--task_id 11 \
--output_dir ./outputs/task-11 "

sc create --name task-12 --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 4 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/SLM-weight-ab && \
torchrun \
--nproc-per-node=4 \
--master-port 4325 \
train.py \
--lr 0.0002 \
--batch_size 8 \
--task_id 12 \
--output_dir ./outputs/task-12 "


sc create --name task-13 --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 4 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/SLM-weight-ab && \
torchrun \
--nproc-per-node=4 \
--master-port 4325 \
train.py \
--lr 0.0002 \
--batch_size 8 \
--task_id 13 \
--model_path /dataset/zhuotaotian/bhpeng/SLM-weight-ab/outputs/task-9/amazon/checkpoint-17970 \
--output_dir ./outputs/task-13 "