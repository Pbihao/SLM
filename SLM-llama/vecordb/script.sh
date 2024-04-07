python train.py \
-g 2 --epoch 2   --lr_step_size 2   \
--lr  0.001    --batch_size  16     \
--log_interval 30 \
--save_path    /dataset/bohaopeng/research/llm/SLM-llama/vectordb/output/model_6*6_medical.pth  \
--config_path  /dataset/bohaopeng/research/llm/SLM-llama/vectordb/output/config_6*6_medical.json  \
--data_paths  medical


sc create --name medical --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 4 \
--cmd "cd /dataset/bohaopeng && . .bashrc && conda activate llm && cd /dataset/bohaopeng/research/llm/SLM-llama/vectordb && \
python train.py \
-g 4 --epoch 2   --lr_step_size 2   \
--lr  0.001    --batch_size  16     \
--log_interval 30 \
--save_path    /dataset/bohaopeng/research/llm/SLM-llama/vectordb/output/model_6*6_medical.pth  \
--config_path  /dataset/bohaopeng/research/llm/SLM-llama/vectordb/output/config_6*6_medical.json  \
--data_paths  medical"

sc create --name ag-news --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 2 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/SLM-t5/vectordb && \
python train.py \
-g 2 --epoch 3   --lr_step_size 2   \
--lr  0.001    --batch_size  4     \
--k_samples  16   \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/SLM-t5/vectordb/output_16_samples/model_6*4_ag_news.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/SLM-t5/vectordb/output_16_samples/config_6*4_ag_news.json  \
--data_paths  ag_news"

sc create --name dbpedia --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 2 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/SLM-t5/vectordb && \
python train.py \
-g 2 --epoch 3   --lr_step_size 8   \
--lr  0.001    --batch_size  4     \
--k_samples  16   \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/SLM-t5/vectordb/output_16_samples/model_6*4_dbpedia.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/SLM-t5/vectordb/output_16_samples/config_6*4_dbpedia.json  \
--data_paths  dbpedia"

sc create --name yahoo --image "harbor.smoa.cc/public/xsemseg:v1.4" --arch ampere \
--gpu 2 \
--cmd "cd /dataset/zhuotaotian && . .bashrc && conda activate llm && cd /dataset/zhuotaotian/bhpeng/SLM-t5/vectordb && \
python train.py \
-g 2 --epoch 3   --lr_step_size 2   \
--lr  0.001    --batch_size  4     \
--k_samples  16   \
--log_interval 30 \
--save_path  /dataset/zhuotaotian/bhpeng/SLM-t5/vectordb/output_16_samples/model_6*4_yahoo.pth  \
--config_path  /dataset/zhuotaotian/bhpeng/SLM-t5/vectordb/output_16_samples/config_6*4_yahoo.json  \
--data_paths  yahoo"


merge_retriever(
    config_paths=["/dataset/zhuotaotian/bhpeng/SLM-t5/vectordb/output_16_samples/config_6*4_ag_news.json", "/dataset/zhuotaotian/bhpeng/SLM-t5/vectordb/output_16_samples/config_6*4_amazon.json", "/dataset/zhuotaotian/bhpeng/SLM-t5/vectordb/output_16_samples/config_6*4_dbpedia.json", "/dataset/zhuotaotian/bhpeng/SLM-t5/vectordb/output_16_samples/config_6*4_yahoo.json"],
    state_dict_paths=['/dataset/zhuotaotian/bhpeng/SLM-t5/vectordb/output_16_samples/model_6*4_ag_news.pth', '/dataset/zhuotaotian/bhpeng/SLM-t5/vectordb/output_16_samples/model_6*4_amazon.pth', '/dataset/zhuotaotian/bhpeng/SLM-t5/vectordb/output_16_samples/model_6*4_dbpedia.pth', '/dataset/zhuotaotian/bhpeng/SLM-t5/vectordb/output_16_samples/model_6*4_yahoo.pth'],
    new_config_path="/dataset/zhuotaotian/bhpeng/SLM-t5/vectordb/output_16_samples/config_6*16_all.json",
    new_state_dict_path="/dataset/zhuotaotian/bhpeng/SLM-t5/vectordb/output_16_samples/model_6*16_all.pth",
    config_same_attr=['weight_topk', 'groups', 'similarity_type', ],
    config_merge_attr=['pool_size'],
)