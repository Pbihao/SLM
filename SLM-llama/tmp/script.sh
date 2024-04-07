CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun \
--nproc-per-node=4 \
--master-port 4325 \
train.py \
--lr 0.001 \
--batch_size 2 \
--epoch 3 \
--deepspeed "./default_offload_opt_param.json" \
--tqdm \
--output_dir ./outputs/history_1e3

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
torchrun \
--nproc-per-node=4 \
--master-port 4325 \
train.py \
--lr 0.0001 \
--batch_size 2 \
--epoch 3 \
--deepspeed "./default_offload_opt_param.json" \
--tqdm \
--output_dir ./outputs/history_1e4


CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
torchrun \
--nproc-per-node=8 \
--master-port 4325 \
train.py \
--lr 0.0002 \
--batch_size 2 \
--epoch 6 \
--deepspeed "./default_offload_opt_param.json" \
--tqdm \
--task medical \
--output_dir ./outputs/medical_oc


CUDA_VISIBLE_DEVICES=0 \
python test.py \
--pbh_dive 0 \
--task medical \
--output_dir "./tmp/results" \
--model_path "/data/bhpeng/SLM-llama/outputs/finance_history_medical/pytorch_model.bin"

CUDA_VISIBLE_DEVICES=0 \
python test.py \
--pbh_dive 0 \
--task medical \
--output_dir "./tmp/results" \
--model_path "/data/bhpeng/SLM-llama/outputs/finance_history_medical/pytorch_model.bin"

CUDA_VISIBLE_DEVICES=1 \
python test.py \
--pbh_dive 1 \
--task medical \
--output_dir "./tmp/results" \
--model_path "/data/bhpeng/SLM-llama/outputs/finance_history_medical/pytorch_model.bin"

CUDA_VISIBLE_DEVICES=2 \
python test.py \
--pbh_dive 2 \
--task medical \
--output_dir "./tmp/results" \
--model_path "/data/bhpeng/SLM-llama/outputs/finance_history_medical/pytorch_model.bin"

CUDA_VISIBLE_DEVICES=3 \
python test.py \
--pbh_dive 3 \
--task medical \
--output_dir "./tmp/results" \
--model_path "/data/bhpeng/SLM-llama/outputs/finance_history_medical/pytorch_model.bin"

CUDA_VISIBLE_DEVICES=4 \
python test.py \
--pbh_dive 4 \
--task medical \
--output_dir "./tmp/results" \
--model_path "/data/bhpeng/SLM-llama/outputs/finance_history_medical/pytorch_model.bin"

CUDA_VISIBLE_DEVICES=5 \
python test.py \
--pbh_dive 5 \
--task medical \
--output_dir "./tmp/results" \
--model_path "/data/bhpeng/SLM-llama/outputs/finance_history_medical/pytorch_model.bin"

CUDA_VISIBLE_DEVICES=6 \
python test.py \
--pbh_dive 6 \
--task medical \
--output_dir "./tmp/results" \
--model_path "/data/bhpeng/SLM-llama/outputs/finance_history_medical/pytorch_model.bin"


CUDA_VISIBLE_DEVICES=7 \
python test.py \
--pbh_dive 7 \
--task medical \
--output_dir "./tmp/results" \
--model_path "/data/bhpeng/SLM-llama/outputs/finance_history_medical/pytorch_model.bin"


python test.py --score --task medical --result_path /data/bhpeng/SLM-llama/tmp/results/medical/result.json