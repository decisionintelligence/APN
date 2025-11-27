#!/bin/bash

echo "Starting CRU Efficiency Test Script..."

# ===================================================================================
#                                 STEP 1: 配置
# ===================================================================================

# 在这里填入您想要使用的 GPU ID
export CUDA_VISIBLE_DEVICES=0

# 模型名称
model_name="CRU"

# 数据集和任务的固定参数 (与APN测试保持一致)
dataset_root_path="storage/datasets/USHCN"
dataset_name="USHCN"
features="M"
seq_len=150
pred_len=3
enc_in=5
dec_in=5
c_out=5
itr=1
# --- 从 CRU.sh 中提取的核心超参数 ---
d_model=50
cru_ts=0.2  # CRU 的特殊参数
bs=32       # 使用训练时的 batch_size 进行时间/内存测试
lr=0.001

# 为测试创建一个唯一的 model_id
model_id="${model_name}_${dataset_name}_sl${seq_len}_pl${pred_len}_dm${d_model}_efficiency"

# 构建基础命令
base_command="python main.py \
    --is_training 0 \
    --model_id \"$model_id\" \
    --model_name \"$model_name\" \
    --dataset_root_path \"$dataset_root_path\" \
    --dataset_name \"$dataset_name\" \
    --features \"$features\" \
    --seq_len \"$seq_len\" \
    --pred_len \"$pred_len\" \
    --enc_in \"$enc_in\" \
    --dec_in \"$dec_in\" \
    --c_out \"$c_out\" \
    --itr \"$itr\" \
    --loss \"MSE\" \
    --batch_size \"$bs\" \
    --learning_rate \"$lr\" \
    --d_model \"$d_model\" \
    --cru_ts \"$cru_ts\""

# ===================================================================================
#                        STEP 2: 选择并执行测试 (一次只运行一个)
# ===================================================================================

# --- 测试1: 参数量和计算量 (FLOPs) ---
#echo "Running Parameters and FLOPs Test for CRU..."
#eval "${base_command} --test_flop 1"

# --- 测试2: GPU峰值显存占用 ---
# echo "Running GPU Peak Memory Test for CRU..."
# eval "${base_command} --test_gpu_memory 1"

# --- 测试3: 训练速度 ---
# echo "Running Training Step Time Test for CRU..."
# eval "${base_command} --test_train_time 1"

# --- 测试4: 推理速度 ---
 echo "Running Inference Step Time Test for CRU..."
 eval "${base_command} --test_inference_time 1"

echo "All tests finished for CRU."