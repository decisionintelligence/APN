#!/bin/bash

# ======================= [ 用户配置区域 ] =======================
# 在下面的括号中填入您希望使用的GPU编号，用空格隔开。
# 示例:
#   - 使用 GPU 2 和 3: GPUS_TO_USE=(2 3)
#   - 只使用 GPU 4:    GPUS_TO_USE=(4)
#   - 使用 GPU 0, 1, 4, 7: GPUS_TO_USE=(0 1 4 7)
GPUS_TO_USE=(0 1 5 6 7)
# =================================================================

# --- GPU并行设置 (自动计算) ---
# NUM_GPUS 将根据您上面列表中的GPU数量自动确定
NUM_GPUS=${#GPUS_TO_USE[@]}
job_count=0

# --- 固定参数 (根据MIMIC-III和HyperIMTS参考进行调整) ---
model_name="$(basename "$(dirname "$(readlink -f "$0")")")"
dataset_root_path=storage/datasets/MIMIC_III
dataset_name=$(basename "$0" .sh)
seq_len=72
pred_len=24
enc_in=96
dec_in=96
c_out=96

# --- 针对MIMIC-III优化的超参数搜索空间 ---
d_models=(64 128)
n_heads_options=(4 8)
batch_sizes=(16 32)
lrs=(0.001)
num_kernels=(8 16)
num_intra_layers=(1 2 3)
dropouts=(0.1 0.2 0.3)

# --- 随机搜索设置 ---
TOTAL_RUNS=128

echo "🚀 开始 SPECTRON 在 MIMIC-III 上的【指定GPU】随机搜索..."
echo "   将在 ${NUM_GPUS} 个指定GPU上运行: (${GPUS_TO_USE[*]})"
echo "   总共将启动 ${TOTAL_RUNS} 组实验..."

# --- 随机搜索循环 ---
for (( i=0; i<${TOTAL_RUNS}; i++ )); do

  # --- 随机采样超参数 ---
  dm=${d_models[$((RANDOM % ${#d_models[@]}))]}
  bs=${batch_sizes[$((RANDOM % ${#batch_sizes[@]}))]}
  lr=${lrs[$((RANDOM % ${#lrs[@]}))]}
  nk=${num_kernels[$((RANDOM % ${#num_kernels[@]}))]}
  nil=${num_intra_layers[$((RANDOM % ${#num_intra_layers[@]}))]}
  dp=${dropouts[$((RANDOM % ${#dropouts[@]}))]}

  while true; do
    nh=${n_heads_options[$((RANDOM % ${#n_heads_options[@]}))]}
    if [ $((dm % nh)) -eq 0 ]; then
      break
    fi
  done

  # --- [核心修改] 从指定的GPU列表中分配GPU ---
  # 1. 计算当前任务在列表中的索引
  index=$((job_count % NUM_GPUS))
  # 2. 从列表中获取实际的GPU编号
  gpu_id=${GPUS_TO_USE[$index]}

  model_id="${model_name}_${dataset_name}_sl${seq_len}_pl${pred_len}_dm${dm}_nh${nh}_nk${nk}_nil${nil}_dp${dp}_lr${lr}_bs${bs}_run${i}"

  echo "----------------------------------------------------"
  echo "📈 启动随机搜索任务 [${job_count}] -> 分配至指定GPU ${gpu_id}"
  echo "   Config: d_model=${dm}, n_heads=${nh}, bs=${bs}, lr=${lr}, kernels=${nk}, intra_layers=${nil}, dropout=${dp}"
  echo "   Model ID: ${model_id}"
  echo "----------------------------------------------------"

  # 在后台启动训练任务，并设置CUDA_VISIBLE_DEVICES为我们从列表中选出的gpu_id
  CUDA_VISIBLE_DEVICES=${gpu_id} python main.py \
    --is_training 1 \
    --model_id "$model_id" \
    --model_name "$model_name" \
    --d_model "$dm" \
    --n_heads "$nh" \
    --dropout "$dp" \
    --spectron_num_kernels "$nk" \
    --spectron_d_max 5.0 \
    --spectron_num_intra_layers "$nil" \
    --dataset_root_path "$dataset_root_path" \
    --dataset_name "$dataset_name" \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --enc_in "$enc_in" \
    --dec_in "$dec_in" \
    --c_out "$c_out" \
    --loss "MSE" \
    --train_epochs 50 \
    --patience 1 \
    --val_interval 1 \
    --itr 1 \
    --batch_size "$bs" \
    --learning_rate "$lr" \
    --use_multi_gpu 0 &

  job_count=$((job_count + 1))
  sleep 2

  if [[ $(jobs -r -p | wc -l) -ge $NUM_GPUS ]]; then
      wait -n
  fi
done

echo "✅ 所有任务已启动，等待最后批次的任务执行完毕..."
wait
echo "🎉🎉🎉 MIMIC-III 随机搜索任务已全部完成！🎉🎉🎉"