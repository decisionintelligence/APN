#!/bin/bash

# =========================================================================
#             HYPE-Former 在 P12 数据集上的超参数搜索脚本
# =========================================================================

# --- 1. GPU 配置 ---
GPU_IDS="0,1,2,3,4,5,6,7"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}
job_count=0

echo "🚀 将在 ${NUM_GPUS} 个GPU (${GPUS[*]}) 上并行执行 HYPE-Former 的超参数搜索任务。"

# --- 2. 固定参数 ---
model_name="HYPEFormer"
dataset_root_path="storage/datasets/P12"
dataset_name="P12"
seq_len=36
pred_len=12
enc_in=36
dec_in=36
c_out=36

# --- 3. HYPE-Former 超参数搜索空间 ---
# 模型结构相关
d_models=(64 128)
n_heads_options=(4 8)
n_layers=(1 2 3)
d_ff_multipliers=(2 4)
dropouts=(0.1 0.2 0.3)

# HYPE-Former 核心参数
patch_lens=(8 12 16)
strides=(8 12 16) #

# 训练相关
batch_sizes=(16 32)
lrs=(0.001 0.0005)

# --- 4. 随机搜索设置 ---
TOTAL_RUNS=1296
echo "🚀 开始 HYPE-Former 在 P12 上的随机搜索，总共将运行 ${TOTAL_RUNS} 组实验..."

# --- 5. 随机搜索主循环 ---
for (( i=1; i<=${TOTAL_RUNS}; i++ )); do

  # --- 随机采样超参数 ---
  dm=${d_models[$((RANDOM % ${#d_models[@]}))]}
  nl=${n_layers[$((RANDOM % ${#n_layers[@]}))]}
  dff_m=${d_ff_multipliers[$((RANDOM % ${#d_ff_multipliers[@]}))]}
  dff=$((dm * dff_m))
  dp=${dropouts[$((RANDOM % ${#dropouts[@]}))]}
  bs=${batch_sizes[$((RANDOM % ${#batch_sizes[@]}))]}
  lr=${lrs[$((RANDOM % ${#lrs[@]}))]}

  # 采样 patch_len 和 stride，并确保 stride <= patch_len
  pl=${patch_lens[$((RANDOM % ${#patch_lens[@]}))]}
  st=$pl
  # 约束条件: 确保 n_heads 能被 d_model 整除
  while true; do
    nh=${n_heads_options[$((RANDOM % ${#n_heads_options[@]}))]}
    if [ $((dm % nh)) -eq 0 ]; then
      break
    fi
  done

  # --- GPU 分配 (轮询机制) ---
  gpu_idx=$((job_count % NUM_GPUS))
  gpu_id=${GPUS[$gpu_idx]}

  # --- 构建唯一的模型ID, 用于日志和模型保存 ---
  model_id="${model_name}_${dataset_name}_sl${seq_len}_plen${pred_len}_dm${dm}_nh${nh}_nl${nl}_pl${pl}_st${st}_dff${dff}_dp${dp}_lr${lr}_bs${bs}_run${i}"

  echo "-----------------------------------------------------------------------"
  echo "📈 启动任务 [${i}/${TOTAL_RUNS}] -> 分配至 GPU ${gpu_id}"
  echo "   Config: d_model=${dm}, n_heads=${nh}, n_layers=${nl}, d_ff=${dff}, patch_len=${pl}, stride=${st}"
  echo "   Training: bs=${bs}, lr=${lr}, dropout=${dp}"
  echo "   Model ID: ${model_id}"
  echo "-----------------------------------------------------------------------"

  # --- 在后台启动训练任务 ---
  CUDA_VISIBLE_DEVICES=${gpu_id} python main.py \
    --is_training 1 \
    --model_id "$model_id" \
    --model_name "$model_name" \
    --d_model "$dm" \
    --n_heads "$nh" \
    --n_layers "$nl" \
    --d_ff "$dff" \
    --dropout "$dp" \
    --patch_len "$pl" \
    --patch_stride "$st" \
    --dataset_root_path "$dataset_root_path" \
    --dataset_name "$dataset_name" \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --enc_in "$enc_in" \
    --dec_in "$dec_in" \
    --c_out "$c_out" \
    --loss "MSE" \
    --train_epochs 300 \
    --patience 10 \
    --val_interval 1 \
    --itr 5 \
    --batch_size "$bs" \
    --learning_rate "$lr" \
    --use_multi_gpu 0 &

  job_count=$((job_count + 1))
  sleep 2

  if [[ $(jobs -r -p | wc -l) -ge $NUM_GPUS ]]; then
      wait -n
  fi

done

echo "✅ 所有 ${TOTAL_RUNS} 组任务已启动，等待最后批次的任务执行完毕..."
wait
echo "🎉🎉🎉 HYPE-Former 在 P12 上的随机搜索任务已全部完成！🎉🎉🎉"