#!/bin/bash

# =====================================================================================
#             TAC-Mixer 在 HumanActivity 上的超参数搜索脚本 (高效GPU利用)
# =====================================================================================

# --- 1. GPU 配置 ---
# 在这里指定你想使用的GPU编号，用逗号隔开
GPU_IDS="0,1,2,3,4,5,6,7"

# 将GPU_IDS字符串转换为bash数组
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

# 使用关联数组来追踪每个GPU上运行的进程ID (PID)
declare -A gpu_pids
for gpu in "${GPUS[@]}"; do
    gpu_pids[$gpu]=""
done

echo "🚀 将在 ${NUM_GPUS} 个GPU (${GPUS[*]}) 上并行执行 TAC-Mixer 任务，采用动态调度策略。"

# --- 2. 固定参数 (已根据 HumanActivity 数据集调整) ---
model_name="TAC-Mixer"
dataset_root_path="storage/datasets/HumanActivity"
dataset_name="HumanActivity"

# 为了与 HyperIMTS 公平比较，采用其设置
seq_len=3000
pred_len=300

# 根据 HumanActivity 数据集统计表格 (# Variable = 12)
enc_in=12
dec_in=12
c_out=12

# --- 3. 超参数搜索空间 (已根据 HumanActivity 数据集特性调整) ---
# d_model: 变量数减少，可以适当降低模型维度
d_models=(32 48 64)
# n_layers: 2-4层是比较通用的范围
n_layers_options=(2 3 4)
# n_heads: 保持通用选项，需确保 d_model % n_heads == 0
n_heads_options=(4 8 16)
# batch_size: seq_len 变得非常长(3000)，必须大幅减小 batch_size 防止OOM
batch_sizes=(4 8 16)
# lr: 参照 HyperIMTS 使用的 0.001，在其附近搜索
lrs=(0.005 0.0075 0.01)
# dropout: 保持通用范围
dropouts=(0.3 0.4 0.5)
# tac_patch_num: seq_len 很长，需要更多的 patch 来切分时间轴
tac_patch_nums=(50 75 100 125)
# tac_mixer_dims_p: 与 patch_num 相关，保持通用范围
tac_mixer_dims_p=(32 64)
# tac_mixer_dims_c: 变量数(12)较少，可以减小 C-Mixer 的隐藏维度
tac_mixer_dims_c=(8 16)
# tac_decoder_k: 解码器局部上下文范围，保持不变
tac_decoder_ks=(0 1 2)


# --- 4. 随机搜索设置 ---
# 可以根据需要调整总运行次数
TOTAL_RUNS=512

# --- 动态寻找空闲GPU的函数 (与P12脚本完全相同，无需修改) ---
find_free_gpu() {
    local free_gpu_id=""
    while [[ -z "$free_gpu_id" ]]; do
        for gpu_id in "${GPUS[@]}"; do
            pid=${gpu_pids[$gpu_id]}
            if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
                free_gpu_id=$gpu_id
                unset gpu_pids[$gpu_id]
                break
            fi
        done

        if [[ -z "$free_gpu_id" ]]; then
            wait -n
        fi
    done
    echo "$free_gpu_id"
}


echo "🚀 开始 TAC-Mixer 在 HumanActivity 上的随机搜索，总共将运行 ${TOTAL_RUNS} 组实验..."

# --- 5. 随机搜索主循环 ---
for (( i=1; i<=${TOTAL_RUNS}; i++ )); do

  # --- 随机采样超参数 ---
  dm=${d_models[$((RANDOM % ${#d_models[@]}))]}
  nl=${n_layers_options[$((RANDOM % ${#n_layers_options[@]}))]}
  bs=${batch_sizes[$((RANDOM % ${#batch_sizes[@]}))]}
  lr=${lrs[$((RANDOM % ${#lrs[@]}))]}
  dp=${dropouts[$((RANDOM % ${#dropouts[@]}))]}
  pnum=${tac_patch_nums[$((RANDOM % ${#tac_patch_nums[@]}))]}
  dp_dim=${tac_mixer_dims_p[$((RANDOM % ${#tac_mixer_dims_p[@]}))]}
  dc_dim=${tac_mixer_dims_c[$((RANDOM % ${#tac_mixer_dims_c[@]}))]}
  k_dec=${tac_decoder_ks[$((RANDOM % ${#tac_decoder_ks[@]}))]}
  dff=$((dm * 4))
  while true; do
    nh=${n_heads_options[$((RANDOM % ${#n_heads_options[@]}))]}
    if [ $((dm % nh)) -eq 0 ]; then
      break
    fi
  done

  # --- GPU 动态分配 ---
  gpu_id=$(find_free_gpu)

  # --- 构建唯一的模型ID ---
  model_id="${model_name}_${dataset_name}_sl${seq_len}_pl${pred_len}_dm${dm}_nl${nl}_nh${nh}_pnum${pnum}_dpd${dp_dim}_dcd${dc_dim}_k${k_dec}_dp${dp}_lr${lr}_bs${bs}_run${i}"

  echo "-----------------------------------------------------------------------"
  echo "📈 启动任务 [${i}/${TOTAL_RUNS}] -> 动态分配至空闲 GPU ${gpu_id}"
  echo "   Arch: d_model=${dm}, n_layers=${nl}, n_heads=${nh}, dropout=${dp}"
  echo "   TAC:  p_num=${pnum}, d_p=${dp_dim}, d_c=${dc_dim}, k_dec=${k_dec}"
  echo "   Train: bs=${bs}, lr=${lr}"
  echo "   Model ID: ${model_id}"
  echo "-----------------------------------------------------------------------"

  # --- 在后台启动训练任务 ---
  (
    CUDA_VISIBLE_DEVICES=${gpu_id} python main.py \
      --is_training 1 \
      --model_id "$model_id" \
      --model_name "$model_name" \
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
      --patience 5 \
      --val_interval 1 \
      --itr 1 \
      --batch_size "$bs" \
      --learning_rate "$lr" \
      --d_model "$dm" \
      --n_layers "$nl" \
      --n_heads "$nh" \
      --d_ff "$dff" \
      --dropout "$dp" \
      --tac_patch_num "$pnum" \
      --tac_mixer_hidden_dim_p "$dp_dim" \
      --tac_mixer_hidden_dim_c "$dc_dim" \
      --tac_decoder_context_k "$k_dec"
  ) &

  # --- 记录新任务的 PID ---
  new_pid=$!
  gpu_pids[$gpu_id]=$new_pid
  echo "   任务已启动，PID: ${new_pid}，已绑定至 GPU ${gpu_id}"

  sleep 1

done

echo "✅ 所有 ${TOTAL_RUNS} 组任务已启动，等待最后批次的任务执行完毕..."
wait
echo "🎉🎉🎉 TAC-Mixer 在 HumanActivity 上的随机搜索任务已全部完成！🎉🎉🎉"