#!/bin/bash

# =====================================================================================
#             改进版 TAC-Mixer 在 P12 上的超参数搜索脚本 (高效GPU利用)
# =====================================================================================

# --- 1. GPU 配置 ---
# 在这里指定你想使用的GPU编号，用逗号隔开，例如 "0,1,2,3" 或 "4,5,6,7"
GPU_IDS="0,1,2,3,4,5,6,7"

# 将GPU_IDS字符串转换为bash数组
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

# [核心改进] 使用关联数组来追踪每个GPU上运行的进程ID (PID)
# 键: GPU ID, 值: 任务的 PID
declare -A gpu_pids
for gpu in "${GPUS[@]}"; do
    gpu_pids[$gpu]=""
done

echo "🚀 将在 ${NUM_GPUS} 个GPU (${GPUS[*]}) 上并行执行 TAC-Mixer 任务，采用动态调度策略。"

# --- 2. 固定参数 (请根据您的数据集进行调整) ---
model_name="TAC-Mixer"
dataset_root_path="storage/datasets/P12"
dataset_name="P12"
seq_len=36
pred_len=3
enc_in=36
dec_in=36
c_out=36

d_models=(32 48 64 96)
n_layers_options=(3 4 5)
n_heads_options=(4 8 16)
batch_sizes=(8 16)
lrs=(0.005 0.0075 0.01)
dropouts=(0.3 0.4 0.5)
tac_patch_nums=(36 60 72)
tac_mixer_dims_p=(32 64)
tac_mixer_dims_c=(16 32)
tac_decoder_ks=(0 1 2)

# --- 4. 随机搜索设置 ---
TOTAL_RUNS=512

# --- [核心改进] 动态寻找空闲GPU的函数 ---
# 该函数会循环检查，直到找到一个空闲的GPU并返回其ID
find_free_gpu() {
    local free_gpu_id=""
    while [[ -z "$free_gpu_id" ]]; do
        # 遍历所有可用的GPU
        for gpu_id in "${GPUS[@]}"; do
            pid=${gpu_pids[$gpu_id]}
            # 检查条件：
            # 1. PID为空 (从未分配过任务或任务已结束并被清理)
            # 2. PID不为空，但该进程已不存在 (kill -0 失败)
            if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
                free_gpu_id=$gpu_id
                # 清理旧的PID记录，以防万一
                unset gpu_pids[$gpu_id]
                break # 找到了，跳出内层 for 循环
            fi
        done

        # 如果遍历完所有GPU都正忙，则等待任意一个后台任务结束
        if [[ -z "$free_gpu_id" ]]; then
            # echo "⏳ 所有GPU正忙，等待一个任务完成以释放资源..."
            wait -n
        fi
    done
    # 将找到的空闲GPU ID返回给调用者
    echo "$free_gpu_id"
}


echo "🚀 开始 TAC-Mixer 在 P12 上的随机搜索，总共将运行 ${TOTAL_RUNS} 组实验..."

# --- 5. 随机搜索主循环 ---
for (( i=1; i<=${TOTAL_RUNS}; i++ )); do

  # --- 随机采样超参数 (与原脚本相同) ---
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

  # --- [核心改进] GPU 动态分配 ---
  # 调用函数来获取一个当前空闲的GPU ID
  gpu_id=$(find_free_gpu)

  # --- 构建唯一的模型ID (与原脚本相同) ---
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

  # --- [核心改进] 记录新任务的 PID ---
  # $! 是 bash 的一个特殊变量，它会保存最后一个被放到后台的进程的PID
  new_pid=$!
  gpu_pids[$gpu_id]=$new_pid
  echo "   任务已启动，PID: ${new_pid}，已绑定至 GPU ${gpu_id}"

  sleep 1 # 短暂休眠，确保日志顺序和PID正确捕获

done

echo "✅ 所有 ${TOTAL_RUNS} 组任务已启动，等待最后批次的任务执行完毕..."
wait
echo "🎉🎉🎉 TAC-Mixer 在 P12 上的随机搜索任务已全部完成！🎉🎉🎉"
