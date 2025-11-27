#!/bin/bash

# =========================================================================
#       SPECTRON (Patched Version) 在 P12 上的超参数搜索脚本 (高效GPU利用版)
# =========================================================================

# --- 1. GPU 配置 ---
# 在这里指定你想使用的GPU编号，用逗号隔开，例如 "0,1,2,3" 或 "4,5,6,7"
GPU_IDS="0"

# 将GPU_IDS字符串转换为bash数组
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

# [核心改进] 使用关联数组来追踪每个GPU上运行的进程ID (PID)
# 键: GPU ID, 值: 任务的 PID
declare -A gpu_pids
for gpu in "${GPUS[@]}"; do
    gpu_pids[$gpu]=""
done

echo "🚀 将在 ${NUM_GPUS} 个GPU (${GPUS[*]}) 上并行执行 SPECTRON 任务，采用动态调度策略。"

# --- 2. 固定参数 ---
model_name="SPECTRON" # 你的模型文件夹名称
dataset_root_path="storage/datasets/P12"
dataset_name="P12"
seq_len=36
enc_in=36
dec_in=36
c_out=36
pred_len=3

# --- 3. 新版超参数搜索空间 (为P12和分块模型优化) ---
d_models=(64 128)
n_heads_options=(4 8)
batch_sizes=(16 32)
lrs=(0.01)

# [核心] 分块相关参数。由于 seq_len=36, patch_len 必须小于36
patch_lens=(12 16 24)
strides=(8 12 16)

# [核心] 模型结构参数
num_kernels=(32 64)
num_intra_layers=(1 2)
dropouts=(0.1 0.2)

# --- 4. 随机搜索设置 ---
TOTAL_RUNS=128 # 总共运行的实验组数

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

echo "🚀 开始 SPECTRON-Patch 在 P12 上的随机搜索，总共将运行 ${TOTAL_RUNS} 组实验..."

# --- 5. 随机搜索主循环 ---
for (( i=1; i<=${TOTAL_RUNS}; i++ )); do

  # --- 随机采样超参数 (与原脚本相同) ---
  dm=${d_models[$((RANDOM % ${#d_models[@]}))]}
  bs=${batch_sizes[$((RANDOM % ${#batch_sizes[@]}))]}
  lr=${lrs[$((RANDOM % ${#lrs[@]}))]}

  pl=${patch_lens[$((RANDOM % ${#patch_lens[@]}))]}
  nk=${num_kernels[$((RANDOM % ${#num_kernels[@]}))]}
  nil=${num_intra_layers[$((RANDOM % ${#num_intra_layers[@]}))]}
  dp=${dropouts[$((RANDOM % ${#dropouts[@]}))]}

  # 确保 stride <= patch_len，这是一个有效的约束
  while true; do
    st=${strides[$((RANDOM % ${#strides[@]}))]}
    if [ $st -le $pl ]; then
      break
    fi
  done

  # 确保 n_heads 能被 d_model 整除
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
  model_id="${model_name}_${dataset_name}_sl${seq_len}_plen${pred_len}_dm${dm}_nh${nh}_pl${pl}_st${st}_nk${nk}_nil${nil}_dp${dp}_lr${lr}_bs${bs}_run${i}"

  echo "-----------------------------------------------------------------------"
  echo "📈 启动任务 [${i}/${TOTAL_RUNS}] -> 动态分配至空闲 GPU ${gpu_id}"
  echo "   Config: d_model=${dm}, n_heads=${nh}, patch_len=${pl}, stride=${st}, kernels=${nk}, intra_layers=${nil}"
  echo "   Training: bs=${bs}, lr=${lr}, dropout=${dp}"
  echo "   Model ID: ${model_id}"
  echo "-----------------------------------------------------------------------"

  # --- 在后台启动训练任务 ---
  # 使用 () & 将命令放入子shell后台执行
  (
    CUDA_VISIBLE_DEVICES=${gpu_id} python main.py \
      --is_training 1 \
      --model_id "$model_id" \
      --model_name "$model_name" \
      --d_model "$dm" \
      --n_heads "$nh" \
      --dropout "$dp" \
      --spectron_num_kernels "$nk" \
      --spectron_d_max 5.0 \
      --spectron_patch_len "$pl" \
      --spectron_patch_stride "$st" \
      --spectron_num_intra_layers "$nil" \
      --spectron_kernel_chunk_size 16 \
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
      --use_multi_gpu 0
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
echo "🎉🎉🎉 P12 随机搜索任务已全部完成！🎉🎉🎉"