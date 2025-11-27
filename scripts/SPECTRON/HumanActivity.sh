#!/bin/bash

# --- GPU并行设置 ---
NUM_GPUS=8
job_count=0

# --- 固定参数 ---
model_name="$(basename "$(dirname "$(readlink -f "$0")")")"
dataset_root_path=storage/datasets/HumanActivity
dataset_name=$(basename "$0" .sh)
seq_len=3000
enc_in=12
dec_in=12
c_out=12
pred_len=1000

d_models=(32 64)             # 固定表现最佳的 d_model
batch_sizes=(4 8)       # 核心搜索变量: Batch Size
lrs=(0.001)           # 核心搜索变量: Learning Rate
num_kernels=(32 64)          # 模型核心参数
num_intra_layers=(3 4)     # 模型核心参数
dropouts=(0.1 0.2 0.3)           # 正则化参数
n_heads_list=(2 4)               # 新增：枚举 head 个数

total_combinations=$(( ${#d_models[@]} * ${#batch_sizes[@]} * ${#lrs[@]} * ${#num_kernels[@]} * ${#num_intra_layers[@]} * ${#dropouts[@]} * ${#n_heads_list[@]} ))
echo "🚀 开始综合精调搜索，总共将运行 ${total_combinations} 组实验..."

# --- 训练循环 ---
for dm in "${d_models[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    for lr in "${lrs[@]}"; do
      for nk in "${num_kernels[@]}"; do
        for nil in "${num_intra_layers[@]}"; do
          for dp in "${dropouts[@]}"; do
            for nh in "${n_heads_list[@]}"; do

              gpu_id=$((job_count % NUM_GPUS))

              # 构建包含 batch_size 和 n_heads 的唯一 model_id
              model_id="${model_name}_${dataset_name}_sl${seq_len}_pl${pred_len}_dm${dm}_nh${nh}_nk${nk}_nil${nil}_dp${dp}_lr${lr}_bs${bs}"

              echo "----------------------------------------------------"
              echo "📈 启动任务 [${job_count}] -> GPU ${gpu_id}"
              echo "   Config: bs=${bs}, lr=${lr}, d_model=${dm}, n_heads=${nh}, kernels=${nk}, intra_layers=${nil}, dropout=${dp}"
              echo "   Model ID: ${model_id}"
              echo "----------------------------------------------------"

              # 在后台启动训练任务，并传入 batch_size 和 n_heads 参数
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
          done
        done
      done
    done
  done
done

echo "✅ 所有任务已启动，等待最后批次的任务执行完毕..."
wait
echo "🎉🎉🎉 全部超参数搜索任务已完成！🎉🎉🎉"