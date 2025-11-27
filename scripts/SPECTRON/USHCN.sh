#!/bin/bash

NUM_GPUS=8
job_count=0

model_name="$(basename "$(dirname "$(readlink -f "$0")")")"
dataset_root_path=storage/datasets/USHCN
dataset_name=$(basename "$0" .sh)

# USHCN 数据集特定参数
seq_len=150
enc_in=5
dec_in=5
c_out=5
pred_len=50

# --- 超参数搜索空间 (Coarse Search, 共 96 组) ---
d_models=(32 64 96)               # 模型容量: USHCN序列短，从小容量开始
batch_sizes=(8 16)             # 批次大小: 数据量不大，测试标准批次
lrs=(0.001)        # 学习率: 关键变量，探索不同数量级
num_kernels=(16 32)             # SPECTRON核心: 捕捉季节性的基函数数量
num_intra_layers=(3 4)          # 模型深度: 浅层防止短序列过拟合
dropouts=(0.1)             # 正则化: 防止过拟合的关键

total_combinations=$(( ${#d_models[@]} * ${#batch_sizes[@]} * ${#lrs[@]} * ${#num_kernels[@]} * ${#num_intra_layers[@]} * ${#dropouts[@]} ))
echo "🚀 开始 SPECTRON 在 USHCN 上的粗搜索，总共将运行 ${total_combinations} 组实验..."

# --- 训练循环 ---
for dm in "${d_models[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    for lr in "${lrs[@]}"; do
      for nk in "${num_kernels[@]}"; do
        for nil in "${num_intra_layers[@]}"; do
          for dp in "${dropouts[@]}"; do

            gpu_id=$((job_count % NUM_GPUS))

            # 构建包含所有超参数的唯一 model_id
            model_id="${model_name}_${dataset_name}_sl${seq_len}_pl${pred_len}_dm${dm}_nh4_nk${nk}_nil${nil}_dp${dp}_lr${lr}_bs${bs}"

            echo "----------------------------------------------------"
            echo "📈 启动任务 [${job_count}] -> GPU ${gpu_id}"
            echo "   Config: bs=${bs}, lr=${lr}, d_model=${dm}, kernels=${nk}, intra_layers=${nil}, dropout=${dp}"
            echo "   Model ID: ${model_id}"
            echo "----------------------------------------------------"

            # 在后台启动训练任务
            CUDA_VISIBLE_DEVICES=${gpu_id} python main.py \
              --is_training 1 \
              --model_id "$model_id" \
              --model_name "$model_name" \
              --d_model "$dm" \
              --n_heads 4 \
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
            # 短暂休眠，避免瞬间启动过多进程导致系统不稳定
            sleep 2

            # 控制并行任务数量，等于GPU数量时等待一个任务完成后再启动新的
            if [[ $(jobs -r -p | wc -l) -ge $NUM_GPUS ]]; then
                wait -n
            fi

          done
        done
      done
    done
  done
done

echo "✅ 所有任务已启动，等待最后批次的任务执行完毕..."
wait
echo "🎉🎉🎉 USHCN 粗搜索任务已全部完成！请检查日志，寻找最优参数区域。🎉🎉🎉"