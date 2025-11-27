#!/bin/bash

GPU_IDS="6"

model_name="ASTGI"
dataset_root_path="storage/datasets/USHCN"
dataset_name="USHCN"
seq_len=150
pred_len=3
enc_in=5
dec_in=5
c_out=5

dm=64
bs=32
lr=0.001
dp=0.25

k_nn=80
n_prop=3
c_dim=24
t_dim=24
mlp_r=2.0
w_c=1.0

model_id="${model_name}_${dataset_name}_sl${seq_len}_pl${pred_len}_dm${dm}_L${n_prop}_k${k_nn}_dc${c_dim}_dt${t_dim}_wc${w_c}_dp${dp}_lr${lr}_bs${bs}_mlp_${mlp_r}_run_optimal"

(
  CUDA_VISIBLE_DEVICES=${GPU_IDS} python main.py \
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
    --train_epochs 300 \
    --patience 5 \
    --val_interval 1 \
    --itr 5 \
    --batch_size "$bs" \
    --learning_rate "$lr" \
    --d_model "$dm" \
    --dropout "$dp" \
    --stppgn_k_neighbors "$k_nn" \
    --stppgn_prop_layers "$n_prop" \
    --stppgn_channel_dim "$c_dim" \
    --stppgn_time_dim "$t_dim" \
    --stppgn_mlp_ratio "$mlp_r" \
    --stppgn_channel_dist_weight "$w_c"
) &

wait