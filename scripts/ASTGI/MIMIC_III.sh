#!/bin/bash

GPU_IDS="7"

model_name="ASTGI"
dataset_root_path="storage/datasets/MIMIC_III"
dataset_name="MIMIC_III"
seq_len=72
pred_len=3
enc_in=96
dec_in=96
c_out=96

dm=192
bs=8
lr=0.0005
dp=0.2

k_nn=96
n_prop=3
c_dim=64
t_dim=128
mlp_r=4.0
w_c=1.0

model_id="${model_name}_${dataset_name}_sl${seq_len}_pl${pred_len}_dm${dm}_L${n_prop}_k${k_nn}_dc${c_dim}_dt${t_dim}_wc${w_c}_dp${dp}_lr${lr}_bs${bs}_mlp${mlp_r}_run_optimal"

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