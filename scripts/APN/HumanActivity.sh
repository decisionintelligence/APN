#!/bin/bash

echo "Starting APN hyperparameter search script..."

GPU_IDS=(4)

# 模型和数据集的固定参数
model_name="APN"
dataset_root_path="storage/datasets/HumanActivity"
dataset_name="HumanActivity"
features="M"
seq_len=3000
pred_len=300
enc_in=12
dec_in=12
c_out=12
train_epochs=200
patience=10

dms=(56)
lrs=(0.01)
bss=(16)
dps=(0)
ps=(300)
te_dims=(8)

tasks=()
for dm in "${dms[@]}"; do
for lr in "${lrs[@]}"; do
for bs in "${bss[@]}"; do
for dp in "${dps[@]}"; do
for p in "${ps[@]}"; do
for te_dim in "${te_dims[@]}"; do

    model_id="${model_name}_${dataset_name}_sl${seq_len}_pl${pred_len}_dm${dm}_lr${lr}_bs${bs}_dp${dp}_p${p}_tedim${te_dim}"

    tasks+=( "python main.py \
        --is_training 1 \
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
        --loss \"MSE\" \
        --train_epochs \"$train_epochs\" \
        --patience \"$patience\" \
        --val_interval 1 \
        --itr 5 \
        --batch_size \"$bs\" \
        --learning_rate \"$lr\" \
        --d_model \"$dm\" \
        --dropout \"$dp\" \
        --apn_npatch \"$p\" \
        --apn_te_dim \"$te_dim\"" )

done; done; done; done; done; done

NUM_GPUS=${#GPU_IDS[@]}
total_tasks=${#tasks[@]}
task_idx=0

pids=()
for (( i=0; i<NUM_GPUS; i++ )); do
    pids+=([i]=0)
done

echo "Total GPUs available: ${NUM_GPUS}"
echo "Total tasks to run: ${total_tasks}"
echo "Starting task dispatcher..."

while [ $task_idx -lt $total_tasks ]; do
    free_gpu_idx=-1
    for i in "${!GPU_IDS[@]}"; do
        if [[ ${pids[$i]} -eq 0 ]] || ! ps -p ${pids[$i]} > /dev/null; then
            free_gpu_idx=$i
            break
        fi
    done

    if [ $free_gpu_idx -ne -1 ]; then
        gpu_id=${GPU_IDS[$free_gpu_idx]}
        command=${tasks[$task_idx]}

        echo "------------------------------------------------------------------------"
        echo "Assigning Task #${task_idx} to GPU #${gpu_id}..."
        echo "COMMAND: CUDA_VISIBLE_DEVICES=${gpu_id} ${command}"
        echo "------------------------------------------------------------------------"

        model_id_val=$(echo "$command" | grep -oP '(?<=--model_id ")[^"]*')
        (
            export CUDA_VISIBLE_DEVICES=${gpu_id}
            eval ${command} 2>&1 | tee "./logs/${model_id_val}.log"
        ) &

        pids[$free_gpu_idx]=$!
        task_idx=$((task_idx + 1))
    else
        wait -n
    fi
done

echo "########################################################################"
echo "All tasks have been launched. Waiting for the final jobs to complete..."
echo "########################################################################"
wait

echo "All experiments finished."