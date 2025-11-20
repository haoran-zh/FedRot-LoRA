#!/bin/bash


# 1. Safety Trap
cleanup() {
    echo ">>> CTRL+C detected. Killing background Python jobs..."
    kill $(jobs -p) 2>/dev/null
    wait
    echo ">>> Clean stop."
    exit 1
}
trap cleanup SIGINT SIGTERM



# --- Configuration ---
GPUS=(2)                # Your GPU IDs
JOBS_PER_GPU=2              # How many to stack per GPU
NUM_GPUS=${#GPUS[@]}
# Total concurrent jobs = 3 GPUs * 2 jobs = 6
BATCH_SIZE=$((NUM_GPUS * JOBS_PER_GPU))

BASE_YAMLS=(
    "federatedscope/glue/yamls/base_fedlora2.yaml"
    "federatedscope/glue/yamls/base_rolora.yaml"
    "federatedscope/glue/yamls/base_worotation.yaml"
)
SEEDS=(12)
TOTAL_TRAIN=10000
LOCAL_STEPS=(10 20)
DATA='rte@glue'
LR_VALUES=(5e-4 1e-3 2e-3 5e-3 1e-2 2e-2 5e-2 1e-1)

counter=0

for BASE_YAML in "${BASE_YAMLS[@]}"
do
    echo "--- Base Config: $BASE_YAML ---"
    for LS in "${LOCAL_STEPS[@]}"
    do
        ROUNDS=$((TOTAL_TRAIN / LS))
        for LR in "${LR_VALUES[@]}"
        do
            for SEED in "${SEEDS[@]}"
            do
                # --- LOGIC TO MAP JOB TO GPU ---
                # 1. Find where we are in the current batch (0 to 5)
                batch_position=$((counter % BATCH_SIZE))

                # 2. Integer division to assign GPU.
                # Positions 0,1 -> GPU index 0. Positions 2,3 -> GPU index 1, etc.
                gpu_idx=$((batch_position / JOBS_PER_GPU))
                CURRENT_GPU=${GPUS[$gpu_idx]}

                echo "Running (Job #$counter): GPU=$CURRENT_GPU | LR=$LR | LS=$LS | Seed=$SEED | Rounds=$ROUNDS"

                LOG_BUFFER="./log_buffer.log"

                # Execute in background
                python federatedscope/main.py \
                    --cfg $BASE_YAML \
                    device $CURRENT_GPU \
                    seed $SEED \
                    federate.total_round_num $ROUNDS \
                    data.type $DATA \
                    train.local_update_steps $LS \
                    train.optimizer.lr $LR > "$LOG_BUFFER" 2>&1 &

                # --- SAFETY STAGGER ---
                # Sleep 60 seconds before starting the next one to prevent
                # simultaneous initialization memory spikes.
                sleep 60

                ((counter++))

                # --- BATCH WAIT ---
                # If we have filled all slots (counter is a multiple of 6), wait.
                if (( counter % BATCH_SIZE == 0 )); then
                    echo ">>> Batch full ($BATCH_SIZE jobs running). Waiting for completion..."
                    wait
                    echo ">>> Batch finished. Starting next batch."
                fi
            done
        done
    done
done

wait
echo "All experiments completed."

# empty log buffer
truncate -s 0 ./log_buffer.log