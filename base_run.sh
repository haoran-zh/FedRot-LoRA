#!/bin/bash

# some rolora code may collapse because gpu vram overflows, need to check after finish

# 1. Safety Trap
cleanup() {
    echo ">>> CTRL+C detected. Killing background Python jobs..."
    kill $(jobs -p) 2>/dev/null
    wait
    echo ">>> Clean stop."
    exit 1
}
trap cleanup SIGINT SIGTERM



# check gpu availability
MEM_THRESHOLD=20000 # Example: Wait if the GPU is using more than 20GB (20000 MiB)
CHECK_INTERVAL=30   # How often (in seconds) to check the GPU status
# --- New Function to Check GPU Availability ---
# Argument 1: GPU ID (e.g., 1)
check_gpu_availability() {
    local TARGET_GPU=$1
    echo "Checking GPU $TARGET_GPU availability..."

    while true; do
        # Use nvidia-smi to query memory usage (in MiB) for the specified GPU
        # The 'pmon' format with 'csv,noheader' is reliable for parsing
        local USED_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader --id=$TARGET_GPU | awk '{print $1}')

        # Handle case where command might fail or return empty (shouldn't happen on healthy systems)
        if [ -z "$USED_MEM" ]; then
            echo "  ⚠️ Warning: Could not read VRAM usage for GPU $TARGET_GPU. Sleeping and retrying..."
            sleep $CHECK_INTERVAL
            continue
        fi

        # Check if used memory is less than the threshold
        if (( USED_MEM < MEM_THRESHOLD )); then
            echo "  ✅ GPU $TARGET_GPU is available ($USED_MEM MiB < $MEM_THRESHOLD MiB). Proceeding."
            break # Exit the loop and continue script execution
        else
            echo "  ⏳ GPU $TARGET_GPU is busy ($USED_MEM MiB). Waiting $CHECK_INTERVAL seconds..."
            sleep $CHECK_INTERVAL
        fi
    done
}


# --- Configuration ---
GPUS=(2)                # Your GPU IDs
JOBS_PER_GPU=1             # How many to stack per GPU
NUM_GPUS=${#GPUS[@]}
# Total concurrent jobs = 3 GPUs * 2 jobs = 6
BATCH_SIZE=$((NUM_GPUS * JOBS_PER_GPU))

BASE_YAMLS=(
#    "federatedscope/glue/yamls/base_rolora.yaml"
    "federatedscope/glue/yamls/base_fedlora2.yaml"
#    "federatedscope/glue/yamls/base_worotation.yaml"
#    "federatedscope/glue/yamls/base_FFAlora.yaml"
)
#    "federatedscope/glue/yamls/base_fedlora2.yaml"
#    "federatedscope/glue/yamls/base_worotation.yaml"
SEEDS=(11 12 13)
TOTAL_TRAIN=10000
LOCAL_STEPS=(20)
DATA='rte@glue'
# 'lr0.005' 'lr0.02'
LR_VALUES=(0.001 0.005 0.02)
ROTATE_LAMBDA=(1.0 0.5)
ROTATE_REG=False
# LR_VALUES=(5e-4 1e-3 2e-3 5e-3 1e-2 2e-2 5e-2 1e-1)
# rte 'lr0.0005' 'lr0.001' 'lr' 'lr0.02'

counter=0

for BASE_YAML in "${BASE_YAMLS[@]}"
do
    echo "--- Base Config: $BASE_YAML ---"
    for LS in "${LOCAL_STEPS[@]}"
    do
        ROUNDS=$((TOTAL_TRAIN / LS))
        for LR in "${LR_VALUES[@]}"
        do
            for LAMBDA in "${ROTATE_LAMBDA[@]}"
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

                  check_gpu_availability $CURRENT_GPU

                  LOG_BUFFER="./log_buffer.log"

                  # Execute in background
                  python federatedscope/main.py \
                      --cfg $BASE_YAML \
                      device $CURRENT_GPU \
                      seed $SEED \
                      lora.rotate_lambda $LAMBDA\
                      lora.rotate_reg $ROTATE_REG \
                      lora.warm_up True \
                      lora.warm_up_rounds 40 \
                      lora.warm_up_lr 2e-2 \
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
done

wait
echo "All experiments completed."

# empty log buffer
truncate -s 0 ./log_buffer.log