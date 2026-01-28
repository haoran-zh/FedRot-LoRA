#!/bin/bash

# --- 1. Safety Trap ---
cleanup() {
    echo ">>> CTRL+C detected. Killing background Python jobs..."
    kill $(jobs -p) 2>/dev/null
    wait
    echo ">>> Clean stop."
    exit 1
}
trap cleanup SIGINT SIGTERM

# --- 2. GPU Monitoring Configuration ---
MEM_THRESHOLD=15000
CHECK_INTERVAL=30
GPUS=(1 2)                # Your GPU IDs
JOBS_PER_GPU=2             # How many to stack per GPU
NUM_GPUS=${#GPUS[@]}
BATCH_SIZE=$((NUM_GPUS * JOBS_PER_GPU))

# --- 3. GPU Availability Function ---
check_gpu_availability() {
    local TARGET_GPU=$1
    echo "Checking GPU $TARGET_GPU availability..."
    while true; do
        local USED_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader --id=$TARGET_GPU | awk '{print $1}')
        if [ -z "$USED_MEM" ]; then
            echo "  ⚠️ Warning: Could not read VRAM. Retrying..."
            sleep $CHECK_INTERVAL
            continue
        fi
        if (( USED_MEM < MEM_THRESHOLD )); then
            echo "  ✅ GPU $TARGET_GPU available ($USED_MEM MiB). Proceeding."
            break
        else
            echo "  ⏳ GPU $TARGET_GPU busy ($USED_MEM MiB). Waiting..."
            sleep $CHECK_INTERVAL
        fi
    done
}

# --- 4. Command Execution Logic ---
COMMANDS_FILE="experiments_ablation.txt"
counter=0

# Ensure the file exists
if [[ ! -f "$COMMANDS_FILE" ]]; then
    echo "Error: $COMMANDS_FILE not found!"
    exit 1
fi

# Read the file line by line
while IFS= read -r cmd || [[ -n "$cmd" ]]; do
    # Skip empty lines or lines starting with #
    [[ -z "$cmd" || "$cmd" =~ ^# ]] && continue

    # --- MAP JOB TO GPU ---
    batch_position=$((counter % BATCH_SIZE))
    gpu_idx=$((batch_position / JOBS_PER_GPU))
    CURRENT_GPU=${GPUS[$gpu_idx]}

    echo "-------------------------------------------------------"
    echo "Running Job #$counter on GPU $CURRENT_GPU"

    # Optional: Uncomment if you want to block until VRAM is free
    # check_gpu_availability $CURRENT_GPU

    # Replace 'device 0' in the text file with our current assigned GPU
    # This assumes the command in txt has 'device 0'
    FINAL_CMD=$(echo "$cmd" | sed "s/device 0/device $CURRENT_GPU/g")

    LOG_BUFFER="./log_buffer_${counter}.log"

    # Execute the command from the file in background
    eval "$FINAL_CMD" > "$LOG_BUFFER" 2>&1 &

    # --- SAFETY STAGGER ---
    # Prevents simultaneous heavy initialization on the same/different GPUs
    sleep 30

    ((counter++))

    # --- BATCH WAIT ---
    if (( counter % BATCH_SIZE == 0 )); then
        echo ">>> Batch full ($BATCH_SIZE jobs running). Waiting for completion..."
        wait
        echo ">>> Batch finished. Starting next batch."
    fi

done < "$COMMANDS_FILE"

# Wait for any remaining background jobs
wait
echo "All experiments from $COMMANDS_FILE completed."