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
MEM_THRESHOLD=10000 # Example: Wait if the GPU is using more than 10GB (10000 MiB)
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
GPUS=(0 1)                # Your GPU IDs
JOBS_PER_GPU=1             # How many to stack per GPU
NUM_GPUS=${#GPUS[@]}
# Total concurrent jobs = 3 GPUs * 2 jobs = 6
BATCH_SIZE=$((NUM_GPUS * JOBS_PER_GPU))

BASE_YAMLS=(
  "federatedscope/glue/yamls/base_rescale.yaml"
#    "federatedscope/glue/yamls/base_rolora.yaml"
#    "federatedscope/glue/yamls/base_fedrot_lora.yaml"
#    "federatedscope/glue/yamls/base_worotation.yaml"  # FedIT
#    "federatedscope/glue/yamls/base_FFAlora.yaml"
)
#    "federatedscope/glue/yamls/base_fedrot_lora.yaml"
#    "federatedscope/glue/yamls/base_worotation.yaml"
SEEDS=(11 12 13)
TOTAL_TRAIN=5000
LOCAL_STEPS=(20)
DATAS=('mnli@glue' 'sst2@glue' 'qnli')  # 'mnli@glue'
# 'lr0.005' 'lr0.02'
#LR_VALUES=(0.0005 0.001 0.005 0.02)
ROTATE_LAMBDA=(1.0)
ROTATE_REG=False
CLIENTNUM=3

# LR_VALUES=(5e-4 1e-3 2e-3 5e-3 1e-2 2e-2 5e-2 1e-1)
# rte 'lr0.0005' 'lr0.001' 'lr' 'lr0.02'
#     "cola": ("sentence", None),
  #    "mnli": ("premise", "hypothesis"),
  #    "mrpc": ("sentence1", "sentence2"),
  #    "qnli": ("question", "sentence"),
  #    "qqp": ("question1", "question2"),
  #    "rte": ("sentence1", "sentence2"),
  #    "sst2": ("sentence", None),
  #    "stsb": ("sentence1", "sentence2"),
  #    "wnli": ("sentence1", "sentence2"),
counter=0

for DATA in "${DATAS[@]}"
do
  for BASE_YAML in "${BASE_YAMLS[@]}"
  do
      echo "--- Base Config: $BASE_YAML ---"

      # --- LOGIC CHANGE HERE ---
      # Check if the current yaml string contains "fedlora2"
      if [[ "$BASE_YAML" == *"base_fedlora2"* ]]; then
          # If yes, we search through the list
          CURRENT_LAMBDAS=("${ROTATE_LAMBDA[@]}")
          LR_VALUES=(0.005 0.02) # skip small lrs for FedLoRA2
          echo ">>> FedLoRA2 detected: Sweeping lambdas: ${CURRENT_LAMBDAS[*]}, lr set to: ${LR_VALUES[*]}"
      else
          # If no, we run only once (using 0 or 1.0 as a placeholder)
          CURRENT_LAMBDAS=(0.0)
          LR_VALUES=(0.005 0.02) # 0.0005 0.001
          echo ">>> Not FedLoRA2: Running single lambda (0.0) all lr"
      fi

      for LS in "${LOCAL_STEPS[@]}"
      do
          ROUNDS=$((TOTAL_TRAIN / LS))
          for LR in "${LR_VALUES[@]}"
          do

              # Loop over the dynamic CURRENT_LAMBDAS list
              for LAMBDA in "${CURRENT_LAMBDAS[@]}"
              do
                for SEED in "${SEEDS[@]}"
                do
                    # --- MAP JOB TO GPU ---
                    batch_position=$((counter % BATCH_SIZE))
                    gpu_idx=$((batch_position / JOBS_PER_GPU))
                    CURRENT_GPU=${GPUS[$gpu_idx]}

                    echo "Running (Job #$counter): GPU=$CURRENT_GPU | YAML=$(basename $BASE_YAML) | LR=$LR | Lambda=$LAMBDA | Seed=$SEED"

                    # check_gpu_availability $CURRENT_GPU  # Ensure this function is defined in your environment

                    LOG_BUFFER="./log_buffer_${counter}.log"

                    # Execute in background
                    python federatedscope/main.py \
                        --cfg $BASE_YAML \
                        device $CURRENT_GPU \
                        seed $SEED \
                        lora.rotate_lambda $LAMBDA \
                        lora.rotate_reg $ROTATE_REG \
                        federate.total_round_num $ROUNDS \
                        federate.client_num $CLIENTNUM \
                        federate.sample_client_num $CLIENTNUM \
                        data.type $DATA \
                        train.local_update_steps $LS \
                        train.optimizer.lr $LR > "$LOG_BUFFER" 2>&1 &

                    # --- SAFETY STAGGER ---
                    sleep 60

                    ((counter++))

                    # --- BATCH WAIT ---
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
done

wait
echo "All experiments completed."

# empty log buffer
truncate -s 0 ./log_buffer.log