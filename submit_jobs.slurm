#!/bin/bash
#SBATCH -J FL_Experiment        # Job name
#SBATCH -o logs/job.%A_%a.out   # Output file (%A=ArrayID, %a=Index)
#SBATCH -e logs/job.%A_%a.err   # Error file
#SBATCH -p gh                   # Partition: Grace Hopper (Production)
#SBATCH -N 1                    # 1 Node per array task
#SBATCH -n 1                    # 1 Task per node
#SBATCH -t 20:00:00             # Time limit
#SBATCH --array=0-3             # Create 4 jobs

# --- Email Notification ---
#SBATCH --mail-type=all
#SBATCH --mail-user=haoran@utexas.edu

# --- Configuration ---
PROJECT_DIR="/work/11240/hz8556/vista/FedSA-LoRA"
CMD_FILE="experiments.txt"
MAX_PARALLEL=4                  # Max concurrent jobs per GPU

# --- Environment Setup ---
echo ">>> Setting up environment..."
cd "$PROJECT_DIR" || { echo "Failed to cd to $PROJECT_DIR"; exit 1; }

# FIX: Initialize conda properly before activating
# (Adjust the path '/work/...' to where your miniconda/anaconda is actually installed if this fails)
source $WORK/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source $HOME/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate fedsa-lora

# --- ENABLE MPS ---
echo ">>> Initializing NVIDIA MPS..."
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d
sleep 2
echo ">>> MPS Initialized."

# --- Split Workload Logic ---
TOTAL_LINES=$(wc -l < "$CMD_FILE")
CHUNK_SIZE=$(( (TOTAL_LINES + SLURM_ARRAY_TASK_COUNT - 1) / SLURM_ARRAY_TASK_COUNT ))
START_LINE=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE + 1 ))
END_LINE=$(( (SLURM_ARRAY_TASK_ID + 1) * CHUNK_SIZE ))

if [ "$END_LINE" -gt "$TOTAL_LINES" ]; then
    END_LINE=$TOTAL_LINES
fi

echo ">>> SLURM Array Task $SLURM_ARRAY_TASK_ID"
echo ">>> Processing lines $START_LINE to $END_LINE from $CMD_FILE"

# --- Execution Loop ---
counter=0
pids=()

# FIX: Process Substitution (< <(...)) keeps variables alive outside the loop
while read -r cmd; do

    echo "Starting: $cmd"

    # FIX: Use 'eval' for safer execution
    eval "$cmd" > "logs/exp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${counter}.log" 2>&1 &
    pid=$!
    pids+=($pid)
    ((counter++))

    # --- Batching ---
    if (( counter % MAX_PARALLEL == 0 )); then
        echo ">>> Reached $MAX_PARALLEL concurrent jobs. Waiting for batch to finish..."
        for p in "${pids[@]}"; do
            wait "$p"
        done
        echo ">>> Batch finished. Cleaning up..."
        pids=()
    fi

done < <(sed -n "${START_LINE},${END_LINE}p" "$CMD_FILE")

# --- Wait for Final Batch ---
# Because we fixed the loop, ${pids[@]} is now visible here!
if [ ${#pids[@]} -gt 0 ]; then
    echo ">>> Waiting for final ${#pids[@]} jobs..."
    for p in "${pids[@]}"; do
        wait "$p"
    done
fi

# --- CLEANUP MPS ---
echo ">>> Stopping MPS..."
echo quit | nvidia-cuda-mps-control
echo ">>> Task Complete."