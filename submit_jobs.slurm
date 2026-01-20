#!/bin/bash
#SBATCH -J llamaco2        # Job name
#SBATCH -o $SCRATCH/logs/job.%A_%a.out   # Output file (%A=ArrayID, %a=Index)
#SBATCH -e $SCRATCH/logs/job.%A_%a.err   # Error file
#SBATCH -p gh                   # Partition: Grace Hopper (Production)
#SBATCH -N 1                    # 1 Node per array task
#SBATCH -n 1                    # 1 Task per node
#SBATCH -t 0:30:00             # Time limit
#SBATCH --array=0-7             # Create 8 jobs

# --- Email Notification ---
#SBATCH --mail-type=all
#SBATCH --mail-user=haoran@utexas.edu

# --- Configuration ---
PROJECT_DIR="/work/11240/hz8556/vista/FedSA-LoRA"
CMD_FILE="experiments_llamacode.txt"
MAX_PARALLEL=1                  # Max concurrent jobs per GPU

# --- Environment Setup ---
echo ">>> Setting up environment..."
cd "$PROJECT_DIR" || { echo "Failed to cd to $PROJECT_DIR"; exit 1; }

# Initialize conda
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

while read -r cmd; do

    echo "Starting: $cmd"

    # 1. Execute the command in the background
    eval "$cmd" > "logs/exp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${counter}.log" 2>&1 &
    pid=$!
    pids+=($pid)
    ((counter++))

    # 2. SAFETY STAGGER (New Addition)
    # Wait 15 seconds to let the GPU memory allocation settle before launching the next one.
    echo ">>> Waiting 10s for initialization..."
    sleep 60

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