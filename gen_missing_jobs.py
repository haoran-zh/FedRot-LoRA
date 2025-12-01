import os
import yaml
import ast
import argparse

# --- Configuration for Search Space ---
# Define the experiments you WANT to exist.
REQUIRED_SEEDS = [11, 12, 13]
REQUIRED_LOCAL_STEPS = [20]
# Note: Ensure these floats match how folders are named (e.g. 0.005 vs 5e-3)
# Mapping: (Command Line Argument value) -> (Folder Name suffix)
LR_MAPPING = {
    '5e-4': 'lr0.0005',
    '1e-3': 'lr0.001',
    '5e-3': 'lr0.005',
    '2e-2': 'lr0.02',
}
# 'lr0.0005' 'lr0.001' 'lr0.002' 'lr0.005' 'lr0.01' 'lr0.02' 'lr0.05' 'lr0.1'

# Map your YAML config files to the Algorithm Names detected by check_algorithm
YAML_TO_ALGO = {
    "federatedscope/glue/yamls/base_rolora.yaml": "RoLoRA",
    "federatedscope/glue/yamls/base_fedlora2.yaml": "FedLoRA2",
    "federatedscope/glue/yamls/base_worotation.yaml": "FedLoRA2_wo_Rotation"
}


# --- Reused Logic from your exp_results.py ---

def check_config(sub_exp_dir):
    config_path = os.path.join(sub_exp_dir, 'config.yaml')
    if not os.path.exists(config_path): return None
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def is_run_complete(subexp_path):
    """
    Checks if a run is 'complete' and valid.
    Returns True if log exists and has data, False otherwise.
    """
    filepath = os.path.join(subexp_path, 'eval_results.log')
    if not os.path.exists(filepath):
        return False

    # Quick check: does the file have lines?
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            if len(lines) < 5:  # Arbitrary threshold: empty logs are bad
                return False
            # Optional: Check if the last line is a high round number?
            # For now, existence of data is enough to skip re-running.
            return True
    except:
        return False


def check_algorithm(config):
    if not config: return 'Unknown'
    # Robust check using .get to avoid KeyErrors
    lora_cfg = config.get('lora', {})
    lora_rotate = lora_cfg.get('rotate', False)
    lora_rolora = lora_cfg.get('rolora', False)
    method = lora_cfg.get('method', '')

    if lora_rotate:
        return 'FedLoRA2'
    elif lora_rolora:
        return 'RoLoRA'
    elif (method == 'shareAB') and not lora_rotate:
        return 'FedLoRA2_wo_Rotation'
    else:
        return 'Unknown Algorithm'


def get_seed_from_config(config):
    if not config: return None
    return config.get('seed', None)


# --- Main Generator Logic ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, default='roberta-large@huggingface_llm')
    parser.add_argument('-d', '--dataset_name', type=str, default='mnli@glue')
    parser.add_argument('-o', '--output_dir', type=str, default='./exp/FedAvg_FacebookAI')
    parser.add_argument('--total_train', type=int, default=10000)
    args = parser.parse_args()

    missing_cmds = []

    print(f"Scanning {args.output_dir} for missing experiments...")

    # Iterate over the defined Grid
    for lr_cmd_val, lr_folder_suffix in LR_MAPPING.items():
        for lstep in REQUIRED_LOCAL_STEPS:
            lstep_str = f"lstep{lstep}"

            # Construct expected Experiment Directory
            exp_name = f'{args.model_name}_on_{args.dataset_name}_{lr_folder_suffix}_{lstep_str}'
            exp_dir = os.path.join(args.output_dir, exp_name)

            # Map found algorithms to the seeds that successfully finished
            # Structure: found_state['FedLoRA2'] = {11, 12}
            found_state = {algo: set() for algo in YAML_TO_ALGO.values()}

            if os.path.exists(exp_dir):
                sub_exps = os.listdir(exp_dir)
                for sub_exp in sub_exps:
                    sub_path = os.path.join(exp_dir, sub_exp)
                    if not os.path.isdir(sub_path): continue

                    config = check_config(sub_path)
                    algo_name = check_algorithm(config)
                    seed = get_seed_from_config(config)

                    # If this is a recognized algorithm and the run is valid
                    if algo_name in found_state and seed is not None:
                        if is_run_complete(sub_path):
                            found_state[algo_name].add(seed)

            # Compare Found vs Required
            rounds = args.total_train // lstep

            for yaml_file, algo_name in YAML_TO_ALGO.items():
                for seed in REQUIRED_SEEDS:
                    if seed not in found_state[algo_name]:
                        print(f"[MISSING] {algo_name} | LR: {lr_cmd_val} | Step: {lstep} | Seed: {seed}")

                        # Construct the command string matching base_run.sh style
                        cmd = (
                            f"python federatedscope/main.py "
                            f"--cfg {yaml_file} "
                            f"seed {seed} "
                            f"federate.total_round_num {rounds} "
                            f"data.type {args.dataset_name} "
                            f"train.local_update_steps {lstep} "
                            f"train.optimizer.lr {lr_cmd_val}"
                        )
                        missing_cmds.append(cmd)

    # --- Write the Bash Script ---
    output_script = "run_missing.sh"

    # We write a bash script that uses the specific GPU logic from your base_run.sh
    # but iterates over a predefined array of commands instead of loops.

    bash_template_header = """#!/bin/bash
# AUTO-GENERATED SCRIPT FOR MISSING EXPERIMENTS

cleanup() {
    echo ">>> CTRL+C detected. Killing background Python jobs..."
    kill $(jobs -p) 2>/dev/null
    wait
    echo ">>> Clean stop."
    exit 1
}
trap cleanup SIGINT SIGTERM

GPUS=(1)                # <--- CHECK YOUR GPU IDs HERE
JOBS_PER_GPU=1          
NUM_GPUS=${#GPUS[@]}
BATCH_SIZE=$((NUM_GPUS * JOBS_PER_GPU))

# Array of missing commands
declare -a COMMANDS=(
"""

    bash_template_footer = """
)

counter=0
total_jobs=${#COMMANDS[@]}

echo "Found $total_jobs missing experiments to run."

for cmd in "${COMMANDS[@]}"
do
    # 1. Find batch position
    batch_position=$((counter % BATCH_SIZE))

    # 2. Assign GPU
    gpu_idx=$((batch_position / JOBS_PER_GPU))
    CURRENT_GPU=${GPUS[$gpu_idx]}

    echo ">>> [Job $((counter+1))/$total_jobs] Running on GPU $CURRENT_GPU: $cmd"

    # Run command in background with specific device attached
    $cmd device $CURRENT_GPU > /dev/null 2>&1 &

    # Stagger start
    sleep 30

    ((counter++))

    # Batch Wait
    if (( counter % BATCH_SIZE == 0 )); then
        echo ">>> Batch full. Waiting..."
        wait
    fi
done

wait
echo "All missing experiments completed."
"""

    with open(output_script, "w") as f:
        f.write(bash_template_header)
        for cmd in missing_cmds:
            f.write(f'    "{cmd}"\n')
        f.write(bash_template_footer)

    print(f"\nSuccessfully generated '{output_script}' with {len(missing_cmds)} commands.")
    print("Please review the GPUS=(...) line in the generated file before running.")


if __name__ == "__main__":
    main()