import math

# --- Configuration ---
algos = {
    "FedLoRA2": {
        "yaml": "federatedscope/glue/yamls/base_fedlora2.yaml",
        # Added 0.0005 and 0.001 here so the logic applies to them
        "lrs": [0.0005, 0.001, 0.005, 0.02],
        "lambdas": [1.0, 0.7, 0.5, 0.3, 0.1],
        # We removed 'warmup' list here because we will determine it by logic
    },
    "RoLoRA": {
        "yaml": "federatedscope/glue/yamls/base_rolora.yaml",
        "lrs": [0.0005, 0.001, 0.005, 0.02],
        "lambdas": [0.0],
        'warmup': [False]  # Kept as fixed list for others
    },
    "FedIT": {
        "yaml": "federatedscope/glue/yamls/base_worotation.yaml",
        "lrs": [0.005, 0.02],
        "lambdas": [0.0],
        'warmup': [False]
    },
    "FFAlora": {
        "yaml": "federatedscope/glue/yamls/base_FFAlora.yaml",
        "lrs": [0.0005, 0.001, 0.005, 0.02],
        "lambdas": [0.0],
        'warmup': [False]
    }
}

seeds = [11, 12, 13]
datasets = ['qqp@glue']
total_train_steps = 5000
local_steps = [20]
rotate_reg = "False"
device_id = 0

commands = []

for data in datasets:
    for algo_name, settings in algos.items():
        base_yaml = settings["yaml"]

        for ls in local_steps:
            rounds = total_train_steps // ls

            for lr in settings["lrs"]:
                for lam in settings["lambdas"]:

                    # --- LOGIC CHANGE START ---
                    # Determine warmup based on algorithm and LR
                    if algo_name == "FedLoRA2":
                        if lr in [0.0005, 0.001]:
                            warmup_val = True
                        else:
                            warmup_val = False
                    else:
                        # For other algorithms, just take the first value from their list
                        # (Since they are all [False], we take False)
                        warmup_val = settings.get('warmup', [False])[0]
                    # --- LOGIC CHANGE END ---

                    for seed in seeds:
                        cmd = (
                            f"python federatedscope/main.py "
                            f"--cfg {base_yaml} "
                            f"device {device_id} "
                            f"seed {seed} "
                            f"lora.rotate_lambda {lam} "
                            f"lora.warm_up {warmup_val} "
                            f"lora.rotate_reg {rotate_reg} "
                            f"federate.total_round_num {rounds} "
                            f"data.type {data} "
                            f"train.local_update_steps {ls} "
                            f"train.optimizer.lr {lr}"
                        )
                        commands.append(cmd)

# Write to file
output_file = "experiments.txt"
with open(output_file, "w") as f:
    for cmd in commands:
        f.write(cmd + "\n")

print(f"Successfully generated {len(commands)} commands in {output_file}.")
print(f"Estimated workload: {len(commands)} jobs.")
print(f"If running 5 per GPU: {math.ceil(len(commands) / 5)} batches needed.")