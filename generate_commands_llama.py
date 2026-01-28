import math, os
from gen_missing_jobs import check_config, check_algorithm, match_exp

# --- Configuration ---
algos = {
    "FedLoRA2": {
        "yaml": "federatedscope/llm/yamls/base_fedlora2.yaml",
        # Added 0.0005 and 0.001 here so the logic applies to them
        "lrs": [0.005, 0.02],
        "lambdas": [0.3, 0.1],
        "warmup_round": [100],
        # We removed 'warmup' list here because we will determine it by logic
    },
    "RoLoRA": {
        "yaml": "federatedscope/llm/yamls/base_rolora.yaml",
        "lrs": [0.0005, 0.001],
        "lambdas": [0.0],
        'warmup': [False]  # Kept as fixed list for others
    },
    "FedIT": {
        "yaml": "federatedscope/llm/yamls/base_worotation.yaml",
        "lrs": [0.005, 0.02],
        "lambdas": [0.0],
        'warmup': [False]
    },
    "FFA-LoRA": {
        "yaml": "federatedscope/llm/yamls/base_FFAlora.yaml",
        "lrs": [0.005, 0.02],
        "lambdas": [0.0],
        'warmup': [False]
    }
}

seeds = [11, 12, 13]
datasets = ['code_search_net@llm']
total_train_steps = 3000
local_steps = [30]
rotate_reg = "False"
device_id = 0
warmup_lr = 0.005 # Default warmup_lr
ClientNum = 3

data_root = "$SCRATCH/data/"

# check existing runs
folder_head = "./exp/FedAvg_meta-llama/Meta-Llama-3-8B@huggingface_llm_on_"
existing_exp = {}

for data in datasets:
    folder_data = f"{folder_head}{data}"
    for algo_name, settings in algos.items():
        for ls in local_steps:
            for lr in settings["lrs"]:
                exp_dir = f"{folder_data}_lr{lr}_lstep{ls}"
                try:
                    sub_exps = os.listdir(exp_dir)
                except:
                    print('path not found, no existing experiments')
                    break
                for sub_exp in sub_exps:
                    sub_path = os.path.join(exp_dir, sub_exp)
                    if not os.path.isdir(sub_path): continue
                    config = check_config(sub_path)
                    algorithm_name = check_algorithm(config)
                    if algorithm_name not in existing_exp:
                        existing_exp[algorithm_name] \
                            = [{
                            'data': config['data']['type'],
                            'seed': config['seed'],
                            'lr': config['train']['optimizer']['lr'],
                            'client_num': config['federate']['client_num'],
                            'warm_up': config['lora']['warm_up'],
                            'warmup_round': config['lora']['warm_up_rounds'],
                            'local_update_steps': config['train']['local_update_steps'],
                            'lambda': config['lora']['rotate_lambda'],
                            }]
                    else:
                        existing_exp[algorithm_name].append(
                            {
                            'data': config['data']['type'],
                            'seed': config['seed'],
                            'lr': config['train']['optimizer']['lr'],
                            'client_num': config['federate']['client_num'],
                            'warm_up': config['lora']['warm_up'],
                            'warmup_round': config['lora']['warm_up_rounds'],
                            'local_update_steps': config['train']['local_update_steps'],
                            'lambda': config['lora']['rotate_lambda'],
                            }
                        )


commands = []

per_exp_config = {}
per_exp_config['client_num']=ClientNum
for data in datasets:
    per_exp_config['data']=data
    for algo_name, settings in algos.items():
        per_exp_config['algorithm_name']=algo_name
        base_yaml = settings["yaml"]

        for ls in local_steps:
            per_exp_config['local_update_steps']=ls
            rounds = total_train_steps // ls

            for lr in settings["lrs"]:
                per_exp_config['lr']=lr
                for lam in settings["lambdas"]:
                    per_exp_config['lambda']=lam
                    for warmup_round in settings.get('warmup_round', [100]):
                        per_exp_config['warmup_round']=warmup_round
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
                        per_exp_config['warm_up']=warmup_val
                        # --- LOGIC CHANGE END ---

                        for seed in seeds:
                            per_exp_config['seed'] = seed
                            # check if this exp has done
                            match = match_exp(per_exp_config, existing_exp)
                            if match is True:
                                print(f"Skipping existing: {per_exp_config}")
                                continue

                            cmd = (
                                f"python federatedscope/main.py "
                                f"--cfg {base_yaml} "
                                f"device {device_id} "
                                f"seed {seed} "
                                f"lora.rotate_lambda {lam} "
                                f"lora.warm_up {warmup_val} "
                                f"lora.warm_up_lr {warmup_lr} "
                                f"lora.warm_up_rounds {warmup_round} "
                                f"lora.rotate_reg {rotate_reg} "
                                f"federate.total_round_num {rounds} "
                                f"federate.client_num {ClientNum} "
                                f"federate.sample_client_num {ClientNum} "
                                f"data.type {data} "
                                f"data.root {data_root} "
                                f"train.local_update_steps {ls} "
                                f"train.optimizer.lr {lr}"
                            )
                            commands.append(cmd)

# Write to file
output_file = "experiments_llamacode.txt"
with open(output_file, "w") as f:
    for cmd in commands:
        f.write(cmd + "\n")

print(f"Successfully generated {len(commands)} commands in {output_file}.")
print(f"Estimated workload: {len(commands)} jobs.")
print(f"If running 5 per GPU: {math.ceil(len(commands) / 5)} batches needed.")