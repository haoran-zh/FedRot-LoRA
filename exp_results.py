import os
import numpy as np
import yaml
import ast
import argparse  # 1. Import argparse


def check_config(sub_exp_dir):
    config_path = os.path.join(sub_exp_dir, 'config.yaml')
    with open(config_path, 'r') as f:
        # Use safe_load to safely parse the YAML file
        config_dict = yaml.safe_load(f)
    return config_dict


def read_log_file(subexp_path):
    """
    Reads a log file where each line is a Python dictionary containing nested metrics.

    It extracts 'Round', 'val_avg_loss', and 'val_accuracy' and converts them
    into a NumPy array suitable for analysis. Lines with a non-integer round
    (like 'Final') are skipped.

    Args:
        subexp_path (str): The path to the subdirectory containing results.

    Returns:
        tuple: (np.ndarray, np.ndarray) for (Accuracy, Loss) for all rounds,
               or (None, None) on error/no data.
    """
    filepath = os.path.join(subexp_path, 'eval_results.log')
    if not os.path.exists(filepath):
        print(f"Error: Log file not found at {filepath}")
        return None, None

    loss_list = []
    acc_list = []

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    # Safely evaluate the string as a Python literal (dictionary)
                    record = ast.literal_eval(line)

                    # Ensure 'Round' is a standard integer (skipping 'Final' round summary)
                    round_num = record.get('Round')
                    if not isinstance(round_num, int):
                        continue

                    results = record.get('Results_raw', {})

                    # Extract the required metrics from the nested dictionary
                    loss = results.get('val_avg_loss')
                    acc = results.get('val_accuracy')

                    if loss is not None and acc is not None:
                        # Append the required metrics
                        acc_list.append(acc)
                        loss_list.append(loss)

                except (ValueError, SyntaxError) as e:
                    # Skip lines that are not valid dictionary strings (e.g., raw text errors)
                    # print(f"Skipping line due to parsing error: '{line[:50]}'...")
                    continue

    except Exception as e:
        print(f"An error occurred during log file processing: {e}")
        return None, None

    # Convert the list of metrics into NumPy arrays
    return np.array(acc_list, dtype=np.float32), np.array(loss_list, dtype=np.float32)


def check_algorithm(config):
    lora_rotate = config['lora']['rotate']
    lora_rolora = config['lora']['rolora']
    method = config['lora']['method']  # shareAB, shareA, shareB, swap
    freeze_A = config['federate']['freeze_A']
    if freeze_A:
        algorithm_name = 'FFA-LoRA'
    elif lora_rotate:
        algorithm_name = 'FedLoRA2'
    elif lora_rolora:
        algorithm_name = 'RoLoRA'
    elif (method == 'shareAB') and not lora_rotate:
        algorithm_name = 'FedLoRA2_wo_Rotation'
    else:
        # Instead of exiting, return a value for robustness
        algorithm_name = 'Unknown Algorithm'
        # print('Unknown Algorithm!')
        # exit()
    return algorithm_name


def main():
    # 2. Setup argument parser
    parser = argparse.ArgumentParser(
        description='Evaluate and compare results from different Federated Learning algorithms.'
    )

    # 3. Add arguments for the hardcoded variables
    parser.add_argument(
        '-m', '--model_name',
        type=str,
        default='roberta-large@huggingface_llm',
        help="Model name used for the experiment."
    )
    parser.add_argument(
        '-d', '--dataset_name',
        type=str,
        default='mnli@glue',
        help="Dataset name used for the experiment."
    )
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        default='./exp/FedAvg_FacebookAI',
        help="Base directory where experiment results are stored."
    )
    parser.add_argument(
        '-l', '--lr',
        type=str,
        default='lr0.005',
        help="Learning rate string used in the experiment directory name."
    )
    parser.add_argument(
        '-s', '--lstep',
        type=str,
        default='lstep10',
        help="Local steps string used in the experiment directory name."
    )

    # Parse arguments
    args = parser.parse_args()

    # 4. Use parsed arguments instead of hardcoded values
    model_name = args.model_name
    dataset_name = args.dataset_name
    output_dir = args.output_dir
    lr = args.lr
    lstep = args.lstep

    exp_name = f'{model_name}_on_{dataset_name}_{lr}_{lstep}'
    exp_dir = os.path.join(output_dir, exp_name)

    if not os.path.exists(exp_dir):
        print(f"Error: Experiment directory not found at {exp_dir}")
        return

    sub_exps = os.listdir(exp_dir)

    eval_dict = {}
    for sub_exp in sub_exps:
        sub_exp_path = os.path.join(exp_dir, sub_exp)
        if not os.path.isdir(sub_exp_path):
            continue

        # read yaml config
        config = check_config(sub_exp_path)
        algorithm_name = check_algorithm(config)

        if algorithm_name == 'Unknown Algorithm':
            continue

        acc, loss = read_log_file(sub_exp_path)

        # Skip if log reading failed or returned no data
        if acc is None or loss is None or acc.size == 0:
            print(f"Skipping {sub_exp} due to missing or empty log data.")
            continue

        # check if the algorithm_name exists in eval_dict
        if algorithm_name not in eval_dict:
            eval_dict[algorithm_name] = {'acc': [], 'loss': []}
        eval_dict[algorithm_name]['acc'].append(acc)
        eval_dict[algorithm_name]['loss'].append(loss)

    # check if each algorithm has the same number of runs
    for algorithm_name in eval_dict:
        acc_array = np.array(eval_dict[algorithm_name]['acc'])

        # Your original code checked for exactly 3 runs.
        # I'll keep that check but note that this might be too rigid.
        if acc_array.shape[0] != 3:
            print(f'Warning: {algorithm_name} has a non-standard number of runs: {acc_array.shape[0]}. (Expected 3)')

    avg_eval = {}
    for algorithm_name in eval_dict:
        acc_array = np.array(eval_dict[algorithm_name]['acc'])
        loss_array = np.array(eval_dict[algorithm_name]['loss'])

        # Calculate mean along the runs axis (axis=0)
        avg_acc = np.mean(acc_array, axis=0)
        avg_loss = np.mean(loss_array, axis=0)
        avg_eval[algorithm_name] = {'avg_acc': avg_acc, 'avg_loss': avg_loss}

    print(f'\n--- Results Summary for {dataset_name} ({lr}, {lstep}) ---')

    # Check if the algorithms exist in avg_eval before printing
    fedlora2_max_acc = max(avg_eval["FedLoRA2"]["avg_acc"]) if "FedLoRA2" in avg_eval and len(
        avg_eval["FedLoRA2"]["avg_acc"]) > 0 else 'N/A'
    fedlora_wo_rot_max_acc = max(
        avg_eval["FedLoRA2_wo_Rotation"]["avg_acc"]) if "FedLoRA2_wo_Rotation" in avg_eval and len(
        avg_eval["FedLoRA2_wo_Rotation"]["avg_acc"]) > 0 else 'N/A'
    rolora_max_acc = max(avg_eval["RoLoRA"]["avg_acc"]) if "RoLoRA" in avg_eval and len(
        avg_eval["RoLoRA"]["avg_acc"]) > 0 else 'N/A'
    ffalora_max_acc = max(avg_eval["FFA-LoRA"]["avg_acc"]) if "FFA-LoRA" in avg_eval and len(
        avg_eval["FFA-LoRA"]["avg_acc"]) > 0 else 'N/A'

    print(f'FedLoRA2 Max Acc: {fedlora2_max_acc}')
    print(f'FedLoRAwoRotation Max Acc: {fedlora_wo_rot_max_acc}')
    print(f'RoLoRA Max Acc: {rolora_max_acc}')
    print(f'FFA-LoRA Max Acc: {ffalora_max_acc}')
    print('---------------------------------------------------------')


# 5. Entry point
if __name__ == "__main__":
    main()