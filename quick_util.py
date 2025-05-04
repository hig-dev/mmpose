import os
import json
from prettytable import PrettyTable

COPY_TO_CLIPBOARD = True
TRAIN_EXTRA_ARGS = ''

work_dir = './work_dirs'
config_folder = 'configs/body_2d_keypoint/topdown_heatmap/mpii/'

train_script = 'quick_train.py'
eval_script = 'quick_eval.py'
export_script = 'quick_export.py'
analyze_script = 'quick_analyze.py'


all_config_files = [os.path.join(config_folder, config_file) for config_file in os.listdir(config_folder) if config_file.endswith('.py') and config_file.startswith('0-')] 
exp_names = sorted([exp_name for exp_name in os.listdir(work_dir) if exp_name.startswith('0-')])


train_config_files = [c for c in all_config_files if '-light' in c]

def _execute_or_copy_command(config_files, extra_args, script):
    command = ""
    for config_file in config_files:
        print(f'Running {script} with config: {config_file}')
        command += f'python {script} {config_file}{extra_args};'

    print("Command:", command.strip())

    if COPY_TO_CLIPBOARD:
        # Copy the command to clipboard
        os.system(f'echo "{command.strip()}" | clip.exe')
        print("Command copied to clipboard.")
    else:
        # Execute the command
        os.system(command.strip())

def orchestrate_training(config_files):
    _execute_or_copy_command(config_files, TRAIN_EXTRA_ARGS, train_script)

def orchestrate_evaluation(config_files):
    _execute_or_copy_command(config_files, '', eval_script)

def orchestrate_export(config_files):
    _execute_or_copy_command(config_files, '', export_script)

def orchestrate_analyze(config_files):
    _execute_or_copy_command(config_files, '', analyze_script)

def create_model_summary():
    model_summary = {exp_name: {} for exp_name in exp_names}
    for exp_name in exp_names:
        eval_result_file = os.path.join(work_dir, exp_name, 'eval_result.json')
        analyze_result_file = os.path.join(work_dir, exp_name, 'analyze_result.json')
        with open(eval_result_file, 'r') as f:
            eval_result = json.load(f)
        with open(analyze_result_file, 'r') as f:
            analyze_result = json.load(f)
        model_summary[exp_name].update(eval_result)
        model_summary[exp_name].update(analyze_result)

    # Prepare headers and rows
    table = PrettyTable()
    table.field_names = ["Model Name", "Params", "FLOPs", "Peak Memory", "PCK", "PCK-AUC"]
    for exp_name, result in model_summary.items():
        table.add_row([
            exp_name,
            f"{result['params'] / 1e6:.2f}M",
            f"{result['flops'] / 1e9:.2f}G",
            f"{result['peak_memory_mb']:.2f}MB",
            f"{result['PCK']:.2f}",
            #f"{result['PCK@0.1']:.2f}",
            f"{result['PCK-AUC']:.2f}",
        ])

    print("Model Summary:")
    print(table)

    # Save to JSON
    summary_path = os.path.join(work_dir, "model_summary.json")
    with open(summary_path, "w") as f:
        json.dump(model_summary, f, indent=4)
    print(f"Model summary saved to {summary_path}")

        


if __name__ == "__main__":
    #orchestrate_training(train_config_files)
    #orchestrate_evaluation(all_config_files)
    #orchestrate_analyze(all_config_files)
    #orchestrate_export(all_config_files)
    create_model_summary()


    