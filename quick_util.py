import os
import json
from prettytable import PrettyTable

COPY_TO_CLIPBOARD = True
TRAIN_EXTRA_ARGS = ""

work_dir = "./work_dirs"
config_folder = "configs/body_2d_keypoint/topdown_heatmap/mpii/"

train_script = "quick_train.py"
eval_script = "quick_eval.py"
export_script = "quick_export.py"
analyze_script = "quick_analyze.py"


all_config_files = sorted(
    [
        os.path.join(config_folder, config_file)
        for config_file in os.listdir(config_folder)
        if config_file.endswith(".py") and config_file.startswith("0-")
    ]
)
exp_names = sorted(
    [exp_name for exp_name in os.listdir(work_dir) if exp_name.startswith("0-")]
)


train_config_files = [c for c in all_config_files if "-light" in c]
export_config_files = [c for c in all_config_files if "-light" in c]


def _execute_or_copy_command(config_files, extra_args, script):
    command = ""
    for config_file in config_files:
        print(f"Running {script} with config: {config_file}")
        command += f"python {script} {config_file}{extra_args};"

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
    _execute_or_copy_command(config_files, "", eval_script)


def orchestrate_export(config_files):
    _execute_or_copy_command(config_files, "", export_script)


def orchestrate_analyze(config_files):
    _execute_or_copy_command(config_files, "", analyze_script)


def orchestrate_tfmicro(models_dir: str):
    tf_models = sorted(
        [
            tf_path
            for tf_path in os.listdir(models_dir)
            if tf_path.endswith("int8.tflite") and "-light" in tf_path
        ]
    )
    command = ""

    for model in tf_models:
        output_dir = os.path.join(
            models_dir, model.replace(".tflite", "_tfmicro_resolver")
        )
        os.makedirs(output_dir, exist_ok=True)
        command += f"python tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver/generate_micro_mutable_op_resolver_from_model.py --common_tflite_path={models_dir} --input_tflite_files={model} --output_dir={output_dir};"

    print("Command:", command.strip())
    if COPY_TO_CLIPBOARD:
        os.system(f'echo "{command.strip()}" | clip.exe')
        print("Command copied to clipboard.")
    else:
        os.system(command.strip())


def create_model_summary():
    model_summary = {exp_name: {} for exp_name in exp_names}
    for exp_name in exp_names:
        eval_result_file = os.path.join(work_dir, exp_name, "eval_result.json")
        analyze_result_file = os.path.join(work_dir, exp_name, "analyze_result.json")
        if not os.path.exists(eval_result_file) or not os.path.exists(
            analyze_result_file
        ):
            print(f"Skipping {exp_name} as one of the result files is missing.")
            continue
        with open(eval_result_file, "r") as f:
            eval_result = json.load(f)
        with open(analyze_result_file, "r") as f:
            analyze_result = json.load(f)
        model_summary[exp_name].update(eval_result)
        model_summary[exp_name].update(analyze_result)

    # Prepare headers and rows
    table = PrettyTable()
    table.field_names = [
        "Model Name",
        "Params",
        "FLOPs",
        "Peak Memory",
        "Iterations",
        "Latency",
        "PCK",
        "PCK@0.1",
        "PCK-AUC",
    ]
    for exp_name, result in model_summary.items():
        print(f"Processing {exp_name}...")
        table.add_row(
            [
                exp_name,
                f"{result['params'] / 1e6:.2f}M",
                f"{result['flops'] / 1e9:.2f}G",
                f"{result['peak_memory_mb']:.2f}MB",
                f"{result['iterations']}",
                f"{result['avg_latency_ms']:.2f}ms",
                f"{result['PCK']:.2f}",
                f"{result['PCK@0.1']:.2f}",
                f"{result['PCK-AUC']:.2f}",
            ]
        )

    print("Model Summary:")
    print(table)

    # Save to JSON
    summary_path = os.path.join(work_dir, "model_summary.json")
    with open(summary_path, "w") as f:
        json.dump(model_summary, f, indent=4)
    print(f"Model summary saved to {summary_path}")


if __name__ == "__main__":
    # orchestrate_training(train_config_files)
    # orchestrate_evaluation(all_config_files)
    # orchestrate_analyze(all_config_files)
    # orchestrate_export(export_config_files)
    # orchestrate_tfmicro("/home/hig/src/mmpose/work_dirs/exported_models")
    create_model_summary()
