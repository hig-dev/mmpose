import os
import os.path as osp

# EXPORT NOTES:
# Export of DeitTiny and DeitSmall models fails with:
# RecursionError: maximum recursion depth exceeded while calling a Python object
# ethos-u-vela version: 4.3.0

calibration_size = 256

def export_vela(
    tflite_path,
    config_path="himax_vela.ini",
    accelerator_config="ethos-u55-64",
    system_config="My_Sys_Cfg",
    memory_mode="My_Mem_Mode_Parent",
):
    cmd = f"vela \
    --config {config_path} \
    --accelerator-config {accelerator_config} \
    --system-config {system_config} \
    --optimise Size \
    --memory-mode {memory_mode} \
    --output-dir {osp.dirname(tflite_path)}/ \
    --recursion-limit 10000 \
    {tflite_path}"
    state = os.system(cmd)
    if not state:
        print("Export of vela model succeeded")
    else:
        print("Export of vela model failed")
    

if __name__ == "__main__":
    pwd = osp.dirname(osp.abspath(__file__))
    models_dir = osp.join(pwd, "work_dirs/exported_models")
    tf_model_paths = sorted(
        [
            osp.join(models_dir, tf_path)
            for tf_path in os.listdir(models_dir)
            if tf_path.endswith("onnx2tf_int8.tflite")
        ]
    )

    print(f"Found {len(tf_model_paths)} TFLite models in {models_dir}:")
    for tf_model_path in tf_model_paths:
        print(f" - {tf_model_path}")

    for tf_model_path in tf_model_paths:
        print(f"Exporting {tf_model_path}")
        export_vela(tf_model_path)
        print(f"Exported {tf_model_path}")
