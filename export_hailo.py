import os
import numpy as np
from hailo_sdk_client import ClientRunner
from hailo_sdk_client.exposed_definitions import CalibrationDataType


def get_model_script_lines(model: str):
    lines = [
        "model_optimization_config(calibration, batch_size=1, calibset_size=1024)\n",
        "model_optimization_flavor(optimization_level=2, compression_level=1)\n",
    ]
    if "efficientvit_b0-mpii-256x256-light" in model:
        lines.extend(
            [
                #'quantization_param([model/matmul1], force_range_in=[0.000, 100.0], force_range_index=0)\n',
                #'quantization_param([model/matmul1], force_range_in=[-1000.0, 1000.0], force_range_index=1)\n',
            ]
        )
    if "efficientvit_b1-mpii-256x256-light" in model:
        lines.extend(
            [
                #'quantization_param([model/matmul1], force_range_in=[0.000, 15.264], force_range_index=0)\n',
                #'quantization_param([model/matmul1], force_range_in=[-1000.0, 1000.0], force_range_index=1)\n',
            ]
        )
    if "efficientvit_b2-mpii-256x256-light" in model:
        lines.extend(
            [
                #'quantization_param([model/matmul3], force_range_in=[0.000, 100.0], force_range_index=0)\n',
                #'quantization_param([model/matmul3], force_range_in=[-3592.442, 3708.924], force_range_index=1)\n',
            ]
        )

    return lines


def export_hailo(
    model_path: str,
    calibration_data,
    skip=False,
    hw_arch: str = "hailo8",
):
    output_path = model_path.replace(".tflite", ".har").replace(".onnx", ".har")
    output_path_quant = output_path.replace(".har", "_int8.har")
    output_path_hef = output_path.replace(".har", ".hef")
    if skip and os.path.exists(output_path_hef):
        print(f"File {output_path_hef} already exists, skipping export.")
        return
    runner = ClientRunner(hw_arch=hw_arch)
    if model_path.endswith(".tflite"):
        runner.translate_tf_model(
            model_path,
        )
    elif model_path.endswith(".onnx"):
        runner.translate_onnx_model(
            model_path,
        )
    else:
        raise ValueError(f"Unsupported model format: {model_path}")
    runner.save_har(output_path)
    runner = ClientRunner(har=output_path, hw_arch=hw_arch)
    model_script_lines = get_model_script_lines(model_path)
    if model_script_lines:
        model_script = "".join(model_script_lines)
        print(f"Using model script:\n{model_script}")
        runner.load_model_script(model_script)
    runner.optimize(calibration_data, CalibrationDataType.np_array)
    runner.save_har(output_path_quant)
    runner = ClientRunner(har=output_path_quant, hw_arch=hw_arch)
    hef = runner.compile()
    with open(output_path_hef, "wb") as f:
        f.write(hef)


if __name__ == "__main__":
    models_dir = "work_dirs/exported_models"
    calibration_data_npy = "work_dirs/mpii_test_data/calibration_inputs.npy"
    calibration_data_nchw = np.load(calibration_data_npy)
    calibration_data_nhwc = calibration_data_nchw.transpose(0, 2, 3, 1)

    mean, std = calibration_data_nchw.mean(axis=(0, 2, 3)), calibration_data_nchw.std(axis=(0, 2, 3))
    print(f"Mean: {mean}, Std: {std}")

    if calibration_data_nhwc.shape[0] < 1024:
        raise ValueError(
            f"Calibration data size {calibration_data_nhwc.shape[0]} is less than 1024."
        )

    onnx_model_paths = sorted(
        [
            os.path.join(models_dir, onnx_path)
            for onnx_path in os.listdir(models_dir)
            if onnx_path.endswith("-light_v17.onnx")
        ]
    )
    tflite_model_paths = sorted(
        [
            os.path.join(models_dir, tflite_path)
            for tflite_path in os.listdir(models_dir)
            if tflite_path.endswith("-light_onnx2tf.tflite")
        ]
    )

    print(f"Found {len(onnx_model_paths)} ONNX models in {models_dir}")
    print(f"Found {len(tflite_model_paths)} TFLITE models in {models_dir}")

    for onnx_model_path in onnx_model_paths:
        print(f"Exporting {onnx_model_path}")
        try:
            export_hailo(onnx_model_path, calibration_data_nhwc, skip=True)
        except Exception as e:
            print(f"Error exporting {onnx_model_path}: {e}")
            continue
        print(f"Exported {onnx_model_path}")
    
    for tflite_model_path in tflite_model_paths:
        print(f"Exporting {tflite_model_path}")
        try:
            export_hailo(tflite_model_path, calibration_data_nhwc, skip=True)
        except Exception as e:
            print(f"Error exporting {tflite_model_path}: {e}")
            continue
        print(f"Exported {tflite_model_path}")


