import os
import numpy as np
from hailo_sdk_client import ClientRunner
from hailo_sdk_client.exposed_definitions import CalibrationDataType

# EXPORT NOTES: 
# hailo-dataflow-compiler version: 3.30.0

# Export of efficientvit_b0-mpii-256x256-light fails with:
# Quantization failed in layer model/ew_mult1 due to unsupported required slope.
# Desired shift is 40.0, but op has only 8 data bits. This error raises when the data or weight range
# are not balanced. Mostly happens when using random calibration-set/weights, the calibration-set is
# not normalized properly or batch-normalization was not used during training.

# Export of efficientvit_b1-mpii-256x256-light fails with:
# Quantization failed in layer model/ew_mult1 due to unsupported required slope.
# Desired shift is 41.0, but op has only 8 data bits. This error raises when the data or weight range
# are not balanced. Mostly happens when using random calibration-set/weights, the calibration-set is
# not normalized properly or batch-normalization was not used during training.

# Export of efficientvit_b2-mpii-256x256-light fails with:
# layer model/matmul3 does not support shift delta. To overcome this issue you should force
# larger range at the inputs of the layer using command
# quantization_param([layer_name], force_range_in=[range_min, range_max], force_range_index=index)
# current range of input 0 is [0.000, 35.384] and input 1 is [-3064.808, 2911.036].
# You should increase the multiplication of these ranges by a factor of 1.636, e.g. you can apply factor of sqrt(1.636) to both inputs:
# quantization_param([model/matmul3], force_range_in=[0.000, 45.258], force_range_index=0)
# quantization_param([model/matmul3], force_range_in=[-3920.080, 3723.396], force_range_index=1)
# Note: Setting the suggested ranges for the matmul3 layer in the model script does not help.

calibration_size = 1024


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
    model_script_lines = [
        f"model_optimization_config(calibration, batch_size=1, calibset_size={calibration_size})\n",
        "model_optimization_flavor(optimization_level=2, compression_level=1)\n",
    ]
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

    mean, std = calibration_data_nchw.mean(axis=(0, 2, 3)), calibration_data_nchw.std(
        axis=(0, 2, 3)
    )
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

    print(f"Found {len(onnx_model_paths)} ONNX models in {models_dir}")

    for onnx_model_path in onnx_model_paths:
        print(f"Exporting {onnx_model_path}")
        try:
            export_hailo(
                onnx_model_path, calibration_data_nhwc[:calibration_size], skip=True
            )
        except Exception as e:
            print(f"Error exporting {onnx_model_path}: {e}")
            continue
        print(f"Exported {onnx_model_path}")
