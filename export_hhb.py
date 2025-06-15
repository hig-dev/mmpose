import os
import numpy as np
import shutil


calibration_size = 256

# EXPORT NOTES:
# Export of efficientvit models fails with:
# Unspported ops ['nn.batch_matmul'] in target th1520
# Export of DeiT models fail with:
# For ONNX model: The following operators are not supported for frontend ONNX: LayerNormalization
# For TFLite model: The following operators are not supported in frontend TFLite: 'BATCH_MATMUL', 'GELU'


def export_hhb(
    model_path: str,
    calibration_npz_path: str,
    skip=False,
    quantization_scheme: str = "uint8_asym",
    hw_arch: str = "th1520",
):
    base_model_name = os.path.basename(model_path).replace(".onnx", "").replace(".tflite", "")
    output_path = model_path.replace(".tflite", f"{quantization_scheme}_hbb").replace(".onnx", f"{quantization_scheme}_hbb")
    if skip and os.path.exists(output_path):
        print(f"File {output_path} already exists, skipping export.")
        return
    
    cmd = f"hhb -D \
    -f {model_path} \
    --input-name images \
    --output-name output \
    --input-shape '1 3 256 256' \
    --quantization-scheme {quantization_scheme} \
    --board {hw_arch} \
    --calibrate-dataset {calibration_npz_path} \
    --with-py-wrapper \
    --output {output_path}"
    state = os.system(cmd)
    if state or not os.path.exists(output_path):
        raise RuntimeError(f"Export failed with state {state} for {model_path} to {output_path}")
    
    # Compile model wrapper
    compile_command = "riscv64-unknown-linux-gnu-gcc model_wrapper.c model.c -fPIC -o model_wrapper.so io.c -I . -I /home/hig/src/mmpose/.hailovenv/lib/python3.10/site-packages/hhb/install_nn2/th1520/include/ -I /home/hig/src/mmpose/.hailovenv/lib/python3.10/site-packages/hhb/install_nn2/th1520/include/shl_public -L /home/hig/src/mmpose/.hailovenv/lib/python3.10/site-packages/hhb/install_nn2/th1520/lib/ -lshl -lm -shared"
    state = os.system(f'cd {output_path} && {compile_command}')
    if state:
        raise RuntimeError(f"Compilation failed with state {state} for {model_path} to {output_path}")
    
    print(f"Exported {model_path} to {output_path}")
    # Now zip the output directory
    output_zip_path = os.path.join(
        os.path.dirname(model_path),
        f"{base_model_name}_{quantization_scheme}_hbb"
    )
    shutil.make_archive(output_zip_path, 'zip', output_path)
    



if __name__ == "__main__":
    pwd = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(pwd, "work_dirs/exported_models")
    calibration_data_npy = "work_dirs/mpii_test_data/calibration_inputs.npy"
    calibration_data_npz = "work_dirs/mpii_test_data/calibration_inputs.npz"
    calibration_data_nchw = np.load(calibration_data_npy)
    np.savez(
        calibration_data_npz,
        images=calibration_data_nchw[:calibration_size],
    )

    if calibration_data_nchw.shape[0] < 1024:
        raise ValueError(
            f"Calibration data size {calibration_data_nchw.shape[0]} is less than 1024."
        )

    onnx_model_paths = sorted(
        [
            os.path.join(models_dir, onnx_path)
            for onnx_path in os.listdir(models_dir)
            if onnx_path.endswith("-light_v17.onnx") and "mobileone" in onnx_path
        ]
    )

    print(f"Found {len(onnx_model_paths)} ONNX models in {models_dir}")

    for onnx_model_path in onnx_model_paths:
        print(f"Exporting {onnx_model_path}")
        export_hhb(onnx_model_path, calibration_data_npz, quantization_scheme="uint8_asym", skip=False)
        export_hhb(onnx_model_path, calibration_data_npz, quantization_scheme="int16_sym", skip=False)
        print(f"Exported {onnx_model_path}")
