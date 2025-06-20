import os
import os.path as osp

# EXPORT NOTES:
# Edge TPU Compiler version 16.0.384591198
# The Edge TPU toolchain is not maintained anymore and the last release was in 2021.
# For this reason, it does not work with newer models that use operators not supported by the Edge TPU.

# Export of efficientvit models fails with:
# ERROR: Didn't find op for builtin opcode 'CONV_2D' version '6'.
# An older version of this builtin might be supported.
# Are you using an old TFLite binary with a newer model?
# Export of DeiT models fail with:
# ERROR: Op builtin_code out of range: 150. Are you using old TFLite binary wit
# newer model? ERROR: Registration failed.

calibration_size = 256

def export_edgetpu(tflite_path):
    cmd = f"edgetpu_compiler \
    --out_dir {osp.dirname(tflite_path)}/ \
    {tflite_path}"
    state = os.system(cmd)
    if not state:
        print("Export of edgetpu model succeeded")
    else:
        print("Export of edgetpu model failed")
    

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
        export_edgetpu(tf_model_path)
        print(f"Exported {tf_model_path}")
