import os
from tvm import relay, transform
from tvm.relay.op.contrib.arm_compute_lib import partition_for_arm_compute_lib

def export_tvm(
    model_path: str,
    input_name: str = "serving_default_images:0",
    input_shape = (1, 256, 256, 3),
    input_dtype = "float32",
):
    output_path = model_path.replace(".tflite", f"_tvm_acl.so")
    
    tflite_model_buf = open(model_path, "rb").read()

    # Get TFLite model from buffer
    try:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    mod, params = relay.frontend.from_tflite(
        tflite_model, shape_dict={input_name: input_shape}, dtype_dict={input_name: input_dtype}
    )

    mod = partition_for_arm_compute_lib(mod)

    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+neon"
    with transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        lib = relay.build(mod, target, params=params)

    # Save the compiled module
    lib.export_library(output_path)
    
    print(f"Exported {model_path} to {output_path}")
    

if __name__ == "__main__":
    pwd = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(pwd, "work_dirs/exported_models")

    tflite_model_paths = sorted(
        [
            os.path.join(models_dir, tflite_path)
            for tflite_path in os.listdir(models_dir)
            if tflite_path.endswith("-light_onnx2tf.tflite")
        ]
    )

    print(f"Found {len(tflite_model_paths)} TFLite models in {models_dir}")

    for tflite_model_path in tflite_model_paths:
        print(f"Exporting {tflite_model_path}")
        export_tvm(tflite_model_path)
        print(f"Exported {tflite_model_path}")