# HOW TO RUN:
# You need to run this script from a configured edgeai-tidl-tools environment.
# That means:
# 1. git clone https://github.com/TexasInstruments/edgeai-tidl-tools.git -b 10_00_08_00
# 2. Follow the setup instructions in the edgeai-tidl-tools repository to set up the environment.
# 3. Copy this script to examples/osrt_python/tvm_dlr
# 4. cd examples/osrt_python/tvm_dlr
# 5. Run the script with the appropriate arguments:
# python quick_export_tidl.py -m <path_to_tflite_models> -c <path_to_calibration_npy>

# EXPORT NOTES:
# DeiT-based models are not supported by TIDL compiler due to:
# tvm.error.OpNotImplemented: The following operators are not supported in frontend TFLite: 'GELU'


import os
import os.path as osp
import sys
import shutil
import argparse
import onnx
import tflite
from tvm import relay
from tvm.relay.backend.contrib import tidl

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)

sys.path.append(parent)
from common_utils import *
from model_configs import *


def export_tidl(
    model_path: str,
    calibration_npy: str,
    output_zip_path: str,
    num_bits=8,
    num_subgraphs_max=16,
    calibration_frames=256,
    calibration_iterations=1,
):
    if not output_zip_path.endswith(".zip"):
        raise ValueError("Output path must end with .zip")
    is_tflite = model_path.endswith(".tflite")
    model_name = (
        os.path.basename(model_path).replace(".onnx", "").replace(".tflite", "")
    )
    model_input_name = "serving_default_images:0" if is_tflite else "images"
    model_input_shape = (1, 256, 256, 3) if is_tflite else (1, 3, 256, 256)
    model_input_dtype = "float32"
    model_output_directory = artifacts_folder + model_name

    # TIDL compiler specifics
    # We are compiling the model for J7 device using
    # a compiler distributed with SDK 7.0
    DEVICE = os.environ["SOC"]
    SDK_VERSION = (7, 0)

    # convert the model to relay IR format

    if model_path.endswith(".onnx"):

        print(model_path)
        onnx_model = onnx.load(model_path)
        mod, params = relay.frontend.from_onnx(
            onnx_model, shape={model_input_name: model_input_shape}
        )
    elif model_path.endswith(".tflite"):

        with open(model_path, "rb") as fp:
            tflite_model = tflite.Model.GetRootAsModel(fp.read(), 0)
            mod, params = relay.frontend.from_tflite(
                tflite_model,
                shape_dict={model_input_name: model_input_shape},
                dtype_dict={model_input_name: model_input_dtype},
            )
    else:
        raise ValueError(
            "Unsupported model format. Please provide an ONNX or TFLite model."
        )

    build_target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu"
    cross_cc_args = {
        "cc": os.path.join(
            os.environ["ARM64_GCC_PATH"], "bin", "aarch64-none-linux-gnu-gcc"
        )
    }
    model_output_directory = model_output_directory + "_device"
    model_output_directory = model_output_directory + "/artifacts"

    # create the directory if not present
    # clear the directory
    os.makedirs(model_output_directory, exist_ok=True)
    for root, dirs, files in os.walk(model_output_directory, topdown=False):
        [os.remove(os.path.join(root, f)) for f in files]
        [os.rmdir(os.path.join(root, d)) for d in dirs]

    assert num_bits in [8, 16, 32]
    assert num_subgraphs_max <= 16

    # Use advanced calibration for 8-bit quantization
    # Use simple calibration for 16-bit quantization and float-mode
    advanced_options = {
        8: {
            "calibration_iterations": calibration_iterations,
            "calibration_frames": calibration_frames,
            # below options are set to default values, include here for reference
            "quantization_scale_type": 0,
            "high_resolution_optimization": 0,
            "pre_batchnorm_fold": 1,
            # below options are only overwritable at accuracy level 9, otherwise ignored
            "activation_clipping": 1,
            "weight_clipping": 1,
            "bias_calibration": 1,
            "channel_wise_quantization": 0,
        },
        16: {
            "calibration_iterations": 1,
        },
        32: {
            "calibration_iterations": 1,
        },
    }

    calibration_inputs = np.load(calibration_npy)

    def process_input_data(input_data: np.ndarray):
        input_data = np.expand_dims(input_data, axis=0)
        if input_data.ndim != len(model_input_shape):
            raise ValueError(
                f"Input data must have {len(model_input_shape)} dimensions, but got {input_data.ndim}"
            )
        if model_input_shape[3] == 3:
            input_data = np.transpose(input_data, (0, 2, 3, 1))
        return input_data

    calib_input_list = [
        {model_input_name: process_input_data(input_data)}
        for input_data in calibration_inputs[:calibration_frames]
    ]

    # Create the TIDL compiler with appropriate parameters
    compiler = tidl.TIDLCompiler(
        DEVICE,
        SDK_VERSION,
        tidl_tools_path=os.environ["TIDL_TOOLS_PATH"],
        artifacts_folder=model_output_directory,
        tensor_bits=num_bits,
        debug_level=2,
        max_num_subgraphs=num_subgraphs_max,
        c7x_codegen=0,
        accuracy_level=(1 if num_bits == 8 else 0),
        advanced_options=advanced_options[num_bits],
    )

    # partition the graph into TIDL operations and TVM operations
    mod, status = compiler.enable(mod, params, calib_input_list)

    # build the relay module into deployables
    with tidl.build_config(tidl_compiler=compiler):
        graph, lib, params = relay.build_module.build(
            mod, target=build_target, params=params
        )

    # remove nodes / params not needed for inference
    tidl.remove_tidl_params(params)

    # save the deployables
    path_lib = os.path.join(model_output_directory, "deploy_lib.so")
    path_graph = os.path.join(model_output_directory, "deploy_graph.json")
    path_params = os.path.join(model_output_directory, "deploy_params.params")

    lib.export_library(path_lib, **cross_cc_args)
    with open(path_graph, "w") as fo:
        fo.write(graph)
    with open(path_params, "wb") as fo:
        fo.write(relay.save_param_dict(params))

    # Delete tempDir if it exists
    temp_dir = os.path.join(model_output_directory, "tempDir")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    # Create zip archive of the artifacts
    output_zip_path_without_ext = osp.splitext(output_zip_path)[0]
    shutil.make_archive(output_zip_path_without_ext, "zip", model_output_directory)
    print(f"Exported TIDL artifacts to {output_zip_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export TIDL artifacts from TFLite models."
    )
    parser.add_argument(
        "-m",
        "--models_dir",
        type=str,
        required=True,
        help="Directory containing TFLite models to export.",
    )
    parser.add_argument(
        "-c",
        "--calibration_npy",
        type=str,
        required=True,
        help="Path to the calibration numpy file.",
    )
    args = parser.parse_args()
    tf_model_paths = sorted(
        [
            osp.join(args.models_dir, tf_path)
            for tf_path in os.listdir(args.models_dir)
            if tf_path.endswith("onnx2tf.tflite") and "Deit" not in tf_path
        ]
    )

    print(f"Found {len(tf_model_paths)} TFLite models in {args.models_dir}:")
    for tf_model_path in tf_model_paths:
        print(f" - {tf_model_path}")

    for tf_model_path in tf_model_paths:
        print(f"Exporting {tf_model_path}")
        model_name = (
            os.path.basename(tf_model_path).replace(".onnx", "").replace(".tflite", "")
        )
        output_zip_path = osp.join(args.models_dir, f"{model_name}_tidl.zip")
        export_tidl(tf_model_path, args.calibration_npy, output_zip_path)
        print(f"Exported {tf_model_path} to {output_zip_path}")
