# Contains code from
# - https://github.com/Seeed-Studio/ModelAssistant/blob/main/tools/export.py
# - https://github.com/open-mmlab/mmdeploy/blob/main/mmdeploy/codebase/base/task.py
import argparse
import os.path as osp
import os
import numpy as np
from mmcv.cnn.bricks.drop import DropPath
import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.export import export as export_torch
import onnx
import ai_edge_torch
import tensorflow as tf
from onnx2tf import onnx2tf

from mmengine.config import Config
from mmengine.runner import Runner
from torch.utils.data import DataLoader
from mmengine.model import revert_sync_batchnorm
from mmpose.structures.pose_data_sample import PoseDataSample

device = "cpu"


def export(config_file: str, export_calibration: bool = False):
    tf.config.set_visible_devices([], "GPU")

    # load config
    cfg = Config.fromfile(config_file)
    exp_name = osp.splitext(osp.basename(config_file))[0]
    cfg.work_dir = osp.join("./work_dirs", exp_name)
    test_data_output_dir = osp.join("./work_dirs", f"mpii_test_data")
    exported_models_dir = osp.join("./work_dirs", f"exported_models")
    os.makedirs(test_data_output_dir, exist_ok=True)
    os.makedirs(exported_models_dir, exist_ok=True)

    # get best checkpoint
    pth_files = [
        f
        for f in os.listdir(cfg.work_dir)
        if f.endswith(".pth") and f.startswith("best_PCK_epoch_")
    ]
    best_pth = sorted(pth_files)[-1]
    print(f"Best checkpoint: {best_pth}")

    cfg.load_from = osp.join(cfg.work_dir, best_pth)
    cfg.train_dataloader.batch_size = 256
    cfg.test_dataloader.batch_size = 1

    # build the runner from config
    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()

    train_dataloader: DataLoader = runner.train_dataloader

    # get the first batch of data for calibration
    calibration_data = export_calibration_data(
        train_dataloader,
        runner.model.data_preprocessor,
        test_data_output_dir,
        limit=1000,
        save=export_calibration,
    )

    sample_input = calibration_data[0]
    sample_input = np.expand_dims(sample_input, axis=0)

    # prepare model
    model = runner.model
    model = prepare_model_for_export(model)
    model = model.to(device)

    # Export to ExecuTorch
    executorch_output_path = osp.join(exported_models_dir, f"{exp_name}.pte")
    export_executorch(model, sample_input, executorch_output_path)

    # Export to ONNX
    onnx_output_path = osp.join(exported_models_dir, f"{exp_name}.onnx")
    # Export to ONNX with opset 17 (needed for Hailo DataFlow Compiler)
    onnx_output_path_v17 = osp.join(exported_models_dir, f"{exp_name}_v17.onnx")
    export_onnx(model, sample_input, onnx_output_path)
    export_onnx(model, sample_input, onnx_output_path_v17, onnx_opset=17)

    # Export to TFLite (using ai_edge_torch)
    tflite_aiedgetorch_output_path = osp.join(exported_models_dir, f"{exp_name}_aiedgetorch.tflite")
    tflite_aiedgetorch_quantized_output_path = osp.join(
        exported_models_dir, f"{exp_name}_aiedgetorch_int8.tflite"
    )
    export_tflite_ai_edge_torch(
        model, calibration_data, tflite_aiedgetorch_output_path, quantize=False
    )
    export_tflite_ai_edge_torch(
        model, calibration_data, tflite_aiedgetorch_quantized_output_path, quantize=True
    )

    # Export TFLite from ONNX (using onnx2tf)
    tf_saved_model_dir = osp.join(exported_models_dir, f"{exp_name}_tf_saved_model")
    export_tf_saved_model(onnx_output_path, tf_saved_model_dir)
    tflite_onnx2tf_output_path = osp.join(
        exported_models_dir, f"{exp_name}_onnx2tf.tflite"
    )
    tflite_onnx2tf_quantized_output_path = osp.join(
        exported_models_dir, f"{exp_name}_onnx2tf_int8.tflite"
    )
    export_tflite_onnx2tf(
        tf_saved_model_dir, calibration_data, tflite_onnx2tf_output_path, quantize=False
    )
    export_tflite_onnx2tf(
        tf_saved_model_dir,
        calibration_data,
        tflite_onnx2tf_quantized_output_path,
        quantize=True,
    )

def prepare_model_for_export(model):
    model.forward = model._forward
    model = revert_sync_batchnorm(model)
    if hasattr(model, "backbone") and hasattr(model.backbone, "switch_to_deploy"):
        model.backbone.switch_to_deploy()

    def identity_forward(x):
        return x

    # replace DropPath forward method to avoid conditional path
    for m in model.modules():
        if isinstance(m, DropPath):
            m.forward = identity_forward

    model.eval()
    return model

def export_onnx(model, sample_input: np.ndarray, output_path, onnx_opset=20):
    torch.onnx.export(
        model,
        torch.from_numpy(sample_input),
        output_path,
        verbose=False,
        input_names=["images"],
        output_names=["output"],
        opset_version=onnx_opset,
    )
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model exported to {output_path}")

def export_executorch(model, sample_input: np.ndarray, output_path):
    exported_program = export_torch(model, (torch.from_numpy(sample_input),))
    executorch_program = to_edge_transform_and_lower(
        exported_program,
        partitioner = [XnnpackPartitioner()]
    ).to_executorch()

    with open(output_path, "wb") as file:
        file.write(executorch_program.buffer)

def export_tflite_ai_edge_torch(model, calibration_data, output_path, quantize=False):
    sample_input = torch.from_numpy(calibration_data[0])
    if len(sample_input.shape) == 3:
        sample_input = sample_input.unsqueeze(0)

    tfl_converter_flags = None
    if quantize:

        def representative_dataset_gen():
            for calibration_input in calibration_data:
                calibration_input = np.expand_dims(calibration_input, axis=0)
                tf_input = tf.constant(calibration_input, dtype=tf.float32)
                yield [tf_input]

        tfl_converter_flags = {
            "optimizations": [tf.lite.Optimize.DEFAULT],
            "target_spec.supported_ops": [tf.lite.OpsSet.TFLITE_BUILTINS_INT8],
            "inference_type": tf.int8,
            "inference_input_type": tf.int8,
            "inference_output_type": tf.int8,
            "representative_dataset": representative_dataset_gen,
            "_experimental_disable_per_channel": False,
        }

    edge_model = ai_edge_torch.convert(
        model, (sample_input,), _ai_edge_converter_flags=tfl_converter_flags
    )
    edge_model.export(output_path)
    print(f"TFLite model exported to {output_path}")


def export_tf_saved_model(onnx_path, output_dir):
    onnx2tf.convert(
        onnx_path,
        output_folder_path=output_dir,
        custom_input_op_name_np_data_path=None,
        output_signaturedefs=True,
        verbosity="warn",
    )


def export_tflite_onnx2tf(
    tf_saved_model_path, calibration_data: np.ndarray, output_path, quantize=False
):
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_path)

    if quantize:

        def representative_dataset_gen():
            for calibration_input in calibration_data:
                calibration_input = np.expand_dims(calibration_input, axis=0)
                tf_input = tf.constant(calibration_input, dtype=tf.float32)
                tf_input_transposed = tf.transpose(tf_input, perm=[0, 2, 3, 1])
                yield [tf_input_transposed]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.representative_dataset = representative_dataset_gen
        converter._experimental_disable_per_channel = False

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"TFLite model exported to {output_path}")

def export_calibration_data(dataloader: DataLoader, data_preprocessor, output_dir: str, limit: int = 1000, save: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    batch_size = dataloader.batch_size
    model_inputs = []
    for data in dataloader:
        preprocessed_data = data_preprocessor(data)
        model_input = preprocessed_data["inputs"]
        model_inputs.append(model_input.cpu().numpy())
        if batch_size * len(model_inputs) >= limit:
            break
    model_inputs_np = np.concatenate(model_inputs, axis=0)
    if save:
        np.save(osp.join(output_dir, "calibration_inputs.npy"), model_inputs_np)
    return model_inputs_np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a model")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--export-calibration-data", action="store_true", help="Export calibration data")

    args = parser.parse_args()
    export(args.config, args.export_calibration_data)
