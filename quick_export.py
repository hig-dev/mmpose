# Contains code from
# - https://github.com/Seeed-Studio/ModelAssistant/blob/main/tools/export.py
# - https://github.com/open-mmlab/mmdeploy/blob/main/mmdeploy/codebase/base/task.py
import argparse
import os.path as osp
import os
import numpy as np
from mmcv.cnn.bricks.drop import DropPath
import torch
import onnx
import ai_edge_torch
import tensorflow as tf
from onnx2tf import onnx2tf

from mmengine.config import Config
from mmengine.runner import Runner
from torch.utils.data import DataLoader
from mmengine.model import revert_sync_batchnorm
from generate_cc_arrays import generate

onnx_opset = 20
device = "cpu"

def export(config_file: str):
    tf.config.set_visible_devices([], 'GPU')

    # load config
    cfg = Config.fromfile(config_file)
    exp_name = osp.splitext(osp.basename(config_file))[0]
    cfg.work_dir = osp.join('./work_dirs', exp_name)

    # get best checkpoint
    pth_files = [f for f in os.listdir(cfg.work_dir) if f.endswith('.pth') and f.startswith('best_PCK_epoch_')]
    best_pth = sorted(pth_files)[-1]
    print(f'Best checkpoint: {best_pth}')
    
    cfg.load_from = osp.join(cfg.work_dir, best_pth) 
    cfg.train_dataloader.batch_size = 256

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    dataloader: DataLoader = runner.train_dataloader

    # get the first batch of data for calibration
    sample_inputs = next(iter(dataloader))
    sample_inputs = runner.model.data_preprocessor(sample_inputs)
    calibration_data = sample_inputs['inputs']
    calibration_data = calibration_data.to(device)
    sample_input = sample_inputs['inputs'][0]
    sample_input = sample_input.to(device)
    sample_input = sample_input.unsqueeze(0)

    # prepare model
    model = runner.model
    model.forward = model._forward
    model = revert_sync_batchnorm(model)
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'switch_to_deploy'):
        model.backbone.switch_to_deploy()
    
    def identity_forward(x):
        return x
    
    # replace DropPath forward method to avoid conditional path
    for m in model.modules():
        if isinstance(m, DropPath):
            m.forward = identity_forward

    runner.call_hook("before_run")
    model = model.to(device)
    model.eval()

    # Export sample input
    sample_input_path_nchw = osp.join(cfg.work_dir, f'sample_input_nchw.npy')
    sample_input_path_nhwc = osp.join(cfg.work_dir, f'sample_input_nhwc.npy')
    export_sample_input(sample_input.numpy(), sample_input_path_nchw)
    export_sample_input(sample_input.numpy().transpose(0, 2, 3, 1), sample_input_path_nhwc)

    # Export to ONNX
    onnx_output_path = osp.join(cfg.work_dir, f'{exp_name}.onnx')
    export_onnx(model, sample_input, onnx_output_path)

    # Export to TFLite (using ai_edge_torch)
    tflite_output_path = osp.join(cfg.work_dir, f'{exp_name}.tflite')
    tflite_quantized_output_path = osp.join(cfg.work_dir, f'{exp_name}_int8.tflite')
    export_tflite(model, calibration_data, tflite_output_path, quantize=False)
    export_tflite(model, calibration_data, tflite_quantized_output_path, quantize=True)
    
    # Export TFLite from ONNX (using onnx2tf)
    tf_saved_model_dir = osp.join(cfg.work_dir, f'{exp_name}_tf_saved_model')
    export_tf_saved_model(onnx_output_path, tf_saved_model_dir)
    tflite_from_onnx_output_path = osp.join(cfg.work_dir, f'{exp_name}_from_onnx.tflite')
    tflite_from_onnx_quantized_output_path = osp.join(cfg.work_dir, f'{exp_name}_from_onnx_int8.tflite')
    export_tflite_from_onnx(tf_saved_model_dir, calibration_data, tflite_from_onnx_output_path, quantize=False)
    export_tflite_from_onnx(tf_saved_model_dir, calibration_data, tflite_from_onnx_quantized_output_path, quantize=True)

    # Export quantized TFLite to C++ arrays
    cc_output_dir = osp.join(cfg.work_dir, f'{exp_name}_cc')
    export_cc_from_tflite(tflite_quantized_output_path, [sample_input_path_nchw], cc_output_dir)
    export_cc_from_tflite(tflite_from_onnx_quantized_output_path, [sample_input_path_nhwc], cc_output_dir)

def export_sample_input(sample_input: np.array, output_path: str):
    if len(sample_input.shape) != 4:
        raise ValueError(f"Sample input should be 4D, but got {sample_input.shape}")
    if sample_input.dtype != np.float32:
        raise ValueError(f"Sample input should be float32, but got {sample_input.dtype}")
    
    np.save(output_path, sample_input)

def export_onnx(model, sample_input, output_path):
    torch.onnx.export(
        model,
        sample_input,
        output_path,
        verbose=False,
        input_names=["images"],
        output_names=["output"],
        opset_version=onnx_opset,
    )
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model exported to {output_path}")

def export_tflite(model, calibration_data, output_path, quantize=False):
    sample_input = calibration_data[0]
    if len(sample_input.shape) == 3:
        sample_input = sample_input.unsqueeze(0)

    tfl_converter_flags = None
    if quantize:
        def representative_dataset_gen():
            for torch_input in calibration_data:
                torch_input = torch_input.unsqueeze(0)
                tf_input = tf.constant(torch_input.numpy(), dtype=tf.float32)
                yield [tf_input]
        tfl_converter_flags = {
            'optimizations': [tf.lite.Optimize.DEFAULT],
            'target_spec.supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS_INT8],
            'inference_type': tf.int8,
            'inference_input_type': tf.int8,
            'inference_output_type': tf.int8,
            'representative_dataset': representative_dataset_gen,
            '_experimental_disable_per_channel': False,
            }

    edge_model = ai_edge_torch.convert(model, (sample_input,), _ai_edge_converter_flags=tfl_converter_flags)
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

def export_tflite_from_onnx(tf_saved_model_path, calibration_data, output_path, quantize=False):
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_path)

    if quantize:
        def representative_dataset_gen():
            for torch_input in calibration_data:
                torch_input = torch_input.unsqueeze(0)
                tf_input = tf.constant(torch_input.numpy(), dtype=tf.float32)
                tf_input_transposed = tf.transpose(tf_input, perm=[0, 2, 3, 1])
                yield [tf_input_transposed]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.representative_dataset = representative_dataset_gen
        converter._experimental_disable_per_channel = False

    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model exported to {output_path}")
    
def export_cc_from_tflite(tflite_path: str, npy_paths: list[str], output_dir: str):
    inputs = [tflite_path] + npy_paths
    os.makedirs(output_dir, exist_ok=True)
    generate(output_dir, inputs)
    print(f"CC files generated in {output_dir}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export a model')
    parser.add_argument('config', help='train config file path')

    args = parser.parse_args()
    export(args.config)
