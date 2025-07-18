import argparse
import os
import os.path as osp
import json
import numpy as np
from torch.utils.data import DataLoader
from mmengine.config import Config
from mmengine.runner import Runner
from mmpose.structures.pose_data_sample import PoseDataSample

device = "cpu"

class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def export_test_data(dataloader: DataLoader, data_preprocessor, output_dir: str) -> np.ndarray:
    """Export test data from the dataloader.
    Args:
        dataloader (DataLoader): The dataloader to export data from.
        data_preprocessor: The data preprocessor to apply to the data.
        output_dir (str): The directory to save the exported data.
    Returns:
        np.ndarray: The first sample input from the dataloader.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_inputs = []
    gt_keypoints = []
    gt_keypoints_visible_mask = []
    gt_head_sizes = []
    meta_infos = []
    for data in dataloader:
        preprocessed_data = data_preprocessor(data)
        model_input = preprocessed_data["inputs"]
        data_sample: PoseDataSample = preprocessed_data["data_samples"][0]
        meta_info = data_sample.metainfo
        gt_instance = data_sample.gt_instances[0]
        gt_keypoints_visible = gt_instance.keypoints_visible.astype(bool)
        if gt_keypoints_visible.ndim == 3:
            gt_keypoints_visible = gt_keypoints_visible[:, :, 0]
        gt_keypoints_visible = gt_keypoints_visible.reshape(1, -1)
        head_size_ = gt_instance["head_size"]
        head_size = np.array([head_size_, head_size_]).reshape(-1, 2)

        model_inputs.append(model_input.numpy())
        gt_keypoints.append(gt_instance.keypoints)
        gt_keypoints_visible_mask.append(gt_keypoints_visible)
        gt_head_sizes.append(head_size)
        meta_infos.append(meta_info)
    model_inputs_np = np.concatenate(model_inputs, axis=0)
    gt_keypoints_np = np.concatenate(gt_keypoints, axis=0)
    gt_keypoints_visible_mask_np = np.concatenate(gt_keypoints_visible_mask, axis=0)
    gt_head_sizes_np = np.concatenate(gt_head_sizes, axis=0)

    np.save(osp.join(output_dir, "model_inputs.npy"), model_inputs_np)
    np.save(osp.join(output_dir, "gt_keypoints.npy"), gt_keypoints_np)
    np.save(
        osp.join(output_dir, "gt_keypoints_visible_mask.npy"),
        gt_keypoints_visible_mask_np,
    )
    np.save(osp.join(output_dir, "gt_head_sizes.npy"), gt_head_sizes_np)
    with open(osp.join(output_dir, "meta_infos.json"), "w") as f:
        json.dump(meta_infos, f, cls=NumpyEncoder)
    return model_inputs_np[0]

def export_sample_input(sample_input: np.ndarray, output_path: str):
    if len(sample_input.shape) != 4:
        raise ValueError(f"Sample input should be 4D, but got {sample_input.shape}")
    if sample_input.dtype != np.float32:
        raise ValueError(
            f"Sample input should be float32, but got {sample_input.dtype}"
        )

    np.save(output_path, sample_input)

def main(config_file: str):
    # load config
    cfg = Config.fromfile(config_file)
    exp_name = osp.splitext(osp.basename(config_file))[0]
    cfg.work_dir = osp.join("./work_dirs", exp_name)
    test_data_output_dir = osp.join("./work_dirs", f"mpii_test_data2")
    os.makedirs(test_data_output_dir, exist_ok=True)
    cfg.train_dataloader.batch_size = 1
    cfg.test_dataloader.batch_size = 1
    # build the runner from config
    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    runner.model.to(device)

    test_dataloader: DataLoader = runner.test_dataloader
    sample_input = export_test_data(
        test_dataloader, runner.model.data_preprocessor, test_data_output_dir
    )
    sample_input_path_nchw = osp.join(test_data_output_dir, f"sample_input_nchw.npy")
    sample_input_path_nhwc = osp.join(test_data_output_dir, f"sample_input_nhwc.npy")

    sample_input = np.expand_dims(sample_input, axis=0)
    export_sample_input(sample_input, sample_input_path_nchw)
    export_sample_input(
        sample_input.transpose(0, 2, 3, 1), sample_input_path_nhwc
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export test data for a mmpose config")
    parser.add_argument("config", help="train config file path")
    args = parser.parse_args()
    main(args.config)