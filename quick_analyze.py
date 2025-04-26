import argparse
import os
import os.path as osp
import json

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.analysis import get_model_complexity_info
import torch
from quick_export import prepare_model_for_export


def analyze(config_file: str):
    # load config
    cfg = Config.fromfile(config_file)
    exp_name = osp.splitext(osp.basename(config_file))[0]
    cfg.work_dir = osp.join("./work_dirs", exp_name)

    # get best checkpoint
    pth_files = [
        f
        for f in os.listdir(cfg.work_dir)
        if f.endswith(".pth") and f.startswith("best_PCK_epoch_")
    ]
    best_pth = sorted(pth_files)[-1]
    print(f"Best checkpoint: {best_pth}")

    cfg.load_from = osp.join(cfg.work_dir, best_pth)
    cfg.test_dataloader.batch_size = 1

    # build the runner from config
    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()

    model = prepare_model_for_export(runner.model)

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        # run a dummy forward pass to warm up the model
        input_shape = (1, 3, cfg.codec.input_size[0], cfg.codec.input_size[1])
        dummy_input = torch.randn(input_shape).cuda()
        model(dummy_input)

    peak_memory_bytes = torch.cuda.max_memory_allocated()
    peak_memory_mb = peak_memory_bytes / (1024**2)
    print(f"Peak CUDA memory: {peak_memory_mb:.2f} MB")

    # analyse model complexity
    input_shape = (3, cfg.codec.input_size[0], cfg.codec.input_size[1])
    complexity_info = analyse_complexity(model, input_shape)

    analyze_result = {
        "model_name": exp_name,
        "peak_memory_mb": peak_memory_mb,
    }
    analyze_result.update(complexity_info)

    with open(osp.join(cfg.work_dir, "analyze_result.json"), "w") as f:
        json.dump(analyze_result, f, indent=4)

    print(f'Analyze result saved to {osp.join(cfg.work_dir, "analyze_result.json")}')


def analyse_complexity(model, input_shape):
    model.forward = model._forward
    complexity_info = get_model_complexity_info(
        model=model,
        input_shape=input_shape,
        inputs=None,
        show_table=True,
        show_arch=False,
    )

    print("Complexity Info:")
    for key, value in complexity_info.items():
        print(f"{key}: {value}")
    return complexity_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval a model")
    parser.add_argument("config", help="config file path")
    args = parser.parse_args()
    analyze(args.config)
