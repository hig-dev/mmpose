import argparse
import os.path as osp

from mmengine.config import Config
from mmengine.runner import Runner

from quick_eval import analyse_complexity

def train(config_file: str, resume: bool = False, quick: bool = False):
    # load config
    cfg = Config.fromfile(config_file)

    # set preprocess configs to model
    if 'preprocess_cfg' in cfg:
        cfg.model.setdefault('data_preprocessor', cfg.get('preprocess_cfg', {}))
    
    cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(config_file))[0])
    cfg.resume = resume
    
    if quick:
        cfg.train_cfg.max_epochs = 1

    # build the runner from config
    runner = Runner.from_cfg(cfg)
    runner.train()

    # analyse model complexity
    input_shape = (3, cfg.codec.input_size[0], cfg.codec.input_size[1])
    analyse_complexity(runner.model, input_shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--quick', action='store_true', help='quick train')
    parser.add_argument('--resume', action='store_true', help='resume from the latest checkpoint')
    args = parser.parse_args()

    train(args.config, args.resume, args.quick)
