import argparse
import os
import os.path as osp

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.analysis import get_model_complexity_info

def eval(config_file: str):
    # load config
    cfg = Config.fromfile(config_file)
    exp_name = osp.splitext(osp.basename(config_file))[0]
    cfg.work_dir = osp.join('./work_dirs', exp_name)
    
     # get best checkpoint
    pth_files = [f for f in os.listdir(cfg.work_dir) if f.endswith('.pth') and f.startswith('best_PCK_epoch_')]
    best_pth = sorted(pth_files)[-1]
    print(f'Best checkpoint: {best_pth}')
    
    cfg.load_from = osp.join(cfg.work_dir, best_pth) 
    cfg.train_dataloader.batch_size = 1

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # evaluate the model
    runner.test()
    
    # analyse model complexity
    input_shape = (3, cfg.codec.input_size[0], cfg.codec.input_size[1])
    analyse_complexity(runner.model, input_shape)
    

def analyse_complexity(model, input_shape):
    model.forward = model._forward
    complexity_info = get_model_complexity_info(
            model=model,
            input_shape=input_shape,
            inputs=None,
            show_table=True,
            show_arch=False)
    
    print('Complexity Info:')
    for key, value in complexity_info.items():
        print(f'{key}: {value}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval a model')
    parser.add_argument('config', help='config file path')
    args = parser.parse_args()
    eval(args.config)
