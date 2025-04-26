import argparse
import os
import json
import os.path as osp

from mmengine.config import Config
from mmengine.runner import Runner

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
    metrics = runner.test()

    eval_result = {
        'model_name': exp_name,
    }
    eval_result.update(metrics)
    
    with open(osp.join(cfg.work_dir, 'eval_result.json'), 'w') as f:
        json.dump(eval_result, f, indent=4)

    print(f'Evaluation result saved to {osp.join(cfg.work_dir, "eval_result.json")}')
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval a model')
    parser.add_argument('config', help='config file path')
    args = parser.parse_args()
    eval(args.config)
