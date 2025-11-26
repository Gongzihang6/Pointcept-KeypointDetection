"""
Main Training Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch


def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )

"""
## 基于PointTransformerV3的关键点预测模型
python tools/train.py --config-file configs/my_dataset/keypoint_ptv3.py
## 基于PointTransformerV3的关键点预测模型从中断点恢复训练
python tools/train.py --config-file configs/my_dataset/keypoint_ptv3.py --options resume=True weight=exp/default/model/model_last.pth

## 基于OctFormer的关键点预测模型
python tools/train.py --config-file configs/my_dataset/keypoint_octformer.py 

"""
if __name__ == "__main__":
    main()
