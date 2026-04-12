"""
Main Training Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""
import warnings
warnings.filterwarnings("ignore")
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

## 基于PointTransformerV1的关键点预测模型
export PYTHONPATH=.
python tools/train.py --config-file configs/my_dataset/keypoint_ptv1.py

## 基于PointTransformerV2的关键点预测模型
python tools/train.py --config-file configs/my_dataset/keypoint_ptv2.py

## 基于Swin3D的关键点预测模型
python tools/train.py --config-file configs/my_dataset/keypoint_swin3d.py
恢复训练代码
export PYTHONPATH=.
source .venv/bin/activate
python tools/train.py --config-file configs/my_dataset/keypoint_swin3d.py --options resume=True weight=exp/keypoint_swin3d/model/model_last.pth

## 基于OA-CNNs的关键点预测模型
python tools/train.py --config-file configs/my_dataset/keypoint_oa_cnns.py

## 基于StratifiedTransformer的关键点预测模型
python tools/train.py --config-file configs/my_dataset/keypoint_stratified_transformer.py

## 基于sparse_unet的关键点预测模型
python tools/train.py --config-file configs/my_dataset/keypoint_sparse_unet.py

## 基于PointTransformerV3Plus的关键点预测模型
python tools/train.py --config-file configs/my_dataset/keypoint_ptv3_plus.py
# 恢复训练
python tools/train.py --config-file configs/my_dataset/keypoint_ptv3_plus.py --options resume=True weight=exp/keypoint_ptv3_plus/model/model_last.pth

## 基于Swin3DPlus的关键点预测模型
python tools/train.py --config-file configs/my_dataset/keypoint_swin3d_plus.py

## 基于Swin3D的SemanticKITTI语义分割模型（迷你版，快速过拟合测试）
python tools/train.py --config-file configs/semantic_kitti/semseg-swin3d-mini.py
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

## 基于Swin3D的猪场主体点云语义分割模型训练
python tools/train.py --config-file configs/pigseg/semseg-swin3d-v1m1-0-base.py
python tools/infer_npy.py --npy-file /home/gzh/point/Pointcept-KeypointDetection/body_npy_output/test/20260329_165838_641.npy

## 基于SpUNet的猪场主体点云语义分割模型训练
python tools/train.py --config-file configs/pigseg/semseg-spunet-v1m1-0-base.py

## 基于PTV3的猪场主体点云语义分割模型训练
python tools/train.py --config-file configs/pigseg/semseg-ptv3-v1m1-0-base.py
# 恢复训练
python tools/train.py --config-file configs/pigseg/semseg-ptv3-v1m1-0-base.py --options resume=True weight=exp/PTV3_PigSeg/model/model_last.pth


## 基于PointTransformerV3的体尺体重回归模型训练
python tools/train.py --config-file configs/my_dataset/ptv3_weight.py
"""
if __name__ == "__main__":
    main()
