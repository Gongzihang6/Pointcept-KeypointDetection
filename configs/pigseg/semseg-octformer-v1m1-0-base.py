"""
===============================================================================
代码作用：
定义 OctFormer 模型在 Pig 二分类数据集上的语义分割训练配置。

功能/实现了什么：
1. 指定 OctFormer 作为语义分割 backbone。
2. 显式收集 offset、normal 和 curvature，满足 OctFormer 前向所需字段。
3. 复用与 PTv3 pigseg 相同的 FocalLoss + LovaszLoss 配方和权重比。

怎么实现的：
通过 Pointcept 的配置系统，将模型、数据流水线、优化器和学习率调度器全部声明在一个 Python 配置文件中，由训练入口动态实例化。
===============================================================================
"""

_base_ = ["../_base_/default_runtime.py"]

# === 核心参数配置 ===
weight = None
resume = False
evaluate = True
test_only = False

num_classes = 2
in_channels = 4
voxel_size = 30

save_path = "exp/OctFormer_PigSeg_0513"

# === 模型配置 ===
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="OctFormer-v1m1",
        in_channels=in_channels,
        num_classes=num_classes,
        fpn_channels=168,
        channels=(96, 192, 384, 384),
        num_blocks=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 24),
        patch_size=26,
        stem_down=2,
        head_up=2,
        dilation=4,
        drop_path=0.5,
        nempty=True,
        octree_depth=11,
        octree_full_depth=2,
    ),
    criteria=[
        dict(type="FocalLoss", gamma=2.0, alpha=[0.1, 0.9], loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=3.0, ignore_index=-1),
    ],
)

# === 数据根目录 ===
data_root = "/autodl-fs/data/body_npy_output"

train_pipeline = [
    dict(type="CenterShift", apply_z=True),
    # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    # dict(type="RandomScale", scale=[0.9, 1.1]),
    # dict(type="RandomFlip", p=0.5),
    # dict(type="RandomJitter", sigma=0.005, clip=0.02),
    dict(
        type="GridSample",
        grid_size=voxel_size,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
    ),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "normal", "segment"),
        feat_keys=("normal", "curvature"),
        offset_keys_dict=dict(offset="coord"),
    ),
]

test_pipeline = [
    dict(type="CenterShift", apply_z=True),
]

# val pipeline should use GridSample in 'train' mode so transforms return a dict
val_pipeline = [
    dict(type="CenterShift", apply_z=True),
    dict(
        type="GridSample",
        grid_size=voxel_size,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
    ),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "normal", "segment"),
        feat_keys=("normal", "curvature"),
        offset_keys_dict=dict(offset="coord"),
    ),
]

data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=["Background", "Pig"],
    train=dict(
        type="PigDataset",
        split="train",
        data_root=data_root,
        transform=train_pipeline,
        test_mode=False,
        loop=1,
    ),
    val=dict(
        type="PigDataset",
        split="val",
        data_root=data_root,
        transform=val_pipeline,
        test_mode=False,
    ),
    test=dict(
        type="PigDataset",
        split="val",
        data_root=data_root,
        transform=test_pipeline,
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=voxel_size,
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=True),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "normal", "index"),
                    feat_keys=("normal", "curvature"),
                    offset_keys_dict=dict(offset="coord"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
            ],
        ),
    ),
)

# === 优化器与调度器 ===
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=0.0002,
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

dataset_type = "PigDataset"
data_loader = dict(
    type=dataset_type,
    dataloader=dict(
        train=dict(batch_size=1, num_workers=4, shuffle=True),
        val=dict(batch_size=1, num_workers=2, shuffle=False),
        test=dict(batch_size=1, num_workers=2, shuffle=False),
    ),
)

epoch = 100
eval_epoch = 10