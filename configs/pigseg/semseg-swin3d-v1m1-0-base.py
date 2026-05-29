"""
===============================================================================
代码作用：
定义 Swin3D 模型在 Pig 二分类数据集上的语义分割训练配置。

功能/实现了什么：
1. 指定了模型结构（Swin3D），并设置类别数为 2（0和1）。
2. 配置了输入特征维度为 4（nx, ny, nz, curvature）。
3. 定义了数据加载路径、Voxelization（体素化下采样）的网格大小。
4. 定义了优化器（AdamW）和学习率调度器。

怎么实现的：
利用 Pointcept 的配置系统，将模型参数 (model)、数据流水线 (data)、优化器 (optimizer) 和训练循环 (hooks) 以字典和类的形式声明，框架在运行时会解析该文件实例化相应模块。
===============================================================================
"""

_base_ = ["../_base_/default_runtime.py"]

# === 核心参数配置 ===
weight = None  # 预训练权重路径，没有就设为 None
resume = False
evaluate = True
test_only = False

num_classes = 2 # 二分类
in_channels = 4 # nx, ny, nz, curvature
# voxel_size 决定了下采样的程度，根据你的猪体型（米为单位），0.01 表示 1cm
voxel_size = 30 

save_path = "exp/Swin3D_PigSeg_0512"
# === 模型配置 ===
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="Swin3D-v1m1",
        in_channels=in_channels,
        num_classes=num_classes,
        base_grid_size=30,
        depths=[2, 4, 9, 4, 4],
        channels=[48, 96, 192, 384, 384],
        num_heads=[6, 6, 12, 24, 24],
        window_sizes=[5, 7, 7, 7, 7],
        quant_size=4,
        drop_path_rate=0.2,
        up_k=3,
        num_layers=5,
        stem_transformer=True,
        down_stride=3,
        upsample="linear_interpolate",
        knn_down=True,
        cRSE="XYZ", # 虽然叫RGB，但实际处理的是连续特征
        fp16_mode=1,
    ),
    criteria=[
        # dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1,weight=[0.1, 0.9]),
        dict(type="FocalLoss", gamma=2.0, alpha=[0.1, 0.9], loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=3.0, ignore_index=-1)
    ]
)

# === 数据流水线与预处理 ===
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
        keys=("coord", "grid_coord", "segment"),
        feat_keys=("normal", "curvature"),
        coord_feat_keys=("normal", "curvature"),
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
        keys=("coord", "grid_coord", "segment"),
        feat_keys=("normal", "curvature"),
        coord_feat_keys=("normal", "curvature"),
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
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("normal", "curvature"),
                    coord_feat_keys=("normal", "curvature"),
                    offset_keys_dict=dict(offset="coord"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
            ],
        ),
    ),
)

# === 训练参数 ===
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=0.0002,
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

# batch_size 取决于你的显存大小和体素化后的点数
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
