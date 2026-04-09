"""
===============================================================================
代码作用：
定义 SpUNet (Sparse UNet) 模型在 Pig (猪主体) 二分类数据集上的语义分割训练完整配置。

功能/实现了什么：
1. 替换了重量级的 Swin3D 模型，采用了极其轻量且高效的纯稀疏卷积网络 SpUNet，大幅降低显存占用并提升训练速度。
2. 修复了数据预处理流水线中的版本兼容性问题（移除了 GridSample 中过时的 keys 参数）。
3. 完善了特征的收集机制，将 3 维法向和 1 维曲率合并作为 4 维几何特征 (coord_feat_keys) 送入网络。
4. 配置了基于体素下采样 (2cm网格) 的数据增强与加载策略。
5. 修复了优化器单参数组与 OneCycleLR 学习率调度器的维度匹配问题。

怎么实现的：
基于 Pointcept 的 Registry 注册机制，将模型 (SpUNet)、数据流水线 (含中心平移、随机旋转、缩放、网格采样等)、数据集加载 (PigDataset)、以及优化器/调度器参数封装为一个完整的 Python 字典结构。框架启动时会读取此文件并动态实例化所有模块。
===============================================================================
"""

_base_ = ["../_base_/default_runtime.py"]

# === 核心基础参数 ===
weight = None  # 预训练权重路径
resume = False
evaluate = True
test_only = False

num_classes = 2 # 二分类：0背景，1猪
in_channels = 4 # 输入特征维度：nx, ny, nz, curvature
voxel_size = 20 # 体素化下采样尺寸（0.02表示2厘米，可根据显存和速度进一步调大到 0.03）

# === 模型配置 (SpUNet) ===
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=in_channels,
        num_classes=num_classes,
        # SpUNet 各层的通道数和块数，如果你的显存还是吃紧，可以把 channels 里的数字全部除以 2
        channels=(32, 64, 128, 256, 256, 128, 96, 96), 
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)
    ]
)

# === 数据根目录 ===
data_root = "/home/gzh/point/Pointcept-KeypointDetection/body_npy_output"

# === 训练集数据流水线 ===
train_pipeline = [
    # 1. 先做空间数据增强（旋转、加噪等，此时坐标出现负数没关系）
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=5.0, clip=20.0), # 5毫米高斯噪声
    
    # 2. 【关键修复】必须在这里平移至正数象限！强制把所有坐标全拉到非负数区间！
    dict(type="CenterShift", apply_z=True),
    
    # 3. 安全体素化下采样
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
    ),
]


# === 验证/测试集数据流水线 ===
test_pipeline = [
    dict(type="CenterShift", apply_z=True),
    dict(
        type="GridSample",
        grid_size=voxel_size,
        hash_type="fnv",
        mode="test",
        return_grid_coord=True, # 注意：删除了报错的 keys 参数
    ),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment"),
        feat_keys=("normal", "curvature"),
        coord_feat_keys=("normal", "curvature"),
    ),
]

# === 数据集配置 ===
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
        loop=1, # 如果训练集很小，可以增加 loop (如 10) 来延长每个 epoch 的时间
    ),
    val=dict(
        type="PigDataset",
        split="val",
        data_root=data_root,
        transform=test_pipeline,
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
                ),
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
            ],
        ),
    ),
)

# === 优化器与学习率调度器 ===
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.05)

# 注意：修复了 max_lr 数组长度与 param_groups 不匹配的问题
scheduler = dict(
    type="OneCycleLR",
    max_lr=0.002,            # 改为了单个数值
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

# === 数据加载器配置 ===
dataset_type = "PigDataset"
data_loader = dict(
    type=dataset_type,
    dataloader=dict(
        # 降了 batch_size，防止爆显存和卡顿
        train=dict(batch_size=4, num_workers=4, shuffle=True), 
        val=dict(batch_size=1, num_workers=2, shuffle=False),
        test=dict(batch_size=1, num_workers=2, shuffle=False),
    ),
)

# === 训练 Epoch 配置 ===
epoch = 100
eval_epoch = 10 # 每 10 个 epoch 在验证集上做一次测试并保存权重
