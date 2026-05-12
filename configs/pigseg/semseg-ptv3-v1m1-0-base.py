"""
===============================================================================
代码作用：
配置基于 PointTransformerV3 (PTv3) 的猪主体二分类语义分割训练文件。

功能：
定义了使用 PTv3 模型进行点云语义分割的全部超参数，包括前沿的网络结构、数据预处理增强流水线、优化器和学习率调度器。

实现了什么：
1. 引入了目前极具统治力的 PTv3 骨干网络，利用 FlashAttention 和空间序列化技术处理点云，极大提升了模型感受野和训练速度。
2. 沿用了之前验证绝对安全的数据预处理流水线（中心化平移防负数、20mm (0.02m) 体素化下采样防内存溢出）。
3. 动态特征收集：将 3 维法向和 1 维曲率自动合并为 4 维特征张量送入 Transformer 编码器。
4. 配置了针对 PTv3 优化的学习率 (0.006) 和权重衰减 (0.05)。

怎么实现的：
利用 Pointcept 的配置解析机制，指定 `model.backbone.type="PT-v3m1"`。通过定义 `enc_depths`、`enc_channels`、`enc_num_head` 等参数精细控制 Transformer 各层块的深度与注意力头数。复用现有的 `PigDataset` 进行数据加载，并将流水线的 `mode` 正确区分为 `train` 和 `test` 避免评估报错。
===============================================================================
"""

_base_ = ["../_base_/default_runtime.py"]

# === 核心基础参数 ===
weight = None
resume = False
evaluate = True
test_only = False

num_classes = 2 # 0:背景, 1:猪
in_channels = 4 # nx, ny, nz, curvature
voxel_size = 40 # 20毫米 (0.02米) 体素化下采样
save_path = "exp/PTV3_PigSeg"
# === 模型配置 (PTv3) ===
# === 模型配置 (PTv3) ===
model = dict(
    type="DefaultSegmentorV2",         # <--- 【关键1】换成专门接盘 PTv3 的 V2 版本！
    num_classes=num_classes,           # <--- 【关键2】类别数放在 V2 分割器这里！
    backbone_out_channels=64,          # <--- 【关键3】PTv3 解码器最后输出的特征维度是 64
    backbone=dict(
        type="PT-v3m1",
        in_channels=in_channels,
        # 注意：这里里面千万别写 num_classes 了！
        order=["z", "z-trans", "hilbert", "hilbert-trans"], 
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256), # 解码器最后输出维度是这里的第一项 64
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False, # <--- 保持你之前改的 False 防止爆显存
        upcast_attention=False,
        upcast_softmax=False,
    ),
    criteria=[
        # dict(type="CrossEntropyLoss", loss_weight=1.0, weight=[1.0, 10.0], ignore_index=-1),
        dict(type="FocalLoss", gamma=2.0, alpha=[0.1, 0.9], loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=3.0, ignore_index=-1)
    ]
)

# === 数据根目录 ===
data_root = "/home/gzh/point/Pointcept-KeypointDetection/body_npy_output"

# === 快速验证流水线 (训练时的 val 使用) ===
val_pipeline = [
    dict(type="CenterShift", apply_z=True),
    dict(
        type="GridSample",
        grid_size=voxel_size,
        hash_type="fnv",
        mode="train", # 训练期验证保持 train 模式
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

# === 训练集流水线 （关闭所有数据增强） ===
train_pipeline = [
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=5.0, clip=20.0),
    # 增强后强制坐标平移归非负
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
    ),
]

# === 精确测试流水线 ===
test_pipeline = [
    dict(type="CenterShift", apply_z=True),
]

# === 数据集挂载 ===
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
        transform=val_pipeline, # 使用修改好的 val_pipeline
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

# === 优化器与调度器 ===
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=0.006,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# === Dataloader ===
dataset_type = "PigDataset"
data_loader = dict(
    type=dataset_type,
    dataloader=dict(
        train=dict(batch_size=8, num_workers=4, shuffle=True),
        val=dict(batch_size=1, num_workers=2, shuffle=False),
        test=dict(batch_size=1, num_workers=2, shuffle=False),
    ),
)

epoch = 100
eval_epoch = 10
