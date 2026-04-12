"""
PTv3 猪体尺与体重回归 配置文件
"""

_base_ = ["../_base_/default_runtime.py"]

# ================= 1. 基础环境与训练超参数 =================
weight = None
resume = False
evaluate = True
test_only = False
seed = 42

# 训练 Epoch 与 Batch Size 设置 (根据您的显存调整)
empty_cache = False
empty_cache_per_epoch = False
epochs = 200
eval_epoch = 10  # 每 10 个 epoch 验证一次
save_path = "exp/weight_ptv3_0411"


# AdamW 优化器
optimizer = dict(
    type="AdamW", 
    lr=0.001, 
    weight_decay=0.05
)

# 学习率调度器 (OneCycle 策略通常收敛极快)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# ================= 2. 模型结构定义 =================
model = dict(
    type="PigBodyRegressor",  # 这是我们在 pointcept/models/__init__.py 中注册的自定义模型类
    # 使用上文自定义的 L1 Regression Loss 
    criteria=[dict(type="RegressionL1Loss", loss_weight=1.0)],
    # 7个回归预测目标
    num_classes=7, 
    backbone_embed_dim=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=4,          # 对应我们的特征 [nx, ny, nz, c]
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),    # 4层下采样
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(1, 2, 4, 8, 16),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
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
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        # 重点：开启 cls_mode，使 PTv3 进行全局特征池化 (Global Pooling)，而非点级别的输出
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("nuisance", 3),
    )
)

# ================= 3. 数据集与增强流水线 =================
dataset_type = "PigWeightDataset"  # 这是我们在 pointcept/models/__init__.py 中注册的自定义数据集类 
# 请确保此路径与您刚才生成的 .npy 文件夹路径一致
data_root = "Processed_Pig_Dataset" 

# 体素化网格大小：0.02 表示 2 厘米左右合并为一个体素
# 请根据您的点云密度和坐标系单位 (m 或 cm) 进行调整。
grid_size = 20

# 训练集增强：禁止 RandomScale！
train_transform = [
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        # 把 normal 和 color 都收集起来
        keys=("coord", "grid_coord", "normal", "color", "category"), 
        # 把 normal(3维) 和 color(1维) 拼接作为模型的总输入特征
        feat_keys=["normal", "color"], 
    ),
]

test_transform = [
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "normal", "color", "category"),
        feat_keys=["normal", "color"],
    ),
]


data = dict(
    num_classes=7,
    ignore_index=-1,
    names=["length", "width", "height", "chest", "waist", "hip", "weight"],
    
    # DataLoader 配置
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=train_transform,
        test_mode=False,
        loop=1, # 如果训练集小，可以增加 loop 倍数以减少 dataloader 重新加载开销
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=test_transform,
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=test_transform,
        test_mode=True,
    ),
)
