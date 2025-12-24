_base_ = ["../_base_/default_runtime.py"]

# ==============================================================================
# Global Settings
# ==============================================================================
epoch = 100 
save_path = "exp/keypoint_sparse_unet"

# [重要] 避免 BN 在 batch=1 时崩溃，同时也兼容 Evaluator 逻辑
batch_size = 12
batch_size_val = 8 

# ==============================================================================
# Model Settings
# ==============================================================================
model = dict(
    type="KeypointSparseUNet",
    num_keypoints=6,
    
    # === SpUNet Backbone 参数 ===
    # 输入维度: xyz(3) 被移除，剩下 normal(3) + curvature(1) = 4
    in_channels=4, 
    num_classes=0, # 不使用内置分类头
    
    # 经典 SpUNet 配置
    base_channels=32,
    channels=(32, 64, 128, 256, 256, 128, 96, 96),
    layers=(2, 3, 4, 6, 2, 2, 2, 2),
    enc_mode=False, # False = 使用完整 U-Net 结构 (Decoder 输出高分辨特征后池化)
                    # True = 仅使用 Encoder (类似 ResNet，直接池化低分辨特征)
                    # 对于回归任务，您可以尝试改为 True 看看是否收敛更快，但 False 通常细节更好
    
    hidden_dim=256, # 回归头隐藏层
)

# ==============================================================================
# Data Settings
# ==============================================================================
num_worker = 4
data_root = "/home/gzh/point/DataSets"
grid_size_val = 0.02

data = dict(
    train=dict(
        type="KeypointDataset",
        split="train",
        data_root=data_root,
        transform=[
            dict(type="Update", keys_dict=dict(index_valid_keys=["coord", "feat", "coord_feat"], grid_size=grid_size_val)),
            dict(type="GridSample", grid_size=grid_size_val, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="ToTensor"),
            dict(type="Collect", 
                 keys=("coord", "grid_coord", "feat", "target", "grid_size", "scale"), 
                 offset_keys_dict=dict(offset="coord"), 
                 feat_keys=("feat",)) 
        ],
        loop=1,
    ),
    val=dict(
        type="KeypointDataset",
        split="val",
        data_root=data_root,
        transform=[
            dict(type="Update", keys_dict=dict(index_valid_keys=["coord", "feat", "coord_feat"], grid_size=grid_size_val)),
            dict(type="GridSample", grid_size=grid_size_val, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="ToTensor"),
            dict(type="Collect", 
                 keys=("coord", "grid_coord", "feat", "target", "grid_size", "scale"), 
                 offset_keys_dict=dict(offset="coord"), 
                 feat_keys=("feat",))
        ],
    ),
    test=dict(
        type="KeypointDataset",
        split="test",
        data_root=data_root,
        transform=[
            dict(type="Update", keys_dict=dict(index_valid_keys=["coord", "feat", "coord_feat"], grid_size=grid_size_val)),
            dict(type="GridSample", grid_size=grid_size_val, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="ToTensor"),
            dict(type="Collect", 
                 keys=("coord", "grid_coord", "feat", "target", "grid_size", "scale"), 
                 offset_keys_dict=dict(offset="coord"), 
                 feat_keys=("feat",))
        ],
    ),
)

# ==============================================================================
# Training Settings
# ==============================================================================
# SGD 通常在 SpUNet 上表现比 AdamW 更稳健，但 AdamW 收敛快。
# 这里沿用您之前成功的 AdamW 配置。
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.02)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=10),
    dict(type="InformationWriter"),
    dict(type="KeypointEvaluator"),
    dict(type="CheckpointSaver", save_freq=20)
]
