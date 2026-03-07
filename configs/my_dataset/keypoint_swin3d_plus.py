_base_ = ["../_base_/default_runtime.py"]

# ==============================================================================
# Global Settings
# ==============================================================================
epoch = 100 
num_worker = 8
batch_size = 8  
save_path = "exp/keypoint_swin3d_plus"

# ==============================================================================
# Model Settings (模型部分：保持 Vote 机制)
# ==============================================================================
model = dict(
    type="KeypointSwin3DVote",
    num_keypoints=6,
    hidden_dim=256,
    vote_radius=0.3, # 投票半径

    backbone_conf=dict(
        type="Swin3D-v1m1",
        
        # [输入通道] 
        # 注意：这里对应 coord_feat 的维度，如果 coord_feat 是 feat 的复制，则为 4
        in_channels=4, 
        
        num_classes=64, # 对应 channels[0]
        num_layers=4,   # [关键] 显式指定层数，防止 IndexError
        
        base_grid_size=0.02,
        quant_size=50, # 保持和你原始能跑的配置一致
        
        depths=[2, 2, 6, 2],
        
        # [关键] 恢复为 4 个通道
        channels=[64, 128, 256, 512],
        
        # [关键] num_heads 必须是偶数，且能被 channels 整除
        num_heads=[4, 8, 16, 32], 
        
        window_sizes=[5, 7, 7, 7],
        up_k=3,
        drop_path_rate=0.2,
        stem_transformer=True,
        down_stride=2,
        upsample="linear",
        knn_down=True,
        cRSE="XYZ_RGB", # Swin3D 特有的相对位置编码设置
        fp16_mode=1,
    ),
)

# ==============================================================================
# Data Settings (数据部分：完全恢复为你提供的“可运行版本”)
# ==============================================================================
data_root = "/home/gzh/point/DataSets"
grid_size_val = 0.02 # 这里的 grid_size 必须与模型的 quant_size/base_grid_size 匹配
# 训练时的 grid_size，如果不一致可以单独设，这里设为一致
grid_size_train = 0.02 

data = dict(
    num_workers=num_worker,
    batch_size=batch_size,
    sample_rate=16, # 保持采样率
    train=dict(
        type="KeypointDataset",
        split="train",
        data_root=data_root,
        transform=[
            # [关键] 恢复 coord_feat
            dict(type="Update", keys_dict=dict(index_valid_keys=["coord", "feat", "coord_feat"], grid_size=grid_size_train)),
            dict(type="GridSample", grid_size=grid_size_train, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="ToTensor"),
            dict(type="Collect", 
                 # [关键] keys 中必须包含 "coord_feat"
                 keys=("coord", "grid_coord", "feat", "target", "coord_feat", "grid_size", "scale"), 
                 offset_keys_dict=dict(offset="coord"), 
                 feat_keys=("feat",),
                 # [关键] 显式指定 coord_feat 的来源
                 coord_feat_keys=("coord_feat",))
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
                 keys=("coord", "grid_coord", "feat", "target", "coord_feat", "grid_size", "scale"), 
                 offset_keys_dict=dict(offset="coord"), 
                 feat_keys=("feat",),
                 coord_feat_keys=("coord_feat",))
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
                 keys=("coord", "grid_coord", "feat", "target", "coord_feat", "grid_size", "scale"), 
                 offset_keys_dict=dict(offset="coord"), 
                 feat_keys=("feat",),
                 coord_feat_keys=("coord_feat",))
        ],
    ),
)

# ==============================================================================
# Training Settings
# ==============================================================================
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.05)
scheduler = dict(type="CosineAnnealingLR", eta_min=1e-5) 

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=10),
    dict(type="InformationWriter"),
    dict(type="KeypointEvaluator"),
    dict(type="CheckpointSaver", save_freq=20)
]