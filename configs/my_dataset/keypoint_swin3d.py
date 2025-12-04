_base_ = ["../_base_/default_runtime.py"]

# ==============================================================================
# Global Settings
# ==============================================================================
epoch = 100 
save_path = "exp/keypoint_swin3d"
# ==============================================================================
# Model Settings (Swin3D)
# ==============================================================================
model = dict(
    type="KeypointSwin3D",
    num_keypoints=6,
    backbone_conf=dict(
        type="Swin3D-v1m1",
        in_channels=4, # 或 4，取决于你的 coord_feat 逻辑
        num_classes=64,
        
        base_grid_size=0.02,
        quant_size=50,       # 必须是整数
        num_layers=4,        # 必须显式指定为 4
        
        depths=[2, 2, 6, 2],
        channels=[64, 128, 256, 512],
        
        # [核心修改] 必须全为偶数！
        # 原来是 [3, 6, 12, 24] -> 改为 [6, 12, 24, 48] 或者 [4, 8, 16, 32]
        num_heads=[4, 8, 16, 32], 
        
        window_sizes=[5, 7, 7, 7],
        up_k=3,
        drop_path_rate=0.2,
        stem_transformer=True,
        down_stride=2,
        upsample="linear",
        knn_down=True,
        cRSE="XYZ_RGB",
        fp16_mode=1, 
    ),

    hidden_dim=256,
)

# ==============================================================================
# Data Settings
# ==============================================================================
num_worker = 4
batch_size = 8
data_root = "/home/gzh/point/DataSets"
grid_size_val = 0.02 # 这里的 grid_size 必须与模型的 quant_size/base_grid_size 匹配

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
                 keys=("coord", "grid_coord", "feat", "target","coord_feat", "grid_size", "scale"), 
                 offset_keys_dict=dict(offset="coord"), 
                 feat_keys=("feat",),
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
