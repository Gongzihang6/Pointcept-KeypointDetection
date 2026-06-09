_base_ = ["../_base_/default_runtime.py"]

# ==============================================================================
# Global Settings
# ==============================================================================
epoch = 100 
save_path = "exp/offset_keypoint_swin3d_0512"
# ==============================================================================
# Model Settings (Swin3D for Offset)
# ==============================================================================
model = dict(
    type="OffsetKeypointSwin3D",
    num_keypoints=6,
    backbone_conf=dict(
        type="Swin3D-v1m1",
        in_channels=4,    # 或 4，取决于 coord_feat 逻辑
        num_classes=64,   # 随意，在 backbone_conf 里占位
        
        base_grid_size=0.02,
        quant_size=50,       # 必须是整数
        num_layers=4,        # 必须显式指定为 4
        
        depths=[2, 2, 6, 2],
        channels=[64, 128, 256, 512],
        
        # Swin3D 中必须全为偶数
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
data_root = "KeyPointDataset_Split"
grid_size_val = 0.02 # 这里的 grid_size 必须与模型的 quant_size/base_grid_size 匹配
offset_radius = 300.0
online_offset = True

data = dict(
    train=dict(
        type="OffsetKeypointDataset",
        split="train",
        data_root=data_root,
        offset_radius=offset_radius,
        online_offset=online_offset,
        transform=[
            dict(type="Update", keys_dict=dict(index_valid_keys=["coord", "feat", "target"], grid_size=grid_size_val)),
            # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomFlip", p=0.5),
            # dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="GridSample", grid_size=grid_size_val, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(type="Collect", 
                 keys=("coord", "grid_coord", "feat", "target", "grid_size", "scale"), 
                 offset_keys_dict=dict(offset="coord"), 
                 feat_keys=("feat",))
        ],
        loop=1,
    ),
    val=dict(
        type="OffsetKeypointDataset",
        split="val",
        data_root=data_root,
        offset_radius=offset_radius,
        online_offset=online_offset,
        transform=[
            dict(type="Update", keys_dict=dict(index_valid_keys=["coord", "feat", "target"], grid_size=grid_size_val)),
            dict(type="GridSample", grid_size=grid_size_val, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="ToTensor"),
            dict(type="Collect", 
                 keys=("coord", "grid_coord", "feat", "target", "grid_size", "scale"), 
                 offset_keys_dict=dict(offset="coord"), 
                 feat_keys=("feat",))
        ],
    ),
    test=dict(
        type="OffsetKeypointDataset",
        split="test",
        data_root=data_root,
        offset_radius=offset_radius,
        online_offset=online_offset,
        transform=[
            dict(type="Update", keys_dict=dict(index_valid_keys=["coord", "feat", "target"], grid_size=grid_size_val)),
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
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.05)
scheduler = dict(type="CosineAnnealingLR", eta_min=1e-5) 

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=10),
    dict(type="InformationWriter"),
    dict(type="OffsetKeypointEvaluator"),
    dict(type="CheckpointSaver", save_freq=20)
]
