_base_ = ["../_base_/default_runtime.py"]

# ==============================================================================
# Global Settings
# ==============================================================================
epoch = 100 
save_path = "exp/keypoint_ptv2"
# ==============================================================================
# Model Settings (修正版 PT-v2m2)
# ==============================================================================
model = dict(
    type="KeypointPTv2",
    num_keypoints=6,
    backbone_conf=dict(
        type="PT-v2m2",  # 确保与模型文件注册名一致
        in_channels=4,   # 你的数据特征维度 (Feature)
        num_classes=0,   # 设为0以跳过分割头
        
        # === PTv2 专用参数 ===
        patch_embed_depth=1,
        patch_embed_channels=48,
        patch_embed_groups=6,
        patch_embed_neighbours=8,
        
        # Encoder 设置
        enc_depths=(2, 2, 6, 2),
        enc_channels=(96, 192, 384, 512),
        enc_groups=(12, 24, 48, 64),      # [修正] 这里是 groups，不是 num_head
        enc_neighbours=(16, 16, 16, 16),
        
        # Decoder 设置
        dec_depths=(1, 1, 1, 1),
        dec_channels=(48, 96, 192, 384),
        dec_groups=(6, 12, 24, 48),       # [修正] 这里是 groups
        dec_neighbours=(16, 16, 16, 16),
        
        # 网格大小设置 (PTv2 核心参数)
        # 假设你的 base grid_size 是 0.02 (在 Data Settings 中设置)
        # PTv2 每一层通常倍增: x3, x6, x12, x24 (相对于 base grid)
        # 或者直接写死物理尺寸:
        grid_sizes=(0.06, 0.12, 0.24, 0.48), 
        
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        enable_checkpoint=False,
        unpool_backend="map",
    ),
    # PTv2 Decoder 最后一层输出通道是 dec_channels[0] (这里是 48)
    # 所以模型代码里自动获取 in_channels 应该是 48
    hidden_dim=256,
)

# ==============================================================================
# Data Settings (保持不变)
# ==============================================================================
num_worker = 4
batch_size = 8
data_root = "/home/gzh/point/DataSets"
grid_size_val = 0.02 # 这个值会影响数据加载时的体素化

data = dict(
    train=dict(
        type="KeypointDataset",
        split="train",
        data_root=data_root,
        transform=[
            dict(type="Update", keys_dict=dict(index_valid_keys=["coord", "feat"], grid_size=grid_size_val)),
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
            dict(type="Update", keys_dict=dict(index_valid_keys=["coord", "feat"], grid_size=grid_size_val)),
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
            dict(type="Update", keys_dict=dict(index_valid_keys=["coord", "feat"], grid_size=grid_size_val)),
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
    dict(type="KeypointEvaluator"),
    dict(type="CheckpointSaver", save_freq=20)
]