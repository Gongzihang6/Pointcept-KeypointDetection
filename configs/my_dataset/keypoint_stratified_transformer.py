_base_ = ["../_base_/default_runtime.py"]

# ==============================================================================
# Global Settings
# ==============================================================================
epoch = 100 
save_path = "exp/keypoint_stratified_transformer"

# ==============================================================================
# Model Settings
# ==============================================================================
model = dict(
    type="KeypointStratifiedTransformer",
    num_keypoints=6,
    
    # === Stratified Transformer Backbone 参数 ===
    # 这里的 in_channels 取决于输入特征维度。
    # 如果你的数据只有 xyz + intensity (共4维)，这里填 4 (ST 内部通常处理 coord+feat)
    # 如果代码中 split 了 coord，这里可能只需要特征维度 (1)。
    # 根据 ST 源码：feats = data_dict["feat"]，如果 feat 只有 intensity，in_channels 应为 1。
    # 假设你的 feat 是 intensity，维度为 1。
    in_channels=4, 
    
    # 下面是 ST-v1m2 的标准参数 (参考 semseg-st-v1m2-0-refined.py)
    channels=[48, 96, 192, 384, 384],
    num_heads=[6, 12, 24, 24],
    depths=[3, 9, 3, 3],
    window_size=[0.2, 0.4, 0.8, 1.6], # 窗口大小，单位米
    quant_size=[0.01, 0.02, 0.04, 0.08], # 量化网格大小，单位米
    mlp_expend_ratio=4.0,
    down_ratio=0.25,
    down_num_sample=16,
    kp_ball_radius=0.05, # 2.5 * 0.02
    kp_max_neighbor=34,
    kp_grid_size=0.02,
    kp_sigma=1.0,
    drop_path_rate=0.2,
    rel_query=True,
    rel_key=True,
    rel_value=True,
    qkv_bias=True,
    stem=True,
    
    hidden_dim=256, # 回归头隐藏层
)

# ==============================================================================
# Data Settings
# ==============================================================================
num_worker = 4
batch_size = 8 # ST 显存占用适中，可以尝试 8-12
data_root = "/home/gzh/point/DataSets"

# ST 对网格大小比较敏感，确保这里的 grid_size 与模型参数匹配
grid_size_val = 0.02 

data = dict(
    train=dict(
        type="KeypointDataset",
        split="train",
        data_root=data_root,
        transform=[
            # [Fix] 移除 key 中的 "offset"
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
