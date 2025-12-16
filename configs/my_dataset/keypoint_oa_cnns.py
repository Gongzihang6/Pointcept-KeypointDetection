_base_ = ["../_base_/default_runtime.py"]

# ==============================================================================
# Global Settings
# ==============================================================================
epoch = 100 
save_path = "exp/keypoint_oa_cnns" # 修改保存路径

# ==============================================================================
# Model Settings (OACNNs)
# ==============================================================================
model = dict(
    type="KeypointOACNNs",
    num_keypoints=6,
    
    # === OA-CNNs Backbone 参数 (参考自官方 Config) ===
    in_channels=4,  # 输入通道数 (coord + feat), 取决于 transform 的 feat_keys
    embed_channels=64,
    enc_channels=[64, 64, 128, 256],
    groups=[4, 4, 8, 16],
    enc_depth=[3, 3, 9, 8],
    dec_channels=[256, 256, 256, 256],
    # 注意：point_grid_size 必须与数据中的 grid_size 配合
    point_grid_size=[[8, 12, 16, 16], [6, 9, 12, 12], [4, 6, 8, 8], [3, 4, 6, 6]],
    dec_depth=[2, 2, 2, 2],
    enc_num_ref=[16, 16, 16, 16],
    
    hidden_dim=256, # 回归头隐藏层维度
)

# ==============================================================================
# Data Settings
# ==============================================================================
num_worker = 4
batch_size = 8 # OACNNs 比较轻量，可以尝试比 Swin3D 更大的 Batch Size
batch_size_val = 8
batch_size_test = 8
data_root = "/home/gzh/point/DataSets"
grid_size_val = 0.02 # Voxelization grid size

data = dict(
    train=dict(
        type="KeypointDataset",
        split="train",
        data_root=data_root,
        transform=[
            # OACNNs 需要 grid_coord，所以 GridSample 是必须的
            dict(type="Update", keys_dict=dict(index_valid_keys=["coord", "feat", "coord_feat"], grid_size=grid_size_val)),
            dict(type="GridSample", grid_size=grid_size_val, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="ToTensor"),
            dict(type="Collect", 
                 keys=("coord", "grid_coord", "feat", "target", "grid_size", "scale"), 
                 offset_keys_dict=dict(offset="coord"), 
                 # 注意：这里我们收集 'feat' 作为特征输入。
                 # 如果你的特征是 Intensity，记得在 KeypointDataset 中处理好。
                 # 这里的 feat_keys=('feat',) 意味着只取 feat 字段
                 feat_keys=("feat",)) 
        ],
        loop=1,
    ),
    val=dict(
        type="KeypointDataset",
        # batch_size=8,  # 修改这里，不要用默认的 1
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
# OACNNs 论文中常用 SGD 或 AdamW
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
