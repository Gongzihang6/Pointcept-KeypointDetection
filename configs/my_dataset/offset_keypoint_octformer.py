_base_ = ["../_base_/default_runtime.py"]

# ======================================================================
# Global Settings
# ======================================================================
epoch = 100
num_worker = 4
batch_size = 8
save_path = "exp/offset_keypoint_octformer_0512"

# ======================================================================
# Model Settings
# ======================================================================
model = dict(
    type="OffsetKeypointOctFormer",
    in_channels=4,
    num_keypoints=6,
    hidden_dim=256,
    fpn_channels=168,
    channels=(96, 192, 384, 384),
    num_blocks=(2, 2, 18, 2),
    num_heads=(6, 12, 24, 24),
    patch_size=26,
    stem_down=2,
    head_up=2,
    dilation=4,
    drop_path=0.5,
    nempty=True,
    octree_depth=11,
    octree_full_depth=2,
    octree_scale_factor=10.24,
)

# ======================================================================
# Data Settings
# ======================================================================
data_root = "KeyPointDataset_Split"
grid_size_val = 0.02

data = dict(
    train=dict(
        type="OffsetKeypointDataset",
        split="train",
        data_root=data_root,
        transform=[
            dict(type="Update", keys_dict=dict(index_valid_keys=["coord", "feat", "target"], grid_size=grid_size_val)),
            dict(
                type="GridSample",
                grid_size=grid_size_val,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "feat", "target", "grid_size", "scale"),
                offset_keys_dict=dict(offset="coord"),
                feat_keys=("feat",),
            ),
        ],
        loop=1,
    ),
    val=dict(
        type="OffsetKeypointDataset",
        split="val",
        data_root=data_root,
        transform=[
            dict(type="Update", keys_dict=dict(index_valid_keys=["coord", "feat", "target"], grid_size=grid_size_val)),
            dict(
                type="GridSample",
                grid_size=grid_size_val,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "feat", "target", "grid_size", "scale"),
                offset_keys_dict=dict(offset="coord"),
                feat_keys=("feat",),
            ),
        ],
    ),
    test=dict(
        type="OffsetKeypointDataset",
        split="test",
        data_root=data_root,
        transform=[
            dict(type="Update", keys_dict=dict(index_valid_keys=["coord", "feat", "target"], grid_size=grid_size_val)),
            dict(
                type="GridSample",
                grid_size=grid_size_val,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "feat", "target", "grid_size", "scale"),
                offset_keys_dict=dict(offset="coord"),
                feat_keys=("feat",),
            ),
        ],
    ),
)

# ======================================================================
# Training Settings
# ======================================================================
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.05)
scheduler = dict(type="CosineAnnealingLR", eta_min=1e-5)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=10),
    dict(type="InformationWriter"),
    dict(type="OffsetKeypointEvaluator"),
    dict(type="CheckpointSaver", save_freq=20),
]