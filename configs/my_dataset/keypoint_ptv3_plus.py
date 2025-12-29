_base_ = ["../_base_/default_runtime.py"]

# ==============================================================================
# Global Settings
# ==============================================================================
epoch = 100
save_path = "exp/keypoint_ptv3_plus"
# ==============================================================================
# Model Settings
# ==============================================================================
model = dict(
    type="KeypointPTv3Plus",  # 使用注册的新模型类 KeypointPTv3Plus
    num_keypoints=6,
    backbone_conf=dict(
        type="PT-v3m1-Plus",  # 指定 Backbone 为我们新写的 Plus 版本
        in_channels=4,        # 7-3=4 (法向量+曲率)
        
        # [核心配置 1] 坐标序列化顺序
        # 原版只有 z 和 hilbert。
        # 这里为了配合 Plus 版的多视角重序列化，建议主要保持 'z' 顺序即可，
        # 因为代码内部会强制使用 'z' 进行重排。
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
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
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        
        # [核心配置 2] 新增参数：大核 CPE
        # 推荐尝试 5 或 7。这里设为 5。
        cpe_kernel_size=5 
    ),
    hidden_dim=256
)

# ==============================================================================
# Data Settings
# ==============================================================================
# 保持原样，与你提供的 keypoint_ptv3.py 一致
num_worker=4
batch_size=4
data_root = "/home/gzh/point/DataSets"

grid_size_train = 0.02
grid_size_val = 0.02

data = dict(
    num_workers=8,
    batch_size=8, 
    sample_rate=16,
    train=dict(
        type="KeypointDataset",
        split="train",
        data_root=data_root,
        transform=[
            dict(type="Update", keys_dict=dict(index_valid_keys=["coord", "feat"], grid_size=grid_size_train)),
            dict(type="GridSample", grid_size=grid_size_train, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="ToTensor"),
            dict(type="Collect", 
                 keys=("coord", "grid_coord", "feat", "target", "grid_size", "scale"), 
                 offset_keys_dict=dict(offset="coord"), 
                 feat_keys=("feat",))
        ],
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

    # [新增] 这里的 Evaluator 会在每个 epoch 后运行，计算指标并填入 comm_info
    dict(type="KeypointEvaluator"),
    
    # [修改] 开启 save_best=True。它会自动读取 KeypointEvaluator 提供的指标
    # 因为我们传的是 -distance，所以它会把“负得最少”的模型当做最佳模型保存
    dict(type="CheckpointSaver", save_freq=5)
]