# configs/my_dataset/keypoint_octformer.py

_base_ = ["../_base_/default_runtime.py"]

# ==============================================================================
# Global Settings
# ==============================================================================
epoch = 100 
num_worker=4
batch_size=8
save_path = "exp/keypoint_octformer"
# ==============================================================================
# Model Settings
# ==============================================================================
model = dict(
    type="KeypointOctFormer",
    in_channels=4,        # 输入通道: 3(Normal) + 1(Curvature)
    num_keypoints=6,      # 请根据实际关键点数量修改
    fpn_channels=168,     # OctFormer Decoder 输出维度
    hidden_dim=256,       # 回归头隐藏层
    
    # OctFormer 结构参数 (与 semseg-octformer-v1m1-0-base.py 保持一致)
    channels=(96, 192, 384, 384),   # 对应论文中4个阶段的C、2C、4C、4C
    num_blocks=(2, 2, 18, 2),   # 对应论文中的N1、N2、N3、N4，表示每个阶段OctFOrmer Block块的数量
    num_heads=(6, 12, 24, 24),  # 每个阶段OctFormer Block 中多头自注意力的头数量
    patch_size=26,      # 每个窗口计算自注意力的点个数
    stem_down=2,        # 指的是进入stage1之前，Embedding模块使八叉树的深度降低2级，反映到分辨率上就是点个数变为1/4
    head_up=2,
    dilation=4,         # OctFormer Block中第二个OctFormer中膨胀自注意力的膨胀系数
    drop_path=0.5,
    nempty=True,
    octree_depth=11,      # 可根据点云场景大小调整，ScanNet常用11
    octree_full_depth=2,
    octree_scale_factor=10.24, # 坐标缩放因子
)

# ==============================================================================
# Data Settings
# ==============================================================================
data_root = "/home/gzh/point/DataSets" # 请确认为您的数据路径
grid_size_val = 0.02 # 网格采样大小

data = dict(
    train=dict(
        type="KeypointDataset",
        split="train",
        data_root=data_root,
        transform=[
            # 1. 数据注入 grid_size (KeypointDataset 加载 N*7 数据)
            # 假设 KeypointDataset 自动将前3列读为 coord，后4列读为 feat
            dict(type="Update", keys_dict=dict(index_valid_keys=["coord", "feat"], grid_size=grid_size_val)),
            
            # 2. GridSample: OctFormer 对点云密度敏感，通常不做过激的下采样，
            # 但为了统一流程和生成 offset，这里使用 GridSample
            dict(type="GridSample", 
                 grid_size=grid_size_val, 
                 hash_type="fnv", 
                 mode="train", 
                 return_grid_coord=True),
            
            # 3. 转 Tensor
            dict(type="ToTensor"),
            
            # 4. 收集字段
            # OctFormer 必须需要 "offset" 字段 (由 offset_keys_dict 生成)
            # 模型内部会从 "feat" 中拆分出 normal
            dict(type="Collect", 
                 keys=("coord", "feat", "target", "grid_size", "scale"), 
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
                 keys=("coord", "feat", "target", "grid_size", "scale"), 
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
                 keys=("coord", "feat", "target", "grid_size", "scale"), 
                 offset_keys_dict=dict(offset="coord"), 
                 feat_keys=("feat",))
        ],
    ),
)

# ==============================================================================
# Training Settings
# ==============================================================================
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.05) # OctFormer 学习率通常比 PTv3 低一点

scheduler = dict(type="CosineAnnealingLR", eta_min=1e-5)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=10),
    dict(type="InformationWriter"),
    dict(type="KeypointEvaluator"), # 确保此 Evaluator 已实现并注册
    dict(type="CheckpointSaver", save_freq=20) 
]
