_base_ = ["../_base_/default_runtime.py"]

# ==============================================================================
# 代码作用：PTv1 关键点检测模型训练配置文件
# 功能：定义模型参数、数据预处理流程、优化器和训练钩子
# 实现逻辑：
#   1. 使用 KeypointPTv1-50 作为骨干网络 (基于 Point Transformer V1)
#   2. 数据处理与 PTv3 保持一致，但注意输入通道数的匹配
#   3. 使用 MSELoss 和自定义的距离评估指标
# ==============================================================================

# ==============================================================================
# Global Settings
# ==============================================================================
epoch = 100 

# ==============================================================================
# Model Settings
# ==============================================================================
model = dict(
    type="KeypointPTv1-50",  # 使用我们实现的 PTv1-50 层版本
    # 输入通道数计算：
    # 原始数据特征 feat=4 (法向量+曲率), 坐标 coord=3
    # PTv1 的实现通常会将 coord 和 feat 拼接，所以 3 + 4 = 7
    in_channels=7,           
    num_keypoints=6,         # 预测 6 个关键点
    hidden_dim=256,          # 回归头隐藏层维度
)

# ==============================================================================
# Data Settings
# ==============================================================================
num_worker = 4
batch_size = 8 
data_root = "/home/gzh/point/DataSets"
grid_size_val = 0.02
save_path = "exp/keypoint_ptv1"

# 数据增强与加载流程 (保持原逻辑，PTv1 同样适用体素化后的点云)
data = dict(
    train=dict(
        type="KeypointDataset",
        split="train",
        data_root=data_root,
        transform=[
            # 1. 更新 grid_size
            dict(type="Update", keys_dict=dict(index_valid_keys=["coord", "feat"], grid_size=grid_size_val)),
            # 2. 体素化下采样 (GridSample)
            dict(type="GridSample", grid_size=grid_size_val, hash_type="fnv", mode="train", return_grid_coord=True),
            # 3. 转 Tensor
            dict(type="ToTensor"),
            # 4. 收集数据 (Collect)
            # offset_keys_dict 会让 collate_fn 生成 batch 索引偏移量 'offset'，这对 PTv1 至关重要
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
    # save_best=True 会自动读取 KeypointEvaluator 输出的指标
    # 通常 KeypointEvaluator 会输出 "mean_dist"，值越小越好，Pointcept 会处理负号逻辑或自动判断
    dict(type="CheckpointSaver", save_freq=20) 
]
