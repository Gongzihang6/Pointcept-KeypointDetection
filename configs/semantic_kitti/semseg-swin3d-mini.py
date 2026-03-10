"""
作用：基于 Swin3D 的 SemanticKITTI 语义分割极限优化配置文件 (适配 Mini 数据集与笔记本显卡)
功能：定义了 Swin3D 的主干网络架构、数据加载管线、数据增强策略以及训练优化器。
实现了什么：
    1. 引入了 DefaultSegmentor 和 CrossEntropyLoss，将任务从回归转为 19 分类的语义分割。
    2. 极限精简了 Data Transform，去掉了耗时的弹性扭曲等复杂增强，只保留了必须的体素化采样 (GridSample)，大幅降低 CPU 负载。
    3. 将全局 batch_size 压缩到了 2，保护笔记本显存。
怎么实现的：通过 Pointcept 的动态注册机制 (Registry)，在运行时读取这些字典配置，实例化对应的 Python/C++ 底层类。
"""

_base_ = ["../_base_/default_runtime.py"]

# ==============================================================================
# 全局训练与硬件策略配置
# ==============================================================================
weight = None
resume = False
evaluate = True
test_only = False
seed = 42

# 【防炸显存核心配置】
batch_size_per_gpu = 1        # 每次喂给显卡 2 帧点云
batch_size_val_per_gpu = 1    # 验证时每次 1 帧
num_worker_per_gpu = 2        # CPU 数据加载线程数，笔记本不宜开太大

# 【快速过拟合配置】
epochs = 20                   # 小数据集跑 20 轮即可见效
eval_epoch = 5                # 每隔 5 轮评估一次 mIoU
save_path = "exp/SemanticKITTI_Swin3D_Mini"
# ==============================================================================
# 模型架构设定 (DefaultSegmentor + Swin3D)
# ==============================================================================
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="Swin3D-v1m1",
        in_channels=4,        # 输入通道: x, y, z, intensity
        num_classes=19,       # SemanticKITTI 的有效语义类别数为 19 类
        
        base_grid_size=0.05,  # SemanticKITTI 标准体素分辨率 (5厘米)
        quant_size=50,
        num_layers=4,
        
        depths=[2, 2, 6, 2],
        channels=[64, 128, 256, 512],
        num_heads=[4, 8, 16, 32], # 必须全为偶数
        window_sizes=[5, 7, 7, 7],
        
        up_k=3,
        drop_path_rate=0.2,
        stem_transformer=True,
        down_stride=2,
        upsample="linear",
        knn_down=True,
        cRSE="XYZ_RGB",       # 这里的 RGB 实际上代指附带的特征 (intensity)
        fp16_mode=1,          # 开启混合精度，进一步节省显存并提速
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1) # -1 代表忽略未标注区域
    ]
)

# ==============================================================================
# 数据集与处理管线 (Data Transform)
# ==============================================================================
# 【极其重要】指向你的微型数据集路径
data_root = "/home/gzh/point/Pointcept-KeypointDetection/SemanticFAST-LIO2-prediction/SemanticKITTI_Mini"
grid_size = 0.05

data = dict(
    num_classes=19,
    ignore_index=-1,
    names=["car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist", "motorcyclist", "road", "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign"],
    
    train=dict(
        type="SemanticKITTIDataset",
        split="train",
        data_root=data_root,
        transform=[
            # 精简版数据处理，只保留必须的核心步骤
            dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="ToTensor"),
            dict(type="Collect", 
                 keys=("coord", "grid_coord", "segment"), 
                 feat_keys=("coord", "strength"),
                 coord_feat_keys=("coord", "strength"))
        ],
        test_mode=False,
        loop=1,
    ),
    
    val=dict(
        type="SemanticKITTIDataset",
        split="val",
        data_root=data_root,
        transform=[
            dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="ToTensor"),
            dict(type="Collect", 
                 keys=("coord", "grid_coord", "segment"), 
                 feat_keys=("coord", "strength"),
                 coord_feat_keys=("coord", "strength"))
        ],
        test_mode=False,
    ),
    
    test=dict(
        type="SemanticKITTIDataset",
        split="val", # 调试阶段直接拿 val 集当 test 集跑
        data_root=data_root,
        transform=[
            dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="ToTensor"),
            dict(type="Collect", 
                 keys=("coord", "grid_coord", "segment"), 
                 feat_keys=("coord", "strength"),
                 coord_feat_keys=("coord", "strength"))
        ],
        test_mode=True,
    ),
)

# ==============================================================================
# 优化器与评估钩子
# ==============================================================================
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.05)
scheduler = dict(type="CosineAnnealingLR", eta_min=1e-5) 

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"), # 核心：使用语义分割专属的评估器计算 mIoU
    dict(type="CheckpointSaver", save_freq=10)
]
