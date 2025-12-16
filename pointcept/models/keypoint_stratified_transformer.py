import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch  # [Fix] 使用正确的导入路径
# 导入原始 Backbone
from pointcept.models.stratified_transformer.stratified_transformer_v1m2_refine import StratifiedTransformer

@MODELS.register_module("KeypointStratifiedTransformer")
class KeypointStratifiedTransformer(StratifiedTransformer):
    """
    基于 Stratified Transformer (v1m2) 的关键点回归模型。
    通过继承复用代码，替换分类头为回归头。
    """
    def __init__(self, 
                 num_keypoints=6, 
                 hidden_dim=256, 
                 **kwargs):
        # 1. 初始化 Backbone
        # 注意：ST 的 num_classes 在这里其实没用了，因为我们会替换掉 classifier
        # 但为了兼容父类 __init__，我们随便传一个值
        super().__init__(num_classes=num_keypoints, **kwargs)
        
        # 2. 替换原有的分类头
        # ST 的 classifier 是一个 MLP，我们把它替换为 Identity，
        # 这样 super().forward() 就会返回 Backbone 输出的特征 (N, C)
        self.classifier = nn.Identity()

        # 3. 确定回归头的输入维度
        # ST 的输出维度通常是 channels[0] (Decoder 的最后一层输出)
        # 默认配置 channels=(48, 96, 192, 384, 384)，所以 channels[0] = 48
        if 'channels' in kwargs:
            in_channels = kwargs['channels'][0]
        else:
            in_channels = 48 # 默认值

        self.num_keypoints = num_keypoints
        output_dim = num_keypoints * 3

        # 4. 构建回归头 (Regression Head)
        self.reg_head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

        # 5. 损失函数
        self.criterion = nn.MSELoss()

    def forward(self, data_dict):
        # === A. 获取 Backbone 特征 ===
        # 调用父类的 forward，由于 self.classifier 被替换为 Identity，
        # 这里返回的是点云的逐点特征，维度 (N, C)
        point_features = super().forward(data_dict)
        
        # === B. 全局池化 (Global Pooling) ===
        # Stratified Transformer 内部处理了 offset，但 forward 没有返回 batch 索引
        # 我们需要利用输入的 offset 重新计算 batch 索引
        offset = data_dict["offset"]
        batch_idx = offset2batch(offset)
        
        # 确保 batch_size 数量正确
        batch_size = len(offset)

        # 使用 scatter_mean 将每个样本的点特征聚合成一个全局特征向量
        # (N, C) -> (Batch_Size, C)
        global_feat = scatter(point_features, batch_idx, dim=0, dim_size=batch_size, reduce="mean")

        # === C. 关键点回归 ===
        pred_flat = self.reg_head(global_feat)
        pred = pred_flat.view(-1, self.num_keypoints, 3)

        result_dict = {}

        # === D. Loss 计算与日志 ===
        if "target" in data_dict:
            target = data_dict["target"]
            
            # 确保维度匹配
            pred_for_loss = pred if pred.shape == target.shape else pred.view(-1, 3)
            
            loss = self.criterion(pred_for_loss, target)
            if loss.ndim > 0: loss = loss.mean()
            result_dict["loss"] = loss

            if self.training:
                with torch.no_grad():
                    k = self.num_keypoints
                    pred_metric = pred.view(-1, k, 3)
                    target_metric = target.view(-1, k, 3)

                    dist = torch.norm(pred_metric - target_metric, p=2, dim=-1)

                    if "scale" in data_dict:
                        scale = data_dict["scale"]
                        if scale.ndim == 1: scale = scale.view(-1, 1)
                        dist = dist * scale
                    
                    result_dict["train/mean_dist"] = dist.mean()
                    
                    kp_dist_mean = dist.mean(dim=0)
                    for i in range(k):
                        result_dict[f"train/kp{i}_dist"] = kp_dist_mean[i]

        if self.training:
            return result_dict
        else:
            result_dict["pred"] = pred
            return result_dict
