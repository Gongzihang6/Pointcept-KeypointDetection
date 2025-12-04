# ==============================================================================
# 代码作用：PTv2 关键点检测模型架构
# 功能：
#   1. 封装 Point Transformer V2 (PTv2) 作为特征提取骨干。
#   2. 截取骨干网络输出的逐点特征。
#   3. 执行全局池化 (Global Average Pooling) 将点云特征聚合为图特征。
#   4. 使用 MLP 回归头预测 num_keypoints 个 3D 坐标。
# 实现细节：
#   - 兼容 Pointcept 新版接口 (返回 Point 对象) 和旧版接口 (返回 Tensor)。
#   - 包含完整的 Loss 计算和真实尺度误差监控 (train/mean_dist)。
# ==============================================================================

import torch
import torch.nn as nn
from pointcept.models.builder import MODELS, build_model

@MODELS.register_module("KeypointPTv2")
class KeypointPTv2(nn.Module):
    def __init__(self, 
                 backbone_conf, 
                 num_keypoints=6, 
                 hidden_dim=256):
        super().__init__()
        # 1. 构建骨干网络 (使用配置文件中定义的 PTv2 参数)
        self.backbone = build_model(backbone_conf)
        
        # 自动获取骨干输出通道数
        # PTv2 配置通常包含 dec_channels, 取最后一层作为输出维度
        if 'dec_channels' in backbone_conf:
            in_channels = backbone_conf['dec_channels'][0]
        else:
            in_channels = 256 # 默认值，如果配置中没写

        # 2. 回归头 (Regression Head)
        self.num_keypoints = num_keypoints
        output_dim = num_keypoints * 3
        
        self.reg_head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 3. 损失函数
        self.criterion = nn.MSELoss()

    def forward(self, data_dict):
        # === 1. 特征提取 (Backbone) ===
        # PTv2 骨干通常返回一个 Point 对象 (包含 .feat, .coord 等) 或 字典
        # 我们假设 data_dict 已经包含 grid_coord (由 Dataset 处理)
        
        # 某些 PTv2 版本可能直接返回 Logits (如果 num_classes > 0)
        # 我们在 Config 中设置 num_classes=None 或 0 来确保获取特征，
        # 或者在这里直接访问 backbone 的 encoder/decoder 输出。
        # Pointcept 惯例：backbone(data_dict) 返回包含 'feat' 的对象。
        
        feat_output = self.backbone(data_dict)
        
        # 兼容性处理：提取特征 tensor 和 batch 索引
        if hasattr(feat_output, "feat"): # 新版 Pointcept (如 PTv3 结构)
            feat = feat_output.feat
            # 如果 feat_output 没有 offset，我们从 data_dict 获取
            offset = feat_output.offset if hasattr(feat_output, "offset") else data_dict["offset"]
        elif isinstance(feat_output, dict):
            feat = feat_output["feat"]
            offset = data_dict["offset"]
        else: # 旧版可能直接返回 Tensor
            feat = feat_output
            offset = data_dict["offset"]

        # === 2. 全局池化 (Global Pooling) ===
        # 将逐点特征 (N, C) 聚合为 Batch 特征 (B, C)
        # 使用 offset 切片每个样本
        offset = offset.int()
        batch_feats = []
        start = 0
        for i in range(len(offset)):
            end = offset[i]
            # 对当前样本的所有点取平均
            sample_feat = feat[start:end].mean(dim=0) 
            batch_feats.append(sample_feat)
            start = end
        
        global_feat = torch.stack(batch_feats, dim=0) # (B, C)

        # === 3. 关键点回归 ===
        pred_flat = self.reg_head(global_feat)
        pred = pred_flat.view(-1, self.num_keypoints, 3)

        result_dict = {}

        # === 4. Loss 与 监控 ===
        if "target" in data_dict:
            target = data_dict["target"]
            
            # Loss 计算
            pred_for_loss = pred if pred.shape == target.shape else pred.view(-1, 3)
            loss = self.criterion(pred_for_loss, target)
            if loss.ndim > 0: loss = loss.mean()
            result_dict["loss"] = loss

            # 真实物理尺度误差监控
            if self.training:
                with torch.no_grad():
                    k = self.num_keypoints
                    # (B, K, 3)
                    pred_metric = pred.view(-1, k, 3)
                    target_metric = target.view(-1, k, 3)

                    # 欧氏距离
                    dist = torch.norm(pred_metric - target_metric, p=2, dim=-1)

                    # 逆归一化 (还原到米/毫米)
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