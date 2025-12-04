# ==============================================================================
# 代码作用：Swin3D 关键点检测模型架构 (最终修正版)
# ==============================================================================

import torch
import torch.nn as nn
from pointcept.models.builder import MODELS, build_model

@MODELS.register_module("KeypointSwin3D")
class KeypointSwin3D(nn.Module):
    def __init__(self, 
                 backbone_conf, 
                 num_keypoints=6, 
                 hidden_dim=256):
        super().__init__()
        # 1. 构建骨干网络 (Swin3D)
        self.backbone = build_model(backbone_conf)
        
        # 2. 自动获取骨干输出通道数
        # Swin3D 的 channels 参数通常是一个列表 [C1, C2, C3, C4]
        if 'channels' in backbone_conf:
            in_channels = backbone_conf['channels'][0]
        else:
            in_channels = 96 # 默认备用值

        # 3. 回归头 (Regression Head)
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
        
        # 4. 损失函数
        self.criterion = nn.MSELoss()

    def forward(self, data_dict):
        # === [核心修复] 构造 Swin3D 必须的 coord_feat ===
        if "coord_feat" not in data_dict:
            coord = data_dict["coord"]
            feat = data_dict["feat"]
            
            # 获取 Stem Layer (第一层卷积) 期望的输入通道数
            # 修正：根据 mink_layers.py，MinkConvBNRelu 使用 conv_layers (Sequential)
            try:
                stem = self.backbone.stem_layer
                if hasattr(stem, "conv_layers"): 
                    # 访问 Sequential 的第一个模块 (MinkowskiConvolution)
                    expected_channels = stem.conv_layers[0].in_channels
                elif hasattr(stem, "conv"):
                    expected_channels = stem.conv.in_channels
                else:
                    # 如果无法推断，默认使用配置中的 feat 维度
                    expected_channels = feat.shape[1]
            except Exception:
                expected_channels = feat.shape[1]

            # 根据模型期望决定是否拼接坐标
            if expected_channels == feat.shape[1] + 3:
                data_dict["coord_feat"] = torch.cat([coord, feat], dim=1)
            else:
                # 默认只使用特征 (如果是 4 通道)
                data_dict["coord_feat"] = feat

        # === 1. 特征提取 (Backbone) ===
        output = self.backbone(data_dict)
        
        # 兼容性处理：Swin3D 可能直接返回 Tensor，也可能返回 SparseTensor
        if hasattr(output, "F"): 
            feat = output.F
        else: 
            feat = output

        # === 2. 全局池化 (Global Pooling) ===
        # 使用 offset 将 batch 中每个样本的点特征聚合
        offset = data_dict["offset"].int()
        batch_feats = []
        start = 0
        
        # 注意：如果 Swin3D 使用了体素化导致点数变化，必须使用 offset2batch 等方式重新计算
        # 但 Pointcept 的 Swin3DUNet 在分割模式下通常会插值回原始点数，或者保持一一对应
        # 为了安全，这里做一个长度检查
        if feat.shape[0] != offset[-1].item():
            # 如果特征点数与原始 offset 不一致 (说明被体素化下采样了)
            # 我们需要重新计算一个 batch 索引
            batch_idx = data_dict.get("batch", None)
            if batch_idx is None:
                # 如果没有 batch 索引，尝试用 coordinates 推断 (SparseTensor 的 coordinates 第一列是 batch_idx)
                if hasattr(output, "C"):
                    batch_idx = output.C[:, 0]
                else:
                    raise RuntimeError("Output feature size mismatch and cannot infer batch index.")
            
            # 使用 scatter_mean 或简单的循环进行池化
            for b in range(len(offset)):
                mask = (batch_idx == b)
                if mask.any():
                    sample_feat = feat[mask].mean(dim=0)
                else:
                    sample_feat = torch.zeros_like(feat[0])
                batch_feats.append(sample_feat)
        else:
            # 正常情况：点数一一对应
            for i in range(len(offset)):
                end = offset[i]
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