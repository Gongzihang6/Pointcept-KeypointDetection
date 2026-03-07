"""
KeypointSwin3DPlus: Swin3D + Voting Mechanism
用于替代简单的 MLP 回归头，通过局部投票机制提高平滑曲面上的关键点回归精度。

Author: Assistant
Based on: Swin3D implementation in Pointcept
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils import offset2batch
from pointcept.models.utils.structure import Point

@MODELS.register_module("KeypointSwin3DVote")
class KeypointSwin3DVote(nn.Module):
    def __init__(self, 
                 backbone_conf, 
                 num_keypoints=6, 
                 hidden_dim=256, 
                 vote_radius=0.4):
        """
        Args:
            backbone_conf (dict): Swin3D Backbone 的配置
            num_keypoints (int): 关键点数量 (默认 6)
            hidden_dim (int): 投票头的隐藏层维度
            vote_radius (float): 投票半径 (单位: 米)。
                                 训练时，只有距离真值小于此半径的点才会被计算 Loss。
                                 建议设置为物体尺度的 1/5 到 1/3 (例如猪长1.8m，设为 0.3-0.5)。
        """
        super().__init__()
        # 1. 构建骨干网络 (Swin3D)
        self.backbone = build_model(backbone_conf)
        self.num_keypoints = num_keypoints
        self.vote_radius = vote_radius
        
        # 2. 自动获取骨干输出通道数
        # 注意：Swin3D 的 channels 通常是一个列表，Swin3DUNet 输出的通道数通常对应第一个 stage 的通道数
        # 如果配置文件没写，默认给 96 (Swin3D-Small 的默认值)
        if 'channels' in backbone_conf:
            in_channels = backbone_conf['channels'][0]
        else:
            in_channels = 96 

        # 3. 投票回归头 (Voting Head)
        # 作用: 逐点预测 [dx, dy, dz] * K
        self.vote_head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3), # 增加一点 Dropout 防止过拟合
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_keypoints * 3) # 输出 K 个 3D 偏移向量
        )

    def forward(self, data_dict):
        # === 0. 预处理 (核心修复) ===
        # 将字典封装为 Point 对象，它会自动根据 offset 生成 batch 索引
        point = Point(data_dict)
        
        # 获取 Batch 索引 [N] (例如: [0, 0, ..., 1, 1, ...])
        # Point 类会自动处理：如果 data_dict 只有 offset，它会调用 offset2batch 生成 batch
        batch_idx = point.batch 

        # 获取坐标和特征
        point_coord = point.coord # 等同于 data_dict["coord"]
        
        # === 1. 特征提取 ===
        # Swin3D Backbone 输出 [N, C]
        point_feat = self.backbone(data_dict) 

        # === 2. 预测偏移量 (Offset Prediction) ===
        # [N, C] -> [N, K*3]
        votes_flat = self.vote_head(point_feat)
        # Reshape -> [N, K, 3]
        votes_offset = votes_flat.view(-1, self.num_keypoints, 3)
        
        # === 3. 计算投票结果 (Voting) ===
        # 预测的关键点位置 = 当前点位置 + 预测的偏移量
        # point_coord: [N, 1, 3] + votes_offset: [N, K, 3] -> [N, K, 3]
        pred_kps_per_point = point_coord.unsqueeze(1) + votes_offset

        # === 4. 训练逻辑 ===
        if self.training:
            target = data_dict["target"] 
            
            # 1. 获取动态参数
            B = batch_idx.max().item() + 1
            K = self.num_keypoints
            N = point_feat.shape[0]

            # 2. 智能重塑 Target 形状 (强制 Reshape)
            if target.numel() == B * K * 3:
                target = target.view(B, K, 3)    
                target_per_point = target[batch_idx] # [N, K, 3]
            elif target.shape[0] == N:
                target_per_point = target.view(N, K, 3)
            else:
                raise ValueError(f"Target shape mismatch.")

            # 3. 计算距离 [N, K]
            dist_to_gt = torch.norm(point_coord.unsqueeze(1) - target_per_point, p=2, dim=-1)
            
            # 4. 生成 Mask [N, K]
            mask = dist_to_gt < self.vote_radius
            
            # 5. 计算 Loss
            loss_reg_all = F.smooth_l1_loss(pred_kps_per_point, target_per_point, reduction='none')
            loss_reg_all = loss_reg_all.mean(dim=-1) # [N, K]
            
            div = mask.float().sum()
            if div < 1.0: div = 1.0
            
            loss_vote = (loss_reg_all * mask.float()).sum() / div

            # === 日志记录 ===
            log_vars = {"loss": loss_vote}
            
            # 6. 监控详细指标 (每个关键点的误差)
            with torch.no_grad():
                # 处理 Scale
                if "scale" in data_dict:
                    scale = data_dict["scale"] 
                    if scale.shape[0] != point_coord.shape[0]: 
                        scale = scale[batch_idx]
                    if scale.ndim == 1:
                        scale = scale.unsqueeze(-1) # [N, 1]
                    
                    # 真实物理距离 [N, K]
                    real_dist = dist_to_gt * scale
                else:
                    real_dist = dist_to_gt

                # (A) 记录总平均误差 (Masked)
                masked_dist_sum = (real_dist * mask.float()).sum()
                masked_count = mask.float().sum()
                
                if masked_count > 0:
                    log_vars["train/masked_dist_err"] = masked_dist_sum / masked_count
                else:
                    log_vars["train/masked_dist_err"] = 0.0
                
                # (B) [新增] 记录每个关键点的误差
                # mask: [N, K], real_dist: [N, K]
                # 我们在 dim=0 (点数维度) 上聚合
                
                for k in range(K):
                    # 取出第 k 个关键点的所有数据
                    k_mask = mask[:, k]          # [N]
                    k_dist = real_dist[:, k]     # [N]
                    
                    k_count = k_mask.float().sum()
                    
                    if k_count > 0:
                        k_err = (k_dist * k_mask.float()).sum() / k_count
                        log_vars[f"train/kp{k}_dist_err"] = k_err
                    else:
                        # 如果没有点在这一类关键点的 Mask 内，记为 0
                        log_vars[f"train/kp{k}_dist_err"] = 0.0

            return log_vars

        # === 5. 推理/验证逻辑 ===
        else:
            # batch_idx 已经在最开头通过 point.batch 获取到了，这里直接用
            batch_size = batch_idx.max() + 1
            final_preds = []

            for b in range(batch_size):
                # 取出当前 Batch 的所有点的投票结果
                idx = (batch_idx == b)
                # votes: [M, K, 3] (M 是该样本的点数)
                sample_votes = pred_kps_per_point[idx] 
                
                # 聚合策略: 取中位数 (Median)
                if sample_votes.shape[0] > 0:
                    sample_pred = sample_votes.median(dim=0).values # [K, 3]
                else:
                    # 极其罕见的保底：如果某个样本没有任何点（几乎不可能发生）
                    sample_pred = torch.zeros((self.num_keypoints, 3), device=point_coord.device)
                
                final_preds.append(sample_pred)

            final_preds = torch.stack(final_preds, dim=0) # [B, K, 3]

            return dict(pred=final_preds)

