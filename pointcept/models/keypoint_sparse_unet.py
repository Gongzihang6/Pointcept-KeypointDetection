"""
Keypoint Regression driven by SparseUNet (SpConv)
"""
import torch
import torch.nn as nn
import spconv.pytorch as spconv
from torch_geometric.utils import scatter

from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch
# 导入原始 Backbone
from pointcept.models.sparse_unet.spconv_unet_v1m1_base import SpUNetBase

@MODELS.register_module("KeypointSparseUNet")
class KeypointSparseUNet(SpUNetBase):
    def __init__(self, 
                 num_keypoints=6, 
                 hidden_dim=256, 
                 **kwargs):
        # [Fix] 如果 config 中包含了 num_classes，先将其弹出，避免与下面的 num_classes=0 冲突
        if "num_classes" in kwargs:
            kwargs.pop("num_classes")

        # 1. 初始化 Backbone
        # num_classes 设为 0，因为我们要自己构建回归头，不需要原有的分类头
        super().__init__(num_classes=0, **kwargs)
        
        # 2. 移除原有的 final 层 (可选，为了节省一点点参数)
        self.final = nn.Identity()

        # 3. 确定回归头的输入维度
        # 根据 SpUNetBase 的逻辑：
        # 如果 enc_mode=False (默认), 输出特征维度是 dec_channels[-1] -> channels[0] (通常是32)
        # 如果 enc_mode=True, 输出特征维度是 channels[num_stages-1] (通常是256)
        
        # 为了稳健获取维度，我们查看 self.channels
        if self.enc_mode:
            in_channels = self.channels[self.num_stages - 1]
        else:
            # Decoder 最后一层输出维度 = channels[0] (根据 layers/channels 的对称设计)
            in_channels = self.channels[-1]

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

    def forward(self, input_dict):
        # === A. 数据准备 (复制自 SpUNetBase.forward) ===
        grid_coord = input_dict["grid_coord"]
        feat = input_dict["feat"]
        offset = input_dict["offset"]
        batch = offset2batch(offset)

        # 构建 SparseConvTensor
        # 注意：这里我们保留了 spconv 的 spatial_shape 计算逻辑
        sparse_shape = torch.add(torch.max(grid_coord, dim=0).values, 96).tolist()
        x = spconv.SparseConvTensor(
            features=feat,
            indices=torch.cat(
                [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=batch[-1].tolist() + 1,
        )

        # === B. Backbone 前向传播 (复制自 SpUNetBase.forward) ===
        x = self.conv_input(x)
        skips = [x]
        
        # Encoder
        for s in range(self.num_stages):
            x = self.down[s](x)
            x = self.enc[s](x)
            skips.append(x)
        x = skips.pop(-1)
        
        # Decoder (仅当 enc_mode=False 时执行)
        if not self.enc_mode:
            for s in reversed(range(self.num_stages)):
                x = self.up[s](x)
                skip = skips.pop(-1)
                # Concatenate skip connections
                x = x.replace_feature(torch.cat((x.features, skip.features), dim=1))
                x = self.dec[s](x)

        # 注意：我们跳过了 self.final(x)，因为那是为分割设计的 1x1 卷积
        
        # === C. 全局池化 (Global Pooling) ===
        # 获取特征和 Batch 索引
        out_feat = x.features
        out_batch_idx = x.indices[:, 0].long()
        batch_size = x.batch_size # spconv 2.x 中这是一个 int

        # 使用 scatter_mean 进行全局平均池化
        # (N_voxel, C) -> (Batch_Size, C)
        global_feat = scatter(out_feat, out_batch_idx, dim=0, dim_size=batch_size, reduce="mean")

        # === D. 关键点回归 ===
        pred_flat = self.reg_head(global_feat)
        pred = pred_flat.view(-1, self.num_keypoints, 3)

        result_dict = {}

        # === E. Loss 计算与日志 ===
        if "target" in input_dict:
            target = input_dict["target"]
            
            # [Fix] 处理 DataLoader 可能展平 target 的情况 (Batch_Size > 1 时)
            pred_for_loss = pred if pred.shape == target.shape else pred.view(-1, 3)
            
            loss = self.criterion(pred_for_loss, target)
            if loss.ndim > 0: loss = loss.mean()
            result_dict["loss"] = loss

            if self.training:
                with torch.no_grad():
                    k = self.num_keypoints
                    pred_metric = pred.view(-1, k, 3)
                    # 确保 target 形状正确用于计算距离
                    target_metric = target.view(-1, k, 3)

                    dist = torch.norm(pred_metric - target_metric, p=2, dim=-1)

                    if "scale" in input_dict:
                        scale = input_dict["scale"]
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
