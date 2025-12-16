import torch
import torch.nn as nn
import spconv.pytorch as spconv
from torch_geometric.utils import scatter

from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch
# 导入原始 Backbone
from pointcept.models.oacnns import OACNNs

@MODELS.register_module("KeypointOACNNs")
class KeypointOACNNs(OACNNs):
    """
    基于 OA-CNNs Backbone 的关键点回归模型。
    继承自 OACNNs 以重用其初始化逻辑，但重写 forward 以支持回归任务。
    """
    def __init__(self, 
                 num_keypoints=6, 
                 hidden_dim=256, 
                 **kwargs):
        # 1. 初始化 OACNNs Backbone
        # kwargs 将包含 enc_channels, dec_channels, groups 等所有骨干参数
        super().__init__(num_classes=num_keypoints, **kwargs)
        
        # 2. 移除原有的分割头 (self.final) 以节省参数 (可选)
        # OACNNs 的 self.final 是一个 SubMConv3d，我们不需要它
        self.final = nn.Identity()

        # 3. 确定回归头的输入维度
        # OACNNs 的输出维度是 Decoder 最后一层的输出维度 (dec_channels[0])
        # 默认配置通常是 256 或 96，取决于具体配置
        if 'dec_channels' in kwargs:
            in_channels = kwargs['dec_channels'][0]
        else:
            # Fallback based on default values in OACNNs source
            in_channels = 96 

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
        # === A. 数据准备 (复制自 OACNNs.forward) ===
        discrete_coord = input_dict["grid_coord"]
        feat = input_dict["feat"]
        offset = input_dict["offset"]
        batch = offset2batch(offset)

        # 构建稀疏张量
        x = spconv.SparseConvTensor(
            features=feat,
            indices=torch.cat([batch.unsqueeze(-1), discrete_coord], dim=1)
            .int()
            .contiguous(),
            spatial_shape=torch.add(
                torch.max(discrete_coord, dim=0).values, 1
            ).tolist(),
            batch_size=batch[-1].tolist() + 1,
        )

        # === B. 骨干网络前向传播 (复制自 OACNNs.forward) ===
        x = self.stem(x)
        skips = [x]
        for i in range(self.num_stages):
            x = self.enc[i](x)
            skips.append(x)
        x = skips.pop(-1)
        for i in reversed(range(self.num_stages)):
            skip = skips.pop(-1)
            x = self.dec[i](x, skip)
        
        # 注意：这里我们跳过了 self.final(x)，直接使用 decoder 输出的特征
        
        # === C. 全局池化 (Global Pooling) ===
        # x 是 SparseConvTensor，x.features 是 (N_voxel, C)
        # x.indices 是 (N_voxel, 4)，第 0 列是 batch_idx
        out_feat = x.features
        out_batch_idx = x.indices[:, 0].long()
        batch_size = x.batch_size

        # 使用 scatter_mean 对每个 Batch 的体素特征求平均
        # 结果维度: (Batch_Size, C)
        global_feat = scatter(out_feat, out_batch_idx, dim=0, dim_size=batch_size, reduce="mean")

        # === D. 关键点回归 ===
        pred_flat = self.reg_head(global_feat)
        pred = pred_flat.view(-1, self.num_keypoints, 3)

        result_dict = {}

        # === E. Loss 计算与日志 ===
        if "target" in input_dict:
            target = input_dict["target"]
            
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

                    # 计算欧氏距离 (Batch, K)
                    dist = torch.norm(pred_metric - target_metric, p=2, dim=-1)

                    # 反归一化 Scale (如果存在)
                    if "scale" in input_dict:
                        scale = input_dict["scale"]
                        if scale.ndim == 1: scale = scale.view(-1, 1)
                        dist = dist * scale
                    
                    # 记录平均误差
                    result_dict["train/mean_dist"] = dist.mean()
                    
                    # 记录每个关键点的误差
                    kp_dist_mean = dist.mean(dim=0)
                    for i in range(k):
                        result_dict[f"train/kp{i}_dist"] = kp_dist_mean[i]

        if self.training:
            return result_dict
        else:
            result_dict["pred"] = pred
            return result_dict
