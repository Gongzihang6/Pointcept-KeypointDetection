import torch
import torch.nn as nn
from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils.structure import Point
import torch_scatter

@MODELS.register_module()
class KeypointPTv3(nn.Module):
    def __init__(self, 
                 backbone_conf, 
                 num_keypoints=6, 
                 hidden_dim=256):
        super().__init__()
        # 1. 构建骨干 (PTv3)
        self.backbone = build_model(backbone_conf)
        
        # 获取骨干输出维度 (假设是 dec_channels 的第一个值)
        in_channels = backbone_conf['dec_channels'][0]

        # 2. 回归头 (MLP)
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
        
        # 定义 Loss
        self.criterion = nn.MSELoss()

    def forward(self, data_dict):
        # 1. 提取特征
        point_output = self.backbone(data_dict)
        feat = point_output.feat
        batch = point_output.batch
        
        # 2. 全局池化
        global_feat = torch_scatter.scatter_mean(feat, batch, dim=0)
        
        # 3. 回归预测
        pred_flat = self.reg_head(global_feat)
        pred = pred_flat.view(-1, self.num_keypoints, 3) 
        
        result_dict = {}
        
        if "target" in data_dict:
            target = data_dict["target"]

            # Loss 计算（在归一化空间计算更稳定）
            if pred.shape != target.shape:
                pred_for_loss = pred.view(-1, 3)
            else:
                pred_for_loss = pred

            loss = self.criterion(pred_for_loss, target)
            if loss.ndim > 0:
                loss = loss.mean()
            
            result_dict["loss"] = loss

            # === [训练监控] 计算真实物理尺度的距离误差 ===
            if self.training:
                with torch.no_grad():
                    k = self.num_keypoints
                    pred_metric = pred.view(-1, k, 3)
                    target_metric = target.view(-1, k, 3)

                    # 1. 计算归一化空间下的欧氏距离
                    dist = torch.norm(pred_metric - target_metric, p=2, dim=-1) # (B, K)

                    # 2. [核心修改] 使用 scale 进行逆归一化
                    if "scale" in data_dict:
                        scale = data_dict["scale"] # (B,) 或 (B, 1)
                        if scale.ndim == 1:
                            scale = scale.view(-1, 1) # 变成 (B, 1) 以便广播
                        
                        # 真实距离 = 归一化距离 * 缩放因子
                        dist = dist * scale
                    
                    # 3. 记录指标
                    result_dict["train/mean_dist"] = dist.mean()
                    
                    kp_dist_mean = dist.mean(dim=0)
                    for i in range(k):
                        result_dict[f"train/kp{i}_dist"] = kp_dist_mean[i]

        # 5. 返回结果
        if self.training:
            return result_dict
        else:
            result_dict["pred"] = pred
            return result_dict
