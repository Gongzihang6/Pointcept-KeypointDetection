import torch
import torch.nn as nn
from pointcept.models.builder import MODELS, build_model
import torch.nn.functional as F

@MODELS.register_module()
class OffsetKeypointPTv3(nn.Module):
    def __init__(self, 
                 backbone_conf, 
                 num_keypoints=6, 
                 hidden_dim=256):
        super().__init__()
        # 1. 构建骨干 (PTv3)
        self.backbone = build_model(backbone_conf)
        
        # 获取骨干输出维度 (假设是 dec_channels 的第一个值)
        in_channels = backbone_conf['dec_channels'][0]

        # 2. 预测头
        self.num_keypoints = num_keypoints
        # 相比之前的全局回归，现在是每个点稠密预测：
        # 对于每个关键点预测4个值：(dx, dy, dz) + 1个 mask置信度
        output_dim = num_keypoints * 4 
        
        # 这里因为是对每个 point 进行 dense 预测，因此直接复用 Linear 在特征维上做仿射变换
        self.head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 定义 Loss (L1 距离用于补偿回归，BCE 用于 mask 预测)
        self.reg_criterion = nn.L1Loss(reduction='none') 
        self.cls_criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, data_dict):
        # 1. 提取所有点的稠密特征
        point_output = self.backbone(data_dict)
        feat = point_output.feat  # (N_total_points, in_channels)
        
        # 2. 对每个点单独做回归预测
        # pred_flat: (N_total_points, K * 4)
        pred_flat = self.head(feat)
        # 转为 (N_total_points, K, 4) 便于拆分 offset 和 mask
        pred = pred_flat.view(-1, self.num_keypoints, 4) 
        
        result_dict = {}
        
        if "target" in data_dict:
            target = data_dict["target"] # GT也是 (N_total_points, K, 4)

            # --- Target 拆包 ---
            # offset_gt: (N, K, 3), mask_gt: (N, K)
            offset_gt = target[..., :3]
            mask_gt = target[..., 3]
            
            # --- Pred 拆包 ---
            offset_pred = pred[..., :3]
            mask_logits = pred[..., 3]  
            
            # ======== 计算 Mask 二分类 Loss (Focal/BCE) ========
            # 对所有点不管是不是有用的都算分类 loss
            cls_loss = self.cls_criterion(mask_logits, mask_gt).mean()
            
            # ======== 计算 Offset 回归 Loss (L1) ========
            # 因为只有属于有效区域的点（即 mask_gt == 1 的点）它的平移目标才是真正精准、有物理意义的。
            # 对于 mask=0 的背景杂点，不应该强迫它回归到远处的关键点。
            valid_mask = (mask_gt > 0.5).float() # (N, K)
            
            # unsqueeze 以便用于乘以 3D 的 offset, valid_mask_exp: (N, K, 1)
            valid_mask_exp = valid_mask.unsqueeze(-1)
            
            # l1_loss 矩阵: (N, K, 3)
            raw_reg_loss = self.reg_criterion(offset_pred, offset_gt)
            
            # 只有 valid_mask 范围内的 L1 取值才会被汇总
            masked_reg_loss = raw_reg_loss * valid_mask_exp
            
            # 计算总回归 Loss 均值 (加上 1e-6 防止除 0)
            reg_loss = masked_reg_loss.sum() / (valid_mask_exp.sum() * 3 + 1e-6)
            
            # 总 Loss (可以增加权重，比如回归的 loss 可以调大)
            # 或者按照原版形式仅传出一个 scalar 'loss'
            loss = cls_loss + reg_loss * 2.0 
            
            result_dict["loss"] = loss

            # === [训练监控] ===
            if self.training:
                # 为了用 WandB/TensorBoard 观测我们加点细类的记录
                with torch.no_grad():
                    result_dict["train/cls_loss"] = cls_loss.item()
                    result_dict["train/reg_loss"] = reg_loss.item()
                    
                    # 近似还原一下真实尺度的距离误差 （可选，这里简化仅为 train 中查看收敛情况）
                    # 在 Offset 任务中我们更需要观察网络对 Mask 和 相对Offset 的预测数值走向
                    result_dict["train/offset_l1_err"] = (torch.abs(offset_pred - offset_gt) * valid_mask_exp).sum() / (valid_mask_exp.sum() * 3 + 1e-6)

        # 把 sigmoid 处理过的 mask 附在预测结果中给 Evaluator/Inference 用
        if not self.training:
            # 推理阶段我们要把 logits 用 sigmoid 转为 0~1 的置信度
            final_pred = pred.clone()
            final_pred[..., 3] = torch.sigmoid(pred[..., 3])
            result_dict["pred"] = final_pred

        return result_dict
