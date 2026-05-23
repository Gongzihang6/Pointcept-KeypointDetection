import torch
import torch.nn as nn
from pointcept.models.builder import MODELS, build_model

@MODELS.register_module()
class OffsetKeypointSwin3D(nn.Module):
    def __init__(self, 
                 backbone_conf, 
                 num_keypoints=6, 
                 hidden_dim=256):
        super().__init__()
        # 1. 构建骨干 (Swin3D)
        self.backbone = build_model(backbone_conf)
        
        # 获取骨干输通道数
        if 'channels' in backbone_conf:
            in_channels = backbone_conf['channels'][0]
        else:
            in_channels = 96
            
        # 2. 预测头
        self.num_keypoints = num_keypoints
        output_dim = num_keypoints * 4 
        
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
        # === [构造 Swin3D 必须的 coord_feat] ===
        if "coord_feat" not in data_dict:
            coord = data_dict["coord"]
            feat = data_dict["feat"]
            
            try:
                stem = self.backbone.stem_layer
                if hasattr(stem, "conv_layers"): 
                    expected_channels = stem.conv_layers[0].in_channels
                elif hasattr(stem, "conv"):
                    expected_channels = stem.conv.in_channels
                else:
                    expected_channels = feat.shape[1]
            except Exception:
                expected_channels = feat.shape[1]

            if expected_channels == feat.shape[1] + 3:
                data_dict["coord_feat"] = torch.cat([coord, feat], dim=1)
            else:
                data_dict["coord_feat"] = feat
                
        # 1. 提取所有点的稠密特征
        output = self.backbone(data_dict)
        if hasattr(output, "F"): 
            feat = output.F
        else: 
            feat = output
        
        # 2. 对每个点单独做回归预测
        pred_flat = self.head(feat)
        pred = pred_flat.view(-1, self.num_keypoints, 4) 
        
        result_dict = {}
        
        if "target" in data_dict:
            target = data_dict["target"] 

            offset_gt = target[..., :3]
            mask_gt = target[..., 3]
            
            offset_pred = pred[..., :3]
            mask_logits = pred[..., 3]  
            
            cls_loss = self.cls_criterion(mask_logits, mask_gt).mean()
            valid_mask = (mask_gt > 0.5).float()
            valid_mask_exp = valid_mask.unsqueeze(-1)
            
            raw_reg_loss = self.reg_criterion(offset_pred, offset_gt)
            masked_reg_loss = raw_reg_loss * valid_mask_exp
            reg_loss = masked_reg_loss.sum() / (valid_mask_exp.sum() * 3 + 1e-6)
            
            loss = cls_loss + reg_loss * 2.0 
            
            result_dict["loss"] = loss

            if self.training:
                with torch.no_grad():
                    result_dict["train/cls_loss"] = cls_loss.item()
                    result_dict["train/reg_loss"] = reg_loss.item()
                    result_dict["train/offset_l1_err"] = (torch.abs(offset_pred - offset_gt) * valid_mask_exp).sum() / (valid_mask_exp.sum() * 3 + 1e-6)

                    dist = torch.norm(offset_pred - offset_gt, p=2, dim=-1)
                    
                    if "scale" in data_dict and "offset" in data_dict:
                        scale = data_dict["scale"].view(-1)
                        offset = data_dict["offset"]
                        
                        b = torch.zeros(offset[-1], dtype=torch.long, device=offset.device)
                        if len(offset) > 1:
                            b[offset[:-1]] = 1
                        batch_idx = torch.cumsum(b, dim=0)
                        
                        if len(batch_idx) != len(dist):
                            if hasattr(output, "C"):
                                batch_idx = output.C[:, 0].to(torch.long)
                        
                        point_scale = scale[batch_idx].unsqueeze(-1)
                        dist = dist * point_scale
                        
                    valid_mask_f = valid_mask.float()
                    valid_sum = valid_mask_f.sum(dim=0).clamp(min=1e-6)
                    kp_real_dist = (dist * valid_mask_f).sum(dim=0) / valid_sum
                    
                    kp_real_dist[valid_mask_f.sum(dim=0) == 0] = 0.0

                    result_dict["train/mean_dist"] = kp_real_dist.mean().item()
                    for i in range(self.num_keypoints):
                        result_dict[f"train/kp{i}_dist"] = kp_real_dist[i].item()

        if not self.training:
            final_pred = pred.clone()
            final_pred[..., 3] = torch.sigmoid(pred[..., 3])
            result_dict["pred"] = final_pred

        return result_dict