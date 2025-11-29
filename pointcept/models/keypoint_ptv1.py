import torch
import torch.nn as nn
from pointcept.models.builder import MODELS
# 从 Point Transformer V1 的语义分割实现中导入基础模块
from pointcept.models.point_transformer.point_transformer_seg import TransitionDown, Bottleneck

@MODELS.register_module()
class KeypointPTv1(nn.Module):
    """
    KeypointPTv1: 基于 Point Transformer V1 的关键点检测模型
    
    参数:
        block (nn.Module): 基础模块，通常是 Bottleneck
        blocks (list): 每个阶段的模块堆叠数量列表
        in_channels (int): 输入特征通道数 (默认为 6, 如 xyz + rgb)
        num_keypoints (int): 需要预测的关键点数量
        hidden_dim (int): 回归头的隐藏层维度
        num_classes (int): (保留参数，用于兼容接口，本模型中不作为分类类别使用)
    """
    def __init__(self, 
                 block=Bottleneck, 
                 blocks=[1, 2, 3, 5, 2], 
                 in_channels=6, 
                 num_keypoints=6, 
                 hidden_dim=256,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        # PTv1 的标准通道配置
        self.in_planes, planes = in_channels, [32, 64, 128, 256, 512]
        share_planes = 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]

        # === 1. 构建骨干网络 (Encoder 1-5) ===
        # Stage 1: N/1
        self.enc1 = self._make_enc(
            block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0]
        )
        # Stage 2: N/4
        self.enc2 = self._make_enc(
            block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1]
        )
        # Stage 3: N/16
        self.enc3 = self._make_enc(
            block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2]
        )
        # Stage 4: N/64
        self.enc4 = self._make_enc(
            block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3]
        )
        # Stage 5: N/256
        self.enc5 = self._make_enc(
            block, planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4]
        )

        # === 2. 回归头 (Regression Head) ===
        self.num_keypoints = num_keypoints
        # 骨干最后一层输出维度是 512 (planes[4])
        backbone_out_dim = planes[4]
        output_dim = num_keypoints * 3
        
        self.reg_head = nn.Sequential(
            nn.Linear(backbone_out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

        # === 3. 损失函数 ===
        self.criterion = nn.MSELoss()

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        """构建 Encoder Stage 的辅助函数"""
        layers = [
            TransitionDown(self.in_planes, planes * block.expansion, stride, nsample)
        ]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks): # 注意：这里是 range(1, blocks) 因为 TransitionDown 算一层
            layers.append(
                block(self.in_planes, self.in_planes, share_planes, nsample=nsample)
            )
        return nn.Sequential(*layers)

    def forward(self, data_dict):
        # 1. 准备数据
        p0 = data_dict["coord"]
        x0 = data_dict["feat"]
        o0 = data_dict["offset"].int()
        
        # 如果输入只有xyz，则直接用xyz作为特征；否则拼接
        if self.in_channels == 3:
            x0 = p0
        else:
            x0 = torch.cat((p0, x0), 1)

        # 2. 骨干网络前向传播
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])

        # 3. 全局池化 (Global Pooling)
        # PTv1 使用 offset 记录每个样本的结束索引，这里手动进行 batch 维度的池化
        x_list = []
        for i in range(o5.shape[0]):
            if i == 0:
                s_i, e_i, cnt = 0, o5[0], o5[0]
            else:
                s_i, e_i, cnt = o5[i - 1], o5[i], o5[i] - o5[i - 1]
            
            # 对该样本的所有点特征求和并取平均
            x_b = x5[s_i:e_i, :].sum(0, True) / cnt
            x_list.append(x_b)
        
        global_feat = torch.cat(x_list, 0) # (Batch_Size, 512)

        # 4. 回归预测
        pred_flat = self.reg_head(global_feat)
        pred = pred_flat.view(-1, self.num_keypoints, 3)

        result_dict = {}

        # 5. Loss 计算与指标监控 (与 KeypointPTv3 保持一致)
        if "target" in data_dict:
            target = data_dict["target"]

            # 调整预测形状以匹配 target 计算 Loss
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

                    # 2. 使用 scale 进行逆归一化 (还原到真实物理尺寸)
                    if "scale" in data_dict:
                        scale = data_dict["scale"] # (B,) 或 (B, 1)
                        if scale.ndim == 1:
                            scale = scale.view(-1, 1) 
                        
                        dist = dist * scale
                    
                    # 3. 记录指标
                    result_dict["train/mean_dist"] = dist.mean()
                    
                    kp_dist_mean = dist.mean(dim=0)
                    for i in range(k):
                        result_dict[f"train/kp{i}_dist"] = kp_dist_mean[i]

        # 6. 返回结果
        if self.training:
            return result_dict
        else:
            result_dict["pred"] = pred
            return result_dict

# 注册不同深度的变体，方便在 Config 中调用
@MODELS.register_module("KeypointPTv1-26")
class KeypointPTv1_26(KeypointPTv1):
    def __init__(self, **kwargs):
        super(KeypointPTv1_26, self).__init__(blocks=[1, 1, 1, 1, 1], **kwargs)

@MODELS.register_module("KeypointPTv1-38")
class KeypointPTv1_38(KeypointPTv1):
    def __init__(self, **kwargs):
        super(KeypointPTv1_38, self).__init__(blocks=[1, 2, 2, 2, 2], **kwargs)

@MODELS.register_module("KeypointPTv1-50")
class KeypointPTv1_50(KeypointPTv1):
    def __init__(self, **kwargs):
        super(KeypointPTv1_50, self).__init__(blocks=[1, 2, 3, 5, 2], **kwargs)