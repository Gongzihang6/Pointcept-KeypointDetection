# pointcept/models/keypoint_octformer.py

import torch
import torch.nn as nn
import torch_scatter
from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch

try:
    import ocnn
    from ocnn.octree import Octree, Points
except ImportError:
    # 如果这里报错，说明您环境里没有安装 ocnn
    raise ImportError(
        "KeypointOctFormer 依赖于 ocnn 库，但在您的环境中未找到。\n"
        "请参考 Pointcept 文档或运行 `pip install ocnn` (需根据 CUDA 版本选择安装方式) 进行安装。"
    )
# 导入 OctFormer 的组件
# 注意：这里假设 pointcept/models/octformer/octformer_v1m1_base.py 文件存在
from pointcept.models.octformer.octformer_v1m1_base import (
    OctreeT, 
    PatchEmbed, 
    OctFormerStage, 
    Downsample, 
    OctFormerDecoder
)
@MODELS.register_module("KeypointOctFormer")
class KeypointOctFormer(nn.Module):
    def __init__(self,
                 in_channels=4,        # 您的数据: 3法向量 + 1曲率
                 num_keypoints=6,      # 关键点数量
                 hidden_dim=256,       # 回归头隐藏层维度
                 # --- OctFormer 骨干参数 (参照 semseg-octformer-v1m1-0-base.py) ---
                 fpn_channels=168,
                 channels=(96, 192, 384, 384),
                 num_blocks=(2, 2, 18, 2),
                 num_heads=(6, 12, 24, 24),
                 patch_size=26,
                 stem_down=2,
                 head_up=2,
                 dilation=4,
                 drop_path=0.5,
                 nempty=True,
                 octree_scale_factor=10.24, # 缩放因子，Octree构建需要
                 octree_depth=11,
                 octree_full_depth=2,
                 **kwargs):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.octree_scale_factor = octree_scale_factor
        self.octree_depth = octree_depth
        self.octree_full_depth = octree_full_depth
        self.stem_down = stem_down
        self.num_stages = len(num_blocks)
        self.patch_size = patch_size
        self.dilation = dilation
        self.nempty = nempty

        # 1. 构建骨干 (OctFormer Backbone)
        drop_ratio = torch.linspace(0, drop_path, sum(num_blocks)).tolist()
        
        # Patch Embedding
        self.patch_embed = PatchEmbed(in_channels, channels[0], stem_down, nempty)
        
        # Encoder Layers
        self.layers = torch.nn.ModuleList([
            OctFormerStage(
                dim=channels[i],
                num_heads=num_heads[i],
                patch_size=patch_size,
                drop_path=drop_ratio[sum(num_blocks[:i]) : sum(num_blocks[: i + 1])],
                dilation=dilation,
                nempty=nempty,
                num_blocks=num_blocks[i],
            )
            for i in range(self.num_stages)
        ])
        
        # Downsamples
        self.downsamples = torch.nn.ModuleList([
            Downsample(channels[i], channels[i + 1], kernel_size=[2], nempty=nempty)
            for i in range(self.num_stages - 1)
        ])
        
        # Decoder (可选，为了获得全分辨率特征，这里保留)
        self.decoder = OctFormerDecoder(
            channels=channels, fpn_channel=fpn_channels, nempty=nempty, head_up=head_up
        )

        # [新增] 插值模块：负责将八叉树特征还原回原始点云
        self.interp = ocnn.nn.OctreeInterp("nearest", nempty)


        # 2. 回归头 (Regression Head)
        # 输入维度为 decoder 的输出维度 (fpn_channels)
        output_dim = num_keypoints * 3
        self.reg_head = nn.Sequential(
            nn.Linear(fpn_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Loss
        self.criterion = nn.MSELoss()

    def forward(self, data_dict):
        coord = data_dict["coord"]
        feat = data_dict["feat"] 
        offset = data_dict["offset"]
        
        # === 数据预处理 ===
        if feat.shape[1] >= 3:
            normal = feat[:, :3]
        else:
            normal = torch.zeros_like(coord)
            
        # 1. 构建 Octree
        batch = offset2batch(offset)
        point = Points(
            points=coord / self.octree_scale_factor,
            normals=normal, 
            features=feat,
            batch_id=batch.unsqueeze(-1),
            batch_size=len(offset),
        )
        octree = ocnn.octree.Octree(
            depth=self.octree_depth,
            full_depth=self.octree_full_depth,
            batch_size=len(offset),
            device=coord.device,
        )
        octree.build_octree(point)
        octree.construct_all_neigh()

        # 2. Backbone Forward
        current_feat = self.patch_embed(octree.features[octree.depth], octree, octree.depth)
        depth = octree.depth - self.stem_down
        
        octree_transformer = OctreeT(
            octree,
            self.patch_size,
            self.dilation,
            self.nempty,
            max_depth=depth,
            start_depth=depth - self.num_stages + 1,
        )
        
        features = {}
        for i in range(self.num_stages):
            depth_i = depth - i
            current_feat = self.layers[i](current_feat, octree_transformer, depth_i)
            features[depth_i] = current_feat
            if i < self.num_stages - 1:
                current_feat = self.downsamples[i](current_feat, octree_transformer, depth_i)
        
        # 3. Decoder
        point_features = self.decoder(features, octree)
        
        # 插值回原始点云
        query_pts = torch.cat([point.points, point.batch_id], dim=1).contiguous()
        point_features = self.interp(point_features, octree, octree.depth, query_pts)
        
        # 4. Global Pooling & Regression
        global_feat = torch_scatter.scatter_mean(point_features, batch, dim=0) 
        
        pred_flat = self.reg_head(global_feat) 
        pred = pred_flat.view(-1, self.num_keypoints, 3) 

        # 5. Loss 与 指标计算
        result_dict = {}
        if self.training:
            if "target" in data_dict:
                target = data_dict["target"]
                
                # Loss 计算：处理形状不匹配
                if pred.shape != target.shape:
                    pred_for_loss = pred.view(-1, 3)
                else:
                    pred_for_loss = pred

                loss = self.criterion(pred_for_loss, target)
                result_dict["loss"] = loss
                
                # === 指标监控 (真实物理尺度) ===
                with torch.no_grad():
                    k = self.num_keypoints
                    target_metric = target.view(-1, k, 3)
                    pred_metric = pred.view(-1, k, 3)

                    # 1. 计算欧氏距离 (B, K)
                    dist = torch.norm(pred_metric - target_metric, p=2, dim=-1)
                    
                    # 2. 逆归一化还原真实尺度
                    if "scale" in data_dict:
                        scale = data_dict["scale"]
                        if scale.ndim == 1:
                            scale = scale.view(-1, 1)
                        dist = dist * scale
                    
                    # 3. [原有] 记录所有点的平均距离
                    result_dict["train/mean_dist"] = dist.mean()
                    
                    # 4. [新增] 记录每一个关键点的平均距离
                    # dist shape: (Batch, Num_Keypoints) -> mean(dim=0) -> (Num_Keypoints,)
                    kp_dist_mean = dist.mean(dim=0)
                    for i in range(k):
                        # 这里的 key 会自动被 Pointcept 的 Logger 捕获并打印/记录
                        result_dict[f"train/kp{i}_dist"] = kp_dist_mean[i]

            return result_dict
        else:
            result_dict["pred"] = pred
            return result_dict
