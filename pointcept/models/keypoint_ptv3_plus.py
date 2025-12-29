import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from functools import partial
from timm.layers import DropPath
from addict import Dict

from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential
from pointcept.models.point_prompt_training import PDNorm

# 导入基础 PTv3 组件
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    SerializedAttention,
    MLP,
    SerializedPooling,
    SerializedUnpooling,
    Embedding,
    RPE
)
from pointcept.models.keypoint_ptv3 import KeypointPTv3


class BlockPlus(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        cpe_kernel_size=3  # 新增参数: CPE 卷积核大小
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm
        self.cpe_kernel_size = cpe_kernel_size

        # [大核卷积位置编码 (Large Kernel xCPE)]
        # 使用更大的卷积核来扩大感受野，增强局部几何特征提取。
        # 对于奇数核大小，padding 通常设为 kernel_size // 2 以保持空间尺寸不变。
        # 虽然 SubMConv3d 的 padding 机制有些特殊，但显式设置 padding 有助于确保边界行为符合预期。
        padding = cpe_kernel_size // 2
        
        # [新代码] Bottleneck CPE 设计
        # 使用 1/4 的通道缩放比例，既节省参数又保留了大核感受野
        mid_channels = channels // 4
        # 确保中间通道至少为 16，避免在浅层过窄
        if mid_channels < 16:
            mid_channels = channels

        self.cpe = PointSequential(
            # 1. 降维 (1x1x1 卷积，相当于 Linear 但保持稀疏性)
            spconv.SubMConv3d(channels, mid_channels, kernel_size=1, bias=False),
            norm_layer(mid_channels),
            nn.ReLU(inplace=True),
            
            # 2. 空间特征提取 (5x5x5 大核卷积)
            # 注意：这里 groups=1 (默认)，因为 mid_channels 很小，所以参数量可控
            spconv.SubMConv3d(
                mid_channels,
                mid_channels,
                kernel_size=cpe_kernel_size,
                padding=padding,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            norm_layer(mid_channels),
            nn.ReLU(inplace=True),

            # 3. 升维 (1x1x1 卷积)
            spconv.SubMConv3d(mid_channels, channels, kernel_size=1, bias=False),
            
            # 4. 原始的 Linear 和 Norm (可选，为了保持结构一致性建议保留)
            # 这里的 Linear 会作用于 point.feat，帮助进一步融合
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        # 位置编码增强
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        
        # Attention 模块
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        # FFN (MLP) 模块
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        
        # 更新 sparse_conv_feat，确保与最新的 feat 同步
        # 注意：如果在外部进行了物理重排，这里的 replace_feature 依然有效，
        # 因为它只是替换了 tensor 的值，不改变 indices。
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


@MODELS.register_module("PT-v3m1-Plus")
class PointTransformerV3Plus(PointModule):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        cpe_kernel_size=5  # 默认使用 5x5x5 大核
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.enc_mode = enc_mode
        self.shuffle_orders = shuffle_orders
        self.cpe_kernel_size = cpe_kernel_size

        # 验证参数一致性
        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)

        # 初始化 Norm 层工厂函数
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        
        act_layer = nn.GELU

        # 输入嵌入层
        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # 构建 Encoder
        # 我们手动存储每个 Stage 的模块，以便在 forward 中手动控制循环，插入重序列化逻辑
        self.enc_stages = nn.ModuleList()
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        
        for s in range(self.num_stages):
            stage_blocks = PointSequential()
            
            # 下采样 Pooling (除 Stage 0 外)
            if s > 0:
                stage_blocks.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            
            # 构建该 Stage 的 Blocks
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            for i in range(enc_depths[s]):
                stage_blocks.add(
                    BlockPlus(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        # order_index=i % len(self.order),
                        order_index=0,
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        cpe_kernel_size=cpe_kernel_size # 传入大核尺寸
                    ),
                    name=f"block{i}",
                )
            self.enc_stages.append(stage_blocks)

        # 构建 Decoder
        if not self.enc_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        BlockPlus(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            # order_index=i % len(self.order),
                            order_index=0,
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                            cpe_kernel_size=cpe_kernel_size
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict):
        point = Point(data_dict)
        
        # 初始序列化 (可选，但通常用于 Embedding 之前的准备)
        # 我们会在后续循环中进行显式的重序列化
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        
        # 初始稀疏化 (Sparsify)
        # 这一步只需做一次，后续的重序列化只需更新索引，不需要重新体素化
        # 这会基于原始坐标创建 point.sparse_conv_feat
        point.sparsify()

        # Embedding 层
        point = self.embedding(point)
        
        # Encoder 循环：引入多视角重序列化 (Multi-View Re-serialization)
        for s, stage in enumerate(self.enc_stages):
            # stage 是一个 PointSequential，包含 Pooling (可选) 和 Blocks
            # 如果 s > 0，它包含 'down' (SerializedPooling) 和 'block0', 'block1'...
            
            # 我们需要先运行 Pooling (如果存在)
            if s > 0:
                # 提取 Blocks 列表
                blocks = []
                for name, module in stage.named_children():
                    if name == "down":
                        # 执行下采样 Pooling
                        point = module(point)
                        # Pooling 之后，point.coord 和 point.grid_coord 已经被更新为下采样后的坐标
                        # SerializedPooling 内部调用了 point.sparsify()，
                        # 所以 point.sparse_conv_feat 已经针对新的网格进行了更新
                    else:
                        blocks.append(module)
                
                # 现在我们处于当前 Stage 的 Blocks 计算之前
                # === [核心改进] 多视角重序列化 (Multi-View Re-serialization) ===
                # 目的：打破单一 Z-order 带来的 Patch 边界效应，通过坐标置换让不同维度的邻域在序列中紧凑排列。
                
                # 1. 坐标轴置换 (Coordinate Permutation)
                # 循环策略: XYZ -> YZX -> ZXY -> XYZ
                # s=1: YZX ([1, 2, 0]) - 让 Y 轴或 Z 轴邻域更紧凑
                # s=2: ZXY ([2, 0, 1]) - 让 Z 轴或 X 轴邻域更紧凑
                # s=3: XYZ ([0, 1, 2]) - 回归原始顺序
                perm_idx = s % 3
                
                if perm_idx == 1:
                    permutation = [1, 2, 0] # YZX
                elif perm_idx == 2:
                    permutation = [2, 0, 1] # ZXY
                else:
                    permutation = [0, 1, 2] # XYZ
                
                # 仅当非原始顺序时执行置换操作
                if perm_idx != 0:
                    # 原地修改 point.coord 和 point.grid_coord
                    # 这样后续的 serialization 就会基于置换后的坐标计算 Z-order code
                    point.coord = point.coord[:, permutation]
                    point.grid_coord = point.grid_coord[:, permutation]
                    
                    # 2. 重新序列化 (Re-serialization)
                    # 基于置换后的坐标，强制使用 "z" 顺序计算新的序列化 Code 和 Order
                    point.serialization(order="z") 
                    
                    # 3. 物理重排 (Physical Re-ordering)
                    # 为了让 Attention 能够利用序列的局部性，我们需要根据新的 Z-order 对数据进行物理排序。
                    # 注意：Point Transformer V3 依赖于物理上的紧凑排列来切分 Patch。
                    order = point.serialized_order[0] # 获取计算出的排序索引
                    
                    # 重排特征、坐标、Batch索引等关键属性
                    point.feat = point.feat[order]
                    point.coord = point.coord[order]
                    point.grid_coord = point.grid_coord[order]
                    point.batch = point.batch[order]
                    if "condition" in point:
                        point.condition = point.condition[order]
                    if "context" in point:
                        point.context = point.context[order]
                    
                    # 4. 更新 Sparse Conv 特征 (Update SparseConvTensor)
                    # CPE 模块依赖于 spconv，而 spconv 的 indices 必须与 point.feat 的物理顺序一致。
                    # 同时，因为我们置换了 grid_coord 的列 (x,y,z)，spconv 的 indices 也需要对应置换。
                    
                    sp = point.sparse_conv_feat
                    # 按照 Z-order 对 indices 进行行排序 (Row Sorting)
                    new_indices = sp.indices[order] 
                    
                    # 按照坐标置换对 indices 进行列置换 (Column Permutation)
                    # spconv indices 通常格式为 [batch_idx, x, y, z] (取决于 Pointcept 实现，通常 dim=1 开始是坐标)
                    # 在 point.sparsify() 中：indices = cat([batch, grid_coord], dim=1)
                    # 所以 indices[:, 1:] 对应 grid_coord (x, y, z)
                    # 我们需要对这部分也应用 permutation，以匹配被置换过的 grid_coord
                    new_indices[:, 1:] = new_indices[:, 1:][:, permutation]
                    
                    # 创建新的 SparseConvTensor，保持 spatial_shape 和 batch_size 不变
                    point.sparse_conv_feat = spconv.SparseConvTensor(
                        point.feat, 
                        new_indices, 
                        sp.spatial_shape, 
                        sp.batch_size
                    )
                    
                    # 5. 重置序列化信息 (Reset Serialized Info)
                    # 因为我们已经完成了物理排序，此时数据在内存中已经是 Z-order 排列了。
                    # 所以 serialized_order 应该重置为恒等映射 (0, 1, 2...)
                    N = len(order)
                    point.serialized_order = torch.arange(N, device=order.device).unsqueeze(0)
                    point.serialized_inverse = torch.arange(N, device=order.device).unsqueeze(0)
                    # Code 也需要按照排序后的顺序更新，以保持一致性
                    point.serialized_code = point.serialized_code[:, order]

                # 运行当前 Stage 的 Blocks
                for block in blocks:
                    point = block(point)

            else:
                # Stage 0: 初始阶段
                # 根据任务设定，Stage 0 (或称 Stage 1) 使用原始坐标 (x,y,z) 进行排序。
                # 这里的输入 point 已经在开头经过了 serialization 和 sparsify，满足要求。
                # 直接运行该 Stage 的所有模块。
                point = stage(point)

        # Decoder 部分
        if not self.enc_mode:
            point = self.dec(point)
            
        return point

@MODELS.register_module()
class KeypointPTv3Plus(KeypointPTv3):
    def __init__(self, backbone_conf, num_keypoints=6, hidden_dim=256):
        # 继承自 KeypointPTv3
        # 只要 backbone_conf 中的 type 是 "PT-v3m1-Plus"，build_model 就会实例化上面的 PointTransformerV3Plus
        super().__init__(backbone_conf, num_keypoints, hidden_dim)

    def forward(self, data_dict):
        # 1. 提取特征
        # 调用 PointTransformerV3Plus 骨干网络
        point_output = self.backbone(data_dict)
        feat = point_output.feat
        batch = point_output.batch
        
        # 2. 全局池化 (Global Pooling)
        # 将每个样本的点云特征聚合为全局特征向量
        # 注意：即使 PointTransformerV3Plus 内部对点进行了重排，
        # 只要 feat 和 batch 保持一一对应，scatter_mean 的结果就是正确的。
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
            # 这是用户关心的逻辑，显式地包含在这里以确保完整性。
            if self.training:
                with torch.no_grad():
                    k = self.num_keypoints
                    pred_metric = pred.view(-1, k, 3)
                    target_metric = target.view(-1, k, 3)

                    # 1. 计算归一化空间下的欧氏距离
                    dist = torch.norm(pred_metric - target_metric, p=2, dim=-1) # (B, K)

                    # 2. [核心步骤] 使用 scale 进行逆归一化
                    # 如果数据集中包含了缩放因子 scale，我们可以还原出真实的物理距离误差
                    if "scale" in data_dict:
                        scale = data_dict["scale"] # (B,) 或 (B, 1)
                        if scale.ndim == 1:
                            scale = scale.view(-1, 1) # 变成 (B, 1) 以便广播
                        
                        # 真实距离 = 归一化距离 * 缩放因子
                        dist = dist * scale
                    
                    # 3. 记录指标到日志
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
