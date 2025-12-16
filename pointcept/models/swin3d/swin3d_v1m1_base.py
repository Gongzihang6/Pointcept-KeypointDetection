"""
代码功能：Swin3D U-Net 主干网络实现 (基于 MinkowskiEngine 和 Pointcept 框架)
主要作用：
    1. 实现 3D 点云的稀疏体素化 (Voxelization)
    2. 使用 Swin Transformer Block (带移动窗口) 提取多尺度特征
    3. 通过 Encoder-Decoder 结构完成 3D 场景的语义分割
    4. 融合了 cRSE (上下文相对信号编码) 处理点云的不规则几何与外观信号

核心模块对应关系：
    - self.stem_layer  <--> 文档中的 "Initial Feature Embedding"
    - self.layers      <--> 文档中的 5 个 Stage (Encoder)
    - self.downsample  <--> 文档中的 "Downsample" (KNN Pooling)
    - self.upsamples   <--> 解码器部分 (Decoder)
    - self.cRSE        <--> 文档中的 "Contextual relative signal encoding"
"""

import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor
from timm.layers import trunc_normal_

from .mink_layers import MinkConvBNRelu, MinkResBlock
from .swin3d_layers import GridDownsample, GridKNNDownsample, BasicLayer, Upsample
from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch, batch2offset


@MODELS.register_module("Swin3D-v1m1")
class Swin3DUNet(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        base_grid_size,         # 基础体素大小 (如 2cm)
        depths,                 # 每个 Stage 中 Transformer Block 的数量
        channels,               # 每个 Stage 的特征通道数
        num_heads,              # 多头注意力的头数
        window_sizes,           # 窗口注意力的大小 (M x M x M)
        quant_size,             # 量化尺寸，用于 cRSE 查表
        drop_path_rate=0.2,     
        up_k=3,                 # 上采样时的 k 近邻数
        num_layers=5,           # 总层数 (Stage 数量)
        stem_transformer=True,
        down_stride=2,          # 下采样步长
        upsample="linear",
        knn_down=True,          # 是否使用 KNN 下采样
        cRSE="XYZ_RGB",         # 指定 cRSE 编码的信号类型
        fp16_mode=0,
    ):
        super().__init__()
        # 计算 stochastic depth (随机深度) 的衰减率，用于正则化
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # 1. 下采样策略选择
        # 对应文档 "Downsample" 章节：Swin3D 推荐使用 KNNPooling 处理稀疏数据
        if knn_down:    # 最近邻下采样或者网格下采样
            downsample = GridKNNDownsample
        else:
            downsample = GridDownsample

        self.cRSE = cRSE    # 使用坐标XYZ和颜色RGB进行上下文相对信号编码

        # 2. Initial Feature Embedding (Stem Layer)
        # 对应文档 "Initial Feature Embedding" 章节
        # 使用 3x3x3 稀疏卷积将原始特征投影到 C1 维度
        if stem_transformer:    # Initial Feature Embedding
            self.stem_layer = MinkConvBNRelu(   # 使用MinkowskiEngine实现稀疏卷积
                in_channels=in_channels,
                out_channels=channels[0],
                kernel_size=3,
                stride=1,
            )
            self.layer_start = 0
        else:
            self.stem_layer = nn.Sequential(    # # 备选的 ResNet 风格 Stem
                MinkConvBNRelu(
                    in_channels=in_channels,
                    out_channels=channels[0],
                    kernel_size=3,
                    stride=1,
                ),
                MinkResBlock(in_channels=channels[0], out_channels=channels[0]),
            )
            self.downsample = downsample(
                channels[0], channels[1], kernel_size=down_stride, stride=down_stride
            )
            self.layer_start = 1

        # 3.构建 Encoder Layers (Stages)
        # 对应 "Swin3D 的整体架构分为 5 个阶段"
        self.layers = nn.ModuleList(
            [
                BasicLayer(
                    dim=channels[i],
                    depth=depths[i],
                    num_heads=num_heads[i],
                    window_size=window_sizes[i],
                    quant_size=quant_size,
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],      # 传入当前层对应的 drop path rate
                    downsample=downsample if i < num_layers - 1 else None,      # 最后一个 Stage 不需要下采样，其他都需要
                    down_stride=down_stride if i == 0 else 2,
                    out_channels=channels[i + 1] if i < num_layers - 1 else None,
                    cRSE=cRSE,    # 每个层都包含 cRSE 模块                                          
                    fp16_mode=fp16_mode,
                )
                for i in range(self.layer_start, num_layers)
            ]
        )
        # 4.构建 Decoder (Upsample Layers)
        # 负责将深层特征恢复分辨率，并与浅层特征融合
        if "attn" in upsample:
            up_attn = True
        else:
            up_attn = False

        self.upsamples = nn.ModuleList(
            [
                Upsample(
                    channels[i],
                    channels[i - 1],
                    num_heads[i - 1],
                    window_sizes[i - 1],
                    quant_size,
                    attn=up_attn,
                    up_k=up_k,
                    cRSE=cRSE,
                    fp16_mode=fp16_mode,
                )
                # 倒序构建，从深层向浅层
                for i in range(num_layers - 1, 0, -1)
            ]
        )

        # 5. 分类头 (Head)
        # 简单的 MLP 输出最终的类别预测
        self.classifier = nn.Sequential(
            nn.Linear(channels[0], channels[0]),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(channels[0], num_classes),
        )
        self.num_classes = num_classes
        self.base_grid_size = base_grid_size
        self.init_weights()

    def forward(self, data_dict):
        # 提取输入数据：网格坐标、特征、原始坐标等
        grid_coord = data_dict["grid_coord"]        
        feat = data_dict["feat"]
        coord_feat = data_dict["coord_feat"]    # 用于 cRSE 的辅助特征 (如 RGB)
        coord = data_dict["coord"]
        offset = data_dict["offset"]
        batch = offset2batch(offset)

        # 1. Voxelization (体素化)
        # 对应文档 "Voxelization" 章节
        # MinkowskiEngine 会根据 coordinates 自动进行稀疏哈希和平均化 (UNWEIGHTED_AVERAGE)
        # 文档提到：最细网格是 2cm (base_grid_size)，并保留原始信号 sv
        in_field = ME.TensorField(
            features=torch.cat(
                [
                    batch.unsqueeze(-1),            # Batch 索引
                    coord / self.base_grid_size,    # 归一化坐标 (相对体素位置)
                    coord_feat / 1.001,             # 归一化颜色等特征
                    feat,                           # 原始点云特征
                ],
                dim=1,
            ),
            # 坐标量化：将浮点坐标转换为整数体素坐标
            coordinates=torch.cat([batch.unsqueeze(-1).int(), grid_coord.int()], dim=1),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=feat.device,
        )


        # 将 TensorField 转换为稀疏张量 (SparseTensor)
        sp = in_field.sparse()

        # 分离特征：一部分用于 cRSE 计算 (coords_sp)，一部分用于网络主干 (sp)
        # coords_sp 包含：Batch, 相对坐标, 颜色等，这些正是 cRSE 计算 Delta S_ij 需要的
        coords_sp = SparseTensor(
            features=sp.F[:, : coord_feat.shape[-1] + 4],
            coordinate_map_key=sp.coordinate_map_key,
            coordinate_manager=sp.coordinate_manager,
        )
        sp = SparseTensor(
            features=sp.F[:, coord_feat.shape[-1] + 4 :],
            coordinate_map_key=sp.coordinate_map_key,
            coordinate_manager=sp.coordinate_manager,
        )
        # # =============== [新增调试代码] ===============
        # print("\n" + "="*30)
        # print(f"DEBUG: coord_feat dim = {coord_feat.shape[-1]}")
        # print(f"DEBUG: feat (input to model) shape = {sp.F.shape}")
        # print(f"DEBUG: Model expects in_channels = {self.stem_layer.conv.in_channels}")
        # print("="*30 + "\n")
        # # ============================================

        # 用于存储 Skip Connections 的栈
        sp_stack = []
        coords_sp_stack = []

        # 2. Initial Embedding (Stem)
        sp = self.stem_layer(sp)
        # 处理特殊的 Stage 0 情况
        if self.layer_start > 0:
            sp_stack.append(sp)
            coords_sp_stack.append(coords_sp)
            sp, coords_sp = self.downsample(sp, coords_sp)

        # 3. Encoder Forward (逐层下采样)
        for i, layer in enumerate(self.layers):
            # 保存当前层的坐标信息用于 cRSE
            coords_sp_stack.append(coords_sp)

            # layer 内部执行：Attention -> FeedForward -> Downsample
            # 特征 下采样特征 下采样坐标
            sp, sp_down, coords_sp = layer(sp, coords_sp)

            # 保存特征用于 Decoder 的 Skip Connection
            sp_stack.append(sp)
            # 确保坐标和特征的 batch index 对齐
            assert (coords_sp.C == sp_down.C).all()
            sp = sp_down

        # 4. Decoder Forward (逐层上采样)
        sp = sp_stack.pop()                 # 取出最深层特征
        coords_sp = coords_sp_stack.pop()   # 取出最深层坐标
        for i, upsample in enumerate(self.upsamples):
            sp_i = sp_stack.pop()                           # 取出上一层的 Skip Connection 特征
            coords_sp_i = coords_sp_stack.pop()             # 取出对应的坐标

            # 执行上采样和特征融合
            sp = upsample(sp, coords_sp, sp_i, coords_sp_i)
            coords_sp = coords_sp_i
            
        # 5. Classifier
        # slice(in_field) 确保输出特征映射回原始输入点的顺序
        output = self.classifier(sp.slice(in_field).F)
        return output

    def init_weights(self):
        """Initialize the weights in backbone."""

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)
