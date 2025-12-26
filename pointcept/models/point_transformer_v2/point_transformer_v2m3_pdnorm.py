"""
Point Transformer V2M3

Enable Prompt-Driven Normalization for Point Prompt Training

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from copy import deepcopy
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr

import einops
from timm.layers import DropPath
import pointops

from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch, batch2offset


class PDBatchNorm(torch.nn.Module):
    def __init__(
        self,
        num_features,
        context_channels=256,
        eps=1e-3,
        momentum=0.01,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        decouple=True,
        adaptive=False,
        affine=True,
    ):
        super().__init__()
        self.conditions = conditions
        self.decouple = decouple
        self.adaptive = adaptive
        self.affine = affine
        if self.decouple:
            self.bns = nn.ModuleList(
                [
                    nn.BatchNorm1d(
                        num_features=num_features,
                        eps=eps,
                        momentum=momentum,
                        affine=affine,
                    )
                    for _ in conditions
                ]
            )
        else:
            self.bn = nn.BatchNorm1d(
                num_features=num_features, eps=eps, momentum=momentum, affine=affine
            )
        if self.adaptive:
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True)
            )

    def forward(self, feat, condition=None, context=None):
        if self.decouple:
            assert condition in self.conditions
            bn = self.bns[self.conditions.index(condition)]
        else:
            bn = self.bn
        feat = bn(feat)
        if self.adaptive:
            assert context is not None
            shift, scale = self.modulation(context).chunk(2, dim=1)
            feat = feat * (1.0 + scale) + shift
        return feat


class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return (
                self.norm(input.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError


class GroupedVectorAttention(nn.Module):
    def __init__(
        self,
        embed_channels,
        groups,
        attn_drop_rate=0.0,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
    ):
        super(GroupedVectorAttention, self).__init__()
        self.embed_channels = embed_channels
        self.groups = groups
        assert embed_channels % groups == 0
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias

        self.linear_q = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.linear_k = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )

        self.linear_v = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        if self.pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        if self.pe_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        self.weight_encoding = nn.Sequential(
            nn.Linear(embed_channels, groups),
            PointBatchNorm(groups),
            nn.ReLU(inplace=True),
            nn.Linear(groups, groups),
        )
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, feat, coord, reference_index):
        query, key, value = (       # 根据输入特征计算注意力计算需要的Q、K、V
            self.linear_q(feat),
            self.linear_k(feat),
            self.linear_v(feat),
        )
        key = pointops.grouping(reference_index, key, coord, with_xyz=True)         # 根据邻域索引获取注意力范围内所有点的K，K的前三维是坐标差
        value = pointops.grouping(reference_index, value, coord, with_xyz=False)    # 根据邻域索引获取注意力范围内所有点的V
        pos, key = key[:, :, 0:3], key[:, :, 3:]    # 从K中分离出坐标差和真正的K
        relation_qk = key - query.unsqueeze(1)      # 计算r(Q,K)，这里是差运算
        if self.pe_multiplier:  # 位置编码乘子法
            pem = self.linear_p_multiplier(pos) # 对坐标差进行线性变换
            relation_qk = relation_qk * pem     # r(Q,K) * pem
        if self.pe_bias:        # 位置编码偏置法
            peb = self.linear_p_bias(pos)
            relation_qk = relation_qk + peb
            value = value + peb

        weight = self.weight_encoding(relation_qk)
        weight = self.attn_drop(self.softmax(weight))

        mask = torch.sign(reference_index + 1)
        weight = torch.einsum("n s g, n s -> n s g", weight, mask)
        value = einops.rearrange(value, "n ns (g i) -> n ns g i", g=self.groups)
        feat = torch.einsum("n s g i, n s g -> n g i", value, weight)
        feat = einops.rearrange(feat, "n g i -> n (g i)")
        return feat


class Block(nn.Module):
    def __init__(
        self,
        embed_channels,
        groups,
        norm_fn=None,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super(Block, self).__init__()
        self.attn = GroupedVectorAttention(
            embed_channels=embed_channels,
            groups=groups,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
        )

        assert norm_fn is not None

        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = norm_fn(embed_channels)
        self.norm2 = norm_fn(embed_channels)
        self.norm3 = norm_fn(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, points, reference_index):
        coord, feat, offset, condition, context = points
        identity = feat
        feat = self.act(self.norm1(self.fc1(feat), condition, context))
        feat = (
            self.attn(feat, coord, reference_index)
            if not self.enable_checkpoint
            else checkpoint(self.attn, feat, coord, reference_index)
        )
        feat = self.act(self.norm2(feat, condition, context))
        feat = self.norm3(self.fc3(feat), condition, context)
        feat = identity + self.drop_path(feat)
        feat = self.act(feat)
        return [coord, feat, offset, condition, context]


class BlockSequence(nn.Module):
    def __init__(
        self,
        depth,
        embed_channels,
        groups,
        neighbours=16,
        norm_fn=None,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super(BlockSequence, self).__init__()

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        elif isinstance(drop_path_rate, float):
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        else:
            drop_path_rates = [0.0 for _ in range(depth)]

        self.neighbours = neighbours
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                embed_channels=embed_channels,
                groups=groups,
                norm_fn=norm_fn,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                enable_checkpoint=enable_checkpoint,
            )
            self.blocks.append(block)

    def forward(self, points):
        coord, feat, offset, condition, context = points
        # reference index query of neighbourhood attention
        # for windows attention, modify reference index query method
        reference_index, _ = pointops.knn_query(self.neighbours, coord, offset)
        for block in self.blocks:
            points = block(points, reference_index)
        return points


class GridPool(nn.Module):
    """
    Partition-based Pooling (Grid Pooling)
    """

    def __init__(self, in_channels, out_channels, grid_size, norm_fn, bias=False):
        super(GridPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        # 定义池化前的特征变换层：Linear -> Norm -> ReLU
        # 这对应论文公式(8)中的 U 变换：f_j * U
        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = norm_fn(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, start=None):
        # 1、数据解包与预处理
        # points 是一个列表/元组，包含点云的各种属性
        # coord: 坐标 (N, 3)
        # feat: 特征 (N, C)
        # offset: 偏移量 (B,)，表示每个点云在 batch 中的结束索引（例如 [1000, 2500] 表示第一个点云是 0~1000，第二个是 1000~2500）
        coord, feat, offset, condition, context = points

        # 将 offset 转换为 batch 索引向量
        # batch: (N,)，每个元素表示该点属于 batch 中的第几个样本（例如 [0,0,...,1,1,...]）
        batch = offset2batch(offset)
        # 特征变换：先降维/升维，再归一化和激活
        # 注意：这里的 self.norm 接收 condition 和 context，说明可能使用了条件归一化（Conditional BN）
        # 这一步对应论文中池化公式里的 MaxPool({f_j * U ...}) 中的 * U 部分
        feat = self.act(self.norm(self.fc(feat), condition, context))

        # 2、 计算网格对齐的原点 (Start Position)
        # 为了保证网格划分对平移不敏感（或者说相对每个场景独立划分），需要先找到每个场景的最小坐标作为“原点”。
        # 计算每个 batch 中点云的最小坐标 (Min Coordinate) 作为网格生成的基准点
        start = (
            segment_csr(
                coord,  # 输入坐标
                # 构建 CSR 指针：[0, count_0, count_0+count_1, ...]
                # 这行代码的作用是根据 batch 索引将 coord 分组
                torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
                reduce="min",   # 对每个 batch 内部求坐标最小值
            )
            if start is None
            else start
            # 此时 start 的形状是 (Batch_Size, 3)，存储了每个点云场景的左下角坐标
        )
        # 3、 网格化 
        # 这是最关键的一步，计算每个点属于哪个网格。
        # 调用 voxel_grid 计算体素/网格索引
        # pos=coord - start[batch]: 将坐标归一化到相对于各自场景原点的位置（相对坐标）
        # size=self.grid_size: 网格的物理尺寸
        # 返回的 cluster 是一个形状为 (N,) 的 Tensor，每个元素是一个整数 ID，代表该点属于哪个网格
        cluster = voxel_grid(
            pos=coord - start[batch], size=self.grid_size, batch=batch, start=0
        )

        # 4、整理索引 为了使用高效的 segment_csr 进行聚合，必须将属于同一个网格的点在内存中物理相邻（即排序）。
        # torch.unique 找出所有被占用的有效网格
        # unique: 有哪些网格 ID 存在 (M,)
        # cluster: (N,) 这里重新赋值了，变成 Inverse Indices，即每个点指向 unique 数组的下标 (0 ~ M-1)
        # counts: (M,) 每个网格里有多少个点
        unique, cluster, counts = torch.unique(
            cluster, sorted=True, return_inverse=True, return_counts=True
        )
        # 根据 cluster ID 对点进行排序
        # 排序后，属于同一个网格的点在 Tensor 中是连续排列的
        # sorted_cluster_indices: 排序后的索引，用于重排 coord 和 feat
        _, sorted_cluster_indices = torch.sort(cluster)

        # 构建 CSR (Compressed Sparse Row) 指针
        # idx_ptr: (M+1,)，指示了每个网格在排序后数组中的 [start, end) 位置
        # 例如: [0, 5, 8] 表示第一个网格对应索引 0~4，第二个对应 5~7
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])

        # 坐标池化：Mean Pooling (求平均)
        # 将同一个网格内的点坐标取平均，作为新点的坐标（重心）
        # 对应论文公式 (8): p'_i = MeanPool(...)
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean")

        # 特征池化：Max Pooling (求最大值)
        # 将同一个网格内的特征取最大值
        # 对应论文公式 (8): f'_i = MaxPool(...)
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")

        # 5、更新 batch 索引
        # 更新 batch 索引
        # 这里的逻辑是：取每个网格段的第一个点的 batch 索引（同一个网格内的点肯定属于同一个 batch）
        batch = batch[idx_ptr[:-1]]
        offset = batch2offset(batch)    # 将 batch 索引还原回 offset 格式，供下一层使用

        # 返回值：
        # 1. 新的点云数据列表 [新坐标, 新特征, 新offset, ...]
        # 2. cluster: 原始点到新网格的映射索引 (N,)。
        #    这个 cluster 非常重要，后续的 Unpooling (反池化) 需要用到它来做索引映射 (Map Unpooling)。
        return [coord, feat, offset, condition, context], cluster


class UnpoolWithSkip(nn.Module):
    """
    Map Unpooling with skip connection
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_fn,
        bias=True,
        skip=True,
        backend="map",
    ):
        super(UnpoolWithSkip, self).__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.skip = skip
        self.backend = backend
        assert self.backend in ["map", "interp"]

        self.proj_linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.proj_norm = norm_fn(out_channels)
        self.proj_act = nn.ReLU(inplace=True)

        self.proj_skip_linear = nn.Linear(skip_channels, out_channels, bias=bias)
        self.proj_skip_norm = norm_fn(out_channels)
        self.proj_skip_act = nn.ReLU(inplace=True)

    def forward(self, points, skip_points, cluster=None):
        coord, feat, offset, condition, context = points
        skip_coord, skip_feat, skip_offset, _, _ = skip_points
        feat = self.proj_act(self.proj_norm(self.proj_linear(feat), condition, context))
        if self.backend == "map" and cluster is not None:
            feat = feat[cluster]
        else:
            feat = pointops.interpolation(coord, skip_coord, feat, offset, skip_offset)
        if self.skip:
            feat = feat + self.proj_skip_act(
                self.proj_skip_norm(
                    self.proj_skip_linear(skip_feat), condition, context
                )
            )
        return [skip_coord, feat, skip_offset, condition, context]


class Encoder(nn.Module):
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        groups,
        norm_fn,
        grid_size=None,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=None,
        drop_path_rate=None,
        enable_checkpoint=False,
    ):
        super(Encoder, self).__init__()

        self.down = GridPool(
            in_channels=in_channels,
            out_channels=embed_channels,
            grid_size=grid_size,
            norm_fn=norm_fn,
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            norm_fn=norm_fn,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.0,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.0,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points):
        points, cluster = self.down(points)
        return self.blocks(points), cluster


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        embed_channels,
        groups,
        depth,
        norm_fn,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=None,
        drop_path_rate=None,
        enable_checkpoint=False,
        unpool_backend="map",
    ):
        super(Decoder, self).__init__()

        self.up = UnpoolWithSkip(
            in_channels=in_channels,
            out_channels=embed_channels,
            skip_channels=skip_channels,
            backend=unpool_backend,
            norm_fn=norm_fn,
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            norm_fn=norm_fn,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.0,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.0,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points, skip_points, cluster):
        points = self.up(points, skip_points, cluster)
        return self.blocks(points)


class GVAPatchEmbed(nn.Module):
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        groups,
        norm_fn,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super(GVAPatchEmbed, self).__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.proj_linear = nn.Linear(in_channels, embed_channels, bias=False)
        self.proj_norm = norm_fn(embed_channels)
        self.proj_act = nn.ReLU(inplace=True)
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            norm_fn=norm_fn,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points):
        coord, feat, offset, condition, context = points
        feat = self.proj_act(self.proj_norm(self.proj_linear(feat), condition, context))
        return self.blocks([coord, feat, offset, condition, context])


@MODELS.register_module("PT-v2m3")
class PointTransformerV2(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        patch_embed_depth=1,
        patch_embed_channels=48,
        patch_embed_groups=6,
        patch_embed_neighbours=8,
        enc_depths=(2, 2, 6, 2),
        enc_channels=(96, 192, 384, 512),
        enc_groups=(12, 24, 48, 64),
        enc_neighbours=(16, 16, 16, 16),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(48, 96, 192, 384),
        dec_groups=(6, 12, 24, 48),
        dec_neighbours=(16, 16, 16, 16),
        grid_sizes=(0.06, 0.12, 0.24, 0.48),
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0,
        enable_checkpoint=False,
        unpool_backend="map",
        context_channels=256,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        norm_decouple=True,
        norm_adaptive=True,
        norm_affine=False,
    ):
        super(PointTransformerV2, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = len(enc_depths)
        assert self.num_stages == len(dec_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(dec_channels)
        assert self.num_stages == len(enc_groups)
        assert self.num_stages == len(dec_groups)
        assert self.num_stages == len(enc_neighbours)
        assert self.num_stages == len(dec_neighbours)
        assert self.num_stages == len(grid_sizes)

        norm_fn = partial(
            PDBatchNorm,
            eps=1e-3,
            momentum=0.01,
            conditions=conditions,
            context_channels=context_channels,
            decouple=norm_decouple,
            adaptive=norm_adaptive,
            affine=norm_affine,
        )

        self.patch_embed = GVAPatchEmbed(
            in_channels=in_channels,
            embed_channels=patch_embed_channels,
            groups=patch_embed_groups,
            depth=patch_embed_depth,
            neighbours=patch_embed_neighbours,
            norm_fn=norm_fn,
            qkv_bias=attn_qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            enable_checkpoint=enable_checkpoint,
        )

        enc_dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))
        ]
        dec_dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_depths))
        ]
        enc_channels = [patch_embed_channels] + list(enc_channels)
        dec_channels = list(dec_channels) + [enc_channels[-1]]
        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(self.num_stages):
            enc = Encoder(
                depth=enc_depths[i],
                in_channels=enc_channels[i],
                embed_channels=enc_channels[i + 1],
                groups=enc_groups[i],
                grid_size=grid_sizes[i],
                neighbours=enc_neighbours[i],
                norm_fn=norm_fn,
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=enc_dp_rates[
                    sum(enc_depths[:i]) : sum(enc_depths[: i + 1])
                ],
                enable_checkpoint=enable_checkpoint,
            )
            dec = Decoder(
                depth=dec_depths[i],
                in_channels=dec_channels[i + 1],
                skip_channels=enc_channels[i],
                embed_channels=dec_channels[i],
                groups=dec_groups[i],
                neighbours=dec_neighbours[i],
                norm_fn=norm_fn,
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dec_dp_rates[
                    sum(dec_depths[:i]) : sum(dec_depths[: i + 1])
                ],
                enable_checkpoint=enable_checkpoint,
                unpool_backend=unpool_backend,
            )
            self.enc_stages.append(enc)
            self.dec_stages.append(dec)
        self.seg_head = (
            nn.Sequential(nn.Linear(dec_channels[0], num_classes))
            if num_classes > 0
            else nn.Identity()
        )

    def forward(self, data_dict):
        coord = data_dict["coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"].int()
        condition = data_dict["condition"][0]
        context = data_dict["context"] if "context" in data_dict.keys() else None

        # a batch of point cloud is a list of coord, feat and offset
        points = [coord, feat, offset, condition, context]
        points = self.patch_embed(points)
        skips = [[points]]
        for i in range(self.num_stages):
            points, cluster = self.enc_stages[i](points)
            skips[-1].append(cluster)  # record grid cluster of pooling
            skips.append([points])  # record points info of current stage

        points = skips.pop(-1)[0]  # unpooling points info in the last enc stage
        for i in reversed(range(self.num_stages)):
            skip_points, cluster = skips.pop(-1)
            points = self.dec_stages[i](points, skip_points, cluster)
        coord, feat, offset, _, _ = points
        seg_logits = self.seg_head(feat)
        return seg_logits
