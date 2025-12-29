import torch
import spconv.pytorch as spconv

try:
    import ocnn
except ImportError:
    ocnn = None
from addict import Dict
from typing import List

from pointcept.models.utils.serialization import encode
from pointcept.models.utils import (
    offset2batch,
    batch2offset,
    offset2bincount,
    bincount2offset,
)


class Point(Dict):
    """
    Pointcept 的 Point 结构

    Pointcept 中的 Point（点云）是一个字典，包含批处理点云的各种属性。
    具有以下名称的属性具有特定定义，如下所示：

    - "coord": 点云的原始坐标；
    - "grid_coord": 特定网格大小的网格坐标（与 GridSampling 相关）；
    Point 还支持以下可选属性：
    - "offset": 如果不存在，初始化为批大小为 1；
    - "batch": 如果不存在，初始化为批大小为 1；
    - "feat": 点云的特征，模型的默认输入；
    - "grid_size": 点云的网格大小（与 GridSampling 相关）；
    (与序列化相关)
    - "serialized_depth": 序列化的深度，2 ** depth * grid_size 描述点云范围的最大值；
    - "serialized_code": 序列化代码列表；
    - "serialized_order": 由代码确定的序列化顺序列表；
    - "serialized_inverse": 由代码确定的反向映射列表；
    (与稀疏化相关：SpConv)
    - "sparse_shape": 稀疏卷积张量的稀疏形状；
    - "sparse_conv_feat": 使用 Point 提供的信息初始化的 SparseConvTensor；
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 如果 "offset" 或 "batch" 中的一个不存在，则通过现有的一个生成
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        """
        Point Cloud Serialization

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]
        """
        self["order"] = order
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            # 如果您不想在数据增强中操作 GridSampling，
            # 请在您的管道中添加以下增强：
            # dict(type="Copy", keys_dict={"grid_size": 0.01})，
            # （根据您的需要调整 `grid_size`）
            assert {"grid_size", "coord"}.issubset(self.keys())

            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:
            # 自适应测量序列化立方体的深度（长度 = 2 ^ depth）
            depth = int(self.grid_coord.max() + 1).bit_length()
        self["serialized_depth"] = depth
        # 序列化代码的最大位长度为 63（int64）
        assert depth * 3 + len(self.offset).bit_length() <= 63
        # 我们遵循 OCNN，为点位置设置深度限制为 16（48位）。
        # 尽管深度限制为小于 16，我们可以使用网格大小为 0.01 米编码 655.36^3 (2^16 * 0.01) 米^3
        # 的立方体。我们认为这对于当前阶段来说已经足够。
        # 我们可以通过优化 z-order 编码函数来解除限制。
        assert depth <= 16

        # 序列化代码按照以下结构排列：
        # [Order1 ([n])，
        #  Order2 ([n])，
        #   ...
        #  OrderN ([n])] (k, n)
        code = [
            encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order
        ]
        code = torch.stack(code)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse

    def sparsify(self, pad=96):
        """
        点云稀疏化

        点云是稀疏的，这里使用 "sparsify" 特指为 SpConv 准备 "spconv.SparseConvTensor"。

        依赖于 ["grid_coord" 或 "coord" + "grid_size", "batch", "feat"]

        pad: 为稀疏形状填充的填充值。
        """
        assert {"feat", "batch"}.issubset(self.keys())
        if "grid_coord" not in self.keys():
            # 如果您不想在数据增强中操作 GridSampling，
            # 请在您的管道中添加以下增强：
            # dict(type="Copy", keys_dict={"grid_size": 0.01})，
            # （根据您的需要调整 `grid_size`）
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()
        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(
                torch.max(self.grid_coord, dim=0).values, pad
            ).tolist()
        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat(
                [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].tolist() + 1,
        )
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat

    def octreelization(self, depth=None, full_depth=None):
        """
        Point Cloud Octreelization

        Generate octree with OCNN
        relay on ["grid_coord", "batch", "feat"]
        """
        assert (
            ocnn is not None
        ), "Please follow https://github.com/octree-nn/ocnn-pytorch install ocnn."
        assert {"feat", "batch"}.issubset(self.keys())
        # 加 1 使网格空间支持移位顺序
        if "grid_coord" not in self.keys():
            # 如果您不想在数据增强中操作 GridSampling，
            # 请在您的管道中添加以下增强：
            # dict(type="Copy", keys_dict={"grid_size": 0.01})，
            # （根据您的需要调整 `grid_size`）
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()
        if depth is None:
            if "depth" in self.keys():
                depth = self.depth
            else:
                depth = int(self.grid_coord.max() + 1).bit_length()
        if full_depth is None:
            full_depth = 1
        self["depth"] = depth
        assert depth <= 16  # OCNN 中的最大值

        # [0, 2**depth] -> [0, 2] -> [-1, 1]
        coord = self.grid_coord / 2 ** (self.depth - 1) - 1.0
        point = ocnn.octree.Points(
            points=coord,
            features=self.feat,
            batch_id=self.batch.unsqueeze(-1),
            batch_size=self.batch[-1] + 1,
        )
        octree = ocnn.octree.Octree(
            depth=depth,
            full_depth=full_depth,
            batch_size=self.batch[-1] + 1,
            device=coord.device,
        )
        octree.build_octree(point)
        octree.construct_all_neigh()

        query_pts = torch.cat([self.grid_coord, point.batch_id], dim=1).contiguous()
        inverse = octree.search_xyzb(query_pts, depth, True)
        assert torch.sum(inverse < 0) == 0  # 所有映射都应该是有效的
        inverse_ = torch.unique(inverse)
        order = torch.zeros_like(inverse_).scatter_(
            dim=0,
            index=inverse,
            src=torch.arange(0, inverse.shape[0], device=inverse.device),
        )
        self["octree"] = octree
        self["octree_order"] = order
        self["octree_inverse"] = inverse
