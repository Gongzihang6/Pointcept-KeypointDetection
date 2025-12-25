"""
Point Transformer V1 for Semantic Segmentation

Might be a bit different from the original paper

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import einops
import pointops

from pointcept.models.builder import MODELS
from .utils import LayerNorm1d


class PointTransformerLayer(nn.Module):
    """
    point transformer层
    “体会基于MLP的向量自注意力机制和普通的自注意力机制的差异”
    根据论文中fig 2结构，本质上就是对输入点云进行如下操作：
    1、对输入点云的特征x进行线性变换，得到查询点、键点、值点的特征表示；
    2、对键点和值点进行knn_and_group操作，得到每个查询点的nsample个最近邻键点的特征表示；
                 Input: (p, x, o)
                        |
      +-----------------+-----------------+
      |                 |                 |
 [linear_q]        [linear_k]        [linear_v]
      |                 |                 |
     x_q               x_k               x_v
      |                 |                 |
      |          [ KNN & Grouping ] <-----+--- (uses p, o)
      |           /     |             \
      |    (neighbors) (neighbors)  (relative p)
      |      x_k       x_v             p_r
      |       |         |               |
      |       |         |          [linear_p]
      |       |         |               |
      |       |         |               v (pos encoding)
      |       |         |           +---+---+
      |       |         |           |       |
      v       v         |           v       v
   (x_k - x_q + p_r)    |        (x_v  +  p_r)
          |             |               |
       (r_qk)           |               |
          |             |               |
      [linear_w]        |               |
          |             |               |
      [Softmax]         |               |
          |             |               |
        (w)             |               |
          \             |               /
           \-----------(⊗)-------------/
                        |
                  [ einsum ] (Weighted Sum)
                        |
                        v
                      Output: (p, x, o)
    """
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            LayerNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_planes),
        )
        self.linear_w = nn.Sequential(
            LayerNorm1d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes, out_planes // share_planes),
            LayerNorm1d(out_planes // share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // share_planes, out_planes // share_planes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        x_k, idx = pointops.knn_query_and_group(    # 进行KNN查询最近的nsample个点，因为with_xyz设置为true，返回的x_k中的前3维是坐标差，然后C维是特征维度
            x_k, p, o, new_xyz=p, new_offset=o, nsample=self.nsample, with_xyz=True
        )
        x_v, _ = pointops.knn_query_and_group(
            x_v,
            p,
            o,
            new_xyz=p,
            new_offset=o,
            idx=idx,
            nsample=self.nsample,
            with_xyz=False,
        )
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        p_r = self.linear_p(p_r)    # 使用p和p的邻居的坐标差进行位置编码
        r_qk = (
            x_k     # n k c
            - x_q.unsqueeze(1)  # n 1 c 在中间邻居点数量维度上增加一维，自动进行广播，计算中心点与每个邻居点之间的特征差异
            + einops.reduce(
                p_r, "n ns (i j) -> n ns j", reduction="sum", j=self.mid_planes     # n k c
            )
        )
        w = self.linear_w(r_qk)  # (n, nsample, c/s)
        w = self.softmax(w)
        x = torch.einsum(
            "n t s i, n t i -> n s i",
            einops.rearrange(x_v + p_r, "n ns (s i) -> n ns s i", s=self.share_planes),     # n k s c/s
            w,  # n k c/s 自动广播 到 n k s c/s ，相当于把特征分为s个部分，每个部分的权重不同，组内权重共享
        )
        x = einops.rearrange(x, "n s i -> n (s i)")     # n c
        return x


class TransitionDown(nn.Module):
    """
    下采样层，用于将输入的点云特征进行下采样
    input: pxo: (p, x, o)
    output: (n, x, o)

    Input: p(N,3), x(N,C), o(B)
        |       |
        |       v
        |   [ Check Stride > 1 ]
        |       |
        +-------+------------------------+
        |                                |
        v (FPS)                          v (KNN)
    [ Farthest Point Sample ]        [ Grouping ]
        |                    >------->  |
        v (indices)          |           v (Local Region)
        (idx)                |  Feature: (M, K, C+3)  最远点采样获取M个特征点，每个特征点knn查询获取K个邻居点
        |                    |           | (C features + 3 rel pos)
        v                    |           v
    [ New Coordinates ]      |        [ Linear ]
        n_p (M, 3) ----------^           |
        |                           [ BN + ReLU ]
        |                                |
        |                           Feature: (M, K, C_out)
        |                                |
        |                           [ Max Pool ] (on K dim)
        |                                |
        v                                v
        n_p (M, 3)                     x_new (M, C_out)
        |                                |
        +---------------+----------------+
                        |
                Output: [n_p, x_new, n_o]   返回下采样后的点云坐标n_p，特征x_new，下采样后的点云数量n_o

    """
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride   # 计算下采样后的点云数量
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)

            # [核心操作 1] 最远点采样 (FPS)
            # 从原始点云 p 中，根据 n_o 规定的数量，采样出分布最远的关键点索引 idx
            # idx 形状为 (m)，m 是下采样后的总点数，下采样后的总点数由stride决定
            idx = pointops.farthest_point_sampling(p, o, n_o)  # (m)
            # 获取采样后的新坐标 n_p，形状 (m, 3)
            n_p = p[idx.long(), :]  # (m, 3)

            # [核心操作 2] kNN 查询与分组
            # 对于每个新点 n_p，在老点 p 中找到最近的 nsample 个邻居
            # 并将邻居的特征聚集起来，knn查询的邻域点数量由nsample决定
            x, _ = pointops.knn_query_and_group(
                x,                      # 输入特征
                p,                      # 输入坐标
                offset=o,               # 输入 offset
                new_xyz=n_p,            # 中心点坐标 (新的点)
                new_offset=n_o,         # 中心点 offset
                nsample=self.nsample,   # kNN 的 k (如 16)
                with_xyz=True,          # 关键参数：True 表示返回的特征中包含相对坐标 (p_neighbor - p_center)
            )   # 此时 x 的形状通常为 (m, nsample, 3 + c)
            x = self.relu(
                self.bn(self.linear(x).transpose(1, 2).contiguous())
            )  # (m, c_out, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c_out)
            p, o = n_p, n_o     # m 3  m
        else:   # 如果步长stride为1，不进行下采样，直接进行线性变换
            x = self.relu(self.bn(self.linear(x)))  # (n, c_out)
        return [p, x, o]


class TransitionUp(nn.Module):
    """
    上采样层，用于将输入的点云特征进行上采样
    input: pxo1: (p, x, o)
    output: (n, x, o)

    
    Structure (Mode 1: Upsampling / Interpolation):
          Target (Fine)                Source (Coarse)
          (p1, x1, o1)                  (p2, x2, o2)
               |                             |
               |                       [ Linear 2 ]
               |                             |
               |                   [ Point Interpolation ]
               |                (Map features from p2 to p1)
               |                             |
               v                             |
          [ Linear 1 ]                       |
               |                             |
               v                             v
               +------------> ( + ) <--------+
                                |
                             Output

    Structure (Mode 2: Global Aggregation - if pxo2 is None):
                  Input: x (Local Features)
                            |
                +-----------+-----------+
                |                       |
          (Local Feats)          (Global Mean)
                |                 [ Linear 2 ]
                |                 [  Repeat  ]
                v                       v
                +--------->( Concat )<--+
                               |
                          [ Linear 1 ]
                               |
                             Output
    """
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(
                nn.Linear(2 * in_planes, in_planes),
                nn.BatchNorm1d(in_planes),
                nn.ReLU(inplace=True),
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True)
            )
        else:
            self.linear1 = nn.Sequential(
                nn.Linear(out_planes, out_planes),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True),
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, out_planes),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True),
            )

    def forward(self, pxo1, pxo2=None):
        # === 模式 2：全局特征聚合 (当没有提供第二个输入 pxo2 时) ===
        # 这通常发生在 Encoder 刚结束，准备进入 Decoder 的最深层 (Stage 5)
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            # 遍历 Batch 中的每一个样本（点云）
            for i in range(o.shape[0]):
                # 计算当前样本在 packed Tensor 中的起始索引 s_i 和结束索引 e_i，以及点数 cnt
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
                # 取出当前样本的所有点特征
                x_b = x[s_i:e_i, :]
                # [核心操作]：
                # 1. x_b.sum(0, True) / cnt: 计算当前样本所有点的平均值（Global Mean Pooling）
                # 2. self.linear2(...): 对全局特征进行变换
                # 3. .repeat(cnt, 1): 将全局特征复制 cnt 次，使其与点数对齐
                # 4. torch.cat((x_b, ...), 1): 将 原始局部特征 与 全局特征 拼接
                x_b = torch.cat(
                    (x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1
                )
                x_tmp.append(x_b)   # (cnt,c)->(c)->(c_new)->(cnt,c_new)
            # 将处理完的 batch 重新拼回一个大 Tensor
            x = torch.cat(x_tmp, 0)
            # 通过 linear1 融合拼接后的特征
            x = self.linear1(x)
        else:
            # === 模式 1：上采样与融合 (标准用法) ===
            # pxo1: Target (Fine)，来自 Encoder 的高分辨率特征（跳跃连接），p1是密集点云，x1是低维特征
            p1, x1, o1 = pxo1
            # pxo2: Source (Coarse)，来自上一层 Decoder 的低分辨率特征，p2是稀疏点云，x2是高维特征
            p2, x2, o2 = pxo2
            
            # [核心操作]：上采样 (Interpolation) + 融合 (Summation)
            # 1. self.linear2(x2): 先变换成低维特征
            # 2. pointops.interpolation(...): 
            #    在 p1 (密集点云) 中寻找 p2 (稀疏点云) 的最近邻，利用距离权重将 x2 插值映射到 p1 的位置。
            #    这一步将深层的特征传播到了密集的点云坐标上
            # 3. self.linear1(x1): 低维特征也变化一下，提取特征
            # 4. + : 将 插值后的深层特征 与 浅层细节特征 相加融合
            x = self.linear1(x1) + pointops.interpolation(
                p2, p1, self.linear2(x2), o2, o1
            )   # 和PointNet++的找邻居一样，对密集点云中的每一个点，在稀疏点云中寻找最近的k个点，权重依然是根据距离反比获得，然后对稀疏点云（高维特征）进行加权求和
            # 不同之处在于，PointNet++将插值得到的高维特征和原来密集点云的低维特征直接在通道维度拼接，但是ptv1是先将高维特征降维（和密集点云中的低维特征维度一致），
            # 然后在密集点云中寻找稀疏点云的邻居，利用距离权重将高降维后的特征插值映射到密集点云的位置，最后将密集点云的新特征和旧特征直接逐元素相加
        return x


class Bottleneck(nn.Module):
    """
    瓶颈块 Bottleneck，包含 3 个线性层和 1 个点变换层

    point transformer v1 中的瓶颈块 Bottleneck，包含 3 个线性层和 1 个点变换层（PointTransformerLayer），特征维度不发生变化

    Input: (p, x, o)
            |
            +---< Identity (x) >----------------------+
            |                                         |
            |                                         |
        [Linear1] -> [BN1] -> [ReLU]                   |
            |                                         |
            v                                         |
        [Transformer] (inputs: p, x_feat, o)           |
            |                                         |
            v                                         |
        [BN2] -> [ReLU]                                |
            |                                         |
            v                                         |
        [Linear3] -> [BN3]                             |
            |                                         |
            v                                         |
            (+) <--------------------------------------+
            |
            [ReLU]
            |
        Output: (p, x, o)

    """
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(Bottleneck, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False) 
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x    # 复制x用于残差连接
        x = self.relu(self.bn1(self.linear1(x)))    # (n, c)
        x = self.relu(self.bn2(self.transformer([p, x, o])))    # (n, c)
        x = self.bn3(self.linear3(x))    # (n, c)
        x += identity
        x = self.relu(x)    # (n, c)
        return [p, x, o]


class PointTransformerSeg(nn.Module):
    def __init__(self, block, blocks, in_channels=6, num_classes=13):
        super().__init__()
        self.in_channels = in_channels
        self.in_planes, planes = in_channels, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        self.enc1 = self._make_enc(
            block,
            planes[0],
            blocks[0],
            share_planes,
            stride=stride[0],
            nsample=nsample[0],
        )  # N/1
        self.enc2 = self._make_enc(
            block,
            planes[1],
            blocks[1],
            share_planes,
            stride=stride[1],
            nsample=nsample[1],
        )  # N/4
        self.enc3 = self._make_enc(
            block,
            planes[2],
            blocks[2],
            share_planes,
            stride=stride[2],
            nsample=nsample[2],
        )  # N/16
        self.enc4 = self._make_enc(
            block,
            planes[3],
            blocks[3],
            share_planes,
            stride=stride[3],
            nsample=nsample[3],
        )  # N/64
        self.enc5 = self._make_enc(
            block,
            planes[4],
            blocks[4],
            share_planes,
            stride=stride[4],
            nsample=nsample[4],
        )  # N/256
        self.dec5 = self._make_dec(
            block, planes[4], 1, share_planes, nsample=nsample[4], is_head=True
        )  # transform p5
        self.dec4 = self._make_dec(
            block, planes[3], 1, share_planes, nsample=nsample[3]
        )  # fusion p5 and p4
        self.dec3 = self._make_dec(
            block, planes[2], 1, share_planes, nsample=nsample[2]
        )  # fusion p4 and p3
        self.dec2 = self._make_dec(
            block, planes[1], 1, share_planes, nsample=nsample[1]
        )  # fusion p3 and p2
        self.dec1 = self._make_dec(
            block, planes[0], 1, share_planes, nsample=nsample[0]
        )  # fusion p2 and p1
        self.cls = nn.Sequential(
            nn.Linear(planes[0], planes[0]),
            nn.BatchNorm1d(planes[0]),
            nn.ReLU(inplace=True),
            nn.Linear(planes[0], num_classes),
        )

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = [
            TransitionDown(self.in_planes, planes * block.expansion, stride, nsample)
        ]
        self.in_planes = planes * block.expansion
        for _ in range(blocks):
            layers.append(
                block(self.in_planes, self.in_planes, share_planes, nsample=nsample)
            )
        return nn.Sequential(*layers)

    def _make_dec(
        self, block, planes, blocks, share_planes=8, nsample=16, is_head=False
    ):
        layers = [
            TransitionUp(self.in_planes, None if is_head else planes * block.expansion)
        ]
        self.in_planes = planes * block.expansion
        for _ in range(blocks):
            layers.append(
                block(self.in_planes, self.in_planes, share_planes, nsample=nsample)
            )
        return nn.Sequential(*layers)

    def forward(self, data_dict):
        p0 = data_dict["coord"]
        x0 = data_dict["feat"]
        o0 = data_dict["offset"].int()
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        x = self.cls(x1)
        return x


@MODELS.register_module("PointTransformer-Seg26")
class PointTransformerSeg26(PointTransformerSeg):
    def __init__(self, **kwargs):
        super(PointTransformerSeg26, self).__init__(
            Bottleneck, [1, 1, 1, 1, 1], **kwargs
        )


@MODELS.register_module("PointTransformer-Seg38")
class PointTransformerSeg38(PointTransformerSeg):
    def __init__(self, **kwargs):
        super(PointTransformerSeg38, self).__init__(
            Bottleneck, [1, 2, 2, 2, 2], **kwargs
        )


@MODELS.register_module("PointTransformer-Seg50")
class PointTransformerSeg50(PointTransformerSeg):
    def __init__(self, **kwargs):
        super(PointTransformerSeg50, self).__init__(
            Bottleneck, [1, 2, 3, 5, 2], **kwargs
        )
