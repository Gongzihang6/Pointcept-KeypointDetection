# Point Transformer



## PTv1

关键词：向量注意力（Vector Attention）、局部邻域（Local Neighborhood）、可学习位置编码；

### 背景

Self-attention 在 NLP 和 2D 图像领域取得了巨大成功，点云本质上是嵌入在度量空间中的几何，具有排列不变性，这与 transformer 的核心操作（自注意力，集合算子（Set Operator））高度契合。

**解决的问题：**

- 以往的点云网络（如 Pointnet++）使用最大池化来聚合信息，可能丢失局部细节；
- 以往的 3D 注意力尝试通常是全局的（计算量过大）或直接沿用标量点积注意力（Scalar Dot-production Attention），未能充分利用 3D 空间信息；

核心创新与网络结构：

1、**Point Transformer Layer (核心层)** 这是网络的基本构建块，其核心是 **向量自注意力 (Vector Self-Attention)** 机制，而非标准的标量注意力。传统的标量注意力，计算每个点的权重，然后对每个点的所有特征都按照同一个权重进行加权，然后累加求和；但是向量自注意力，通过 mlp 计算点的相对权重（使用 `x_k-x_q+p_r` 作为特征）时，把点的特征维度分为 s 组，这样计算某个点的贡献时，它的所有特征不是都具有相同的贡献，特征聚合和表达能力更强。具体计算公式如下：

$$
y_{i} = \sum_{x_{j} \in \mathcal{X}(i)} \rho(\gamma(\varphi(x_{i}) - \psi(x_{j}) + \delta)) \odot (\alpha(x_{j}) + \delta)
$$

其中 $\mathcal{X}(i)$ 表示 $x_i$ 的邻域， $\alpha()$ 表示线性变换 `linear_v`，类似自注意力里面的 $W^V$；$\delta$ 是位置编码，使用当前点和当前点的 KNN 获取的邻居点的坐标差使用 `linear_p` 线性变换获取；$\varphi()$ 表示线性变换 `linear_k`，$\psi()$ 表示线性变换 `linear_q`，相当于自注意力里面的 $W^K$ 和 $W^Q$；$\gamma()$ 是线性变换 `linear_W`，用于计算权重；$\rho()$ 是归一化函数，将权重归一化，比如 softmax 函数；$\odot$ 是哈达玛积，表示逐元素相乘；

其中 $x_i(n,c)$ 是我们当前要更新特征的点，$x_j(n,k,c)$ 是 $x_i$ 使用 KNN 查询获取到的邻居点，所以坐标差的形状是 $(n,k,c)$，权重形状为 $(n,k,c/s)$；在计算的时候，将 $\alpha(x_j)$ 的 $(n,k,c)$ 拆分为 $(n,k,s,c/s)$，也就是将通道维分组，分成 $s$ 组，组内共享注意力权重，组间权重不同，实现更细粒度的控制，组 $s$ 越小，粒度越细。

2、**局部注意力：** 为了处理大规模场景，注意力限制在局部邻域内（通过 k-NN 搜索 $k$ 个最近邻），而不是全局 。

**整体架构**

如下图所示，整体架构采用了经典的U-Net结构（编码器-解码器），包含下采样（Transition Down）和上采样（Transition Up）模块，以及核心的point transformer block（借鉴了Botterneck的结构）

![ptv1网络整体架构图](https://cdn.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/2025%5C11%5Cimage-20251128212829097.png)

### transition down

下采样模块，实现方法如下图流程所示，主要包括FPS最远点采样和KNN邻域查询，对于原始输入点坐标（通常是$x,y,z$）和特征（通常是颜色RGB或者法向量），假设原始输入点云点个数为$N$，首先经过FPS下采样选择$M$个点作为特征点，然后以这$M$个特征点为中心，查询邻域内最近的$K$个点，聚合这$K$个点的坐标和特征，这就得到了Linear层的输入$(M,K,C_{in}+3)$，然后经过Linear层处理，提升特征维度到$C_{out}$，然后在$K$这个维度上进行最大池化，得到这$M$个点的新特征$(M,C_{out})$，然后和这$M$个点的坐标，一起返回。

<img src="https://cdn.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/2025%5C12%5CGenerated%20Image%20December%2025%2C%202025%20-%205_28PM.jpeg" alt="Generated Image December 25, 2025 - 5_28PM" style="zoom: 50%;" />

具体代码实现如下：位于`pointcept/models/point_transformer/point_transformer_seg.py`中

??? note

    ```python
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
    ```

### point transformer block

这里借鉴了BottnerNeck的残差连接结构，避免网络层数导致的梯度消失问题。具体网络结构如下：

![point transformer block](https://cdn.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/2025%5C12%5CGenerated%20Image%20December%2025%2C%202025%20-%206_02PM.jpeg)

核心是中间的point Transformer Layer层，前面已经介绍了公式和计算流程。下面是实现代码：

??? note

    ```python
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
    ```

### transitionUp

分为两种情况，第一种是从encoder5到decoder1的运算，这种情况下点没有变多，只是对特征进行了重组运算

具体来说，将encoder5的输出首先在点个数维度做一个全局平均池化（也就是对所有点特征取平均值），然后一个Linear层变换一下（不改变特征维度），然后重复原先的点云个数cnt次，这样变换之后形状不发生改变，然后再和原始encoder5的输出在通道维concat起来，拼接起来之后再使用Linear提取一次特征，就得到transitionUp模块的输出。

第二种情况是decoder层之间的变化，此时需要进行跨层融合，具体来说，对于decoder1到decoder2的变化，decoder1需要和encoder4融合，decoder1的点稀疏，特征维度高，encoder4的点密集，特征维度低，这里和PointNet++的特征广播有点像，我们先把decoder1层的高维特征降低到和encoder4一致，然后遍历encoder4的每一个点，在decoder1找距离它最近的k个点，然后对这k个点的特征按照距离反比加权求和，得到encoder4中这个点的新特征；得到encoder4中每个点的新特征后，对每个点原来的旧特征进行线性变换（不改变维度），然后将变换后的旧特征和新特征进行逐元素相加（因为通道已经对齐，可以相加），就得到最终输出

详细代码如下：

??? note

    ```python
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
    ```



以上就是ptv1所有主要模块的解读了，如果用于分类任务，则不需要decoder模块，直接将encoder5的输出特征平均池化，再加上一个分类MLP头就可以了。如果是分割任务，则需要decoder模块，将下采样失去的点，通过transitionUp模块逐步添加回来，恢复点数后，再利用每个点的特征进行逐点分类，这就是分割任务。

