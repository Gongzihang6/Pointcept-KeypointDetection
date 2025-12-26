<p class="theme-switcher-title">
  🎨 换个颜色，换个心情
</p>

<div class="color-picker-container">
  <button class="color-btn" data-color="red" style="background-color: #ef5350;">red</button>
  <button class="color-btn" data-color="pink" style="background-color: #ec407a;">pink</button>
  <button class="color-btn" data-color="purple" style="background-color: #ab47bc;">purple</button>
  <button class="color-btn" data-color="indigo" style="background-color: #5c6bc0;">indigo</button>
  <button class="color-btn" data-color="blue" style="background-color: #42a5f5;">blue</button>
  <button class="color-btn" data-color="cyan" style="background-color: #26c6da;">cyan</button>
  <button class="color-btn" data-color="teal" style="background-color: #26a69a;">teal</button>
  <button class="color-btn" data-color="green" style="background-color: #66bb6a;">green</button>
  <button class="color-btn" data-color="orange" style="background-color: #ffa726;">orange</button>
  <button class="color-btn" data-color="brown" style="background-color: #8d6e63;">brown</button>
  <button class="color-btn" data-color="grey" style="background-color: #bdbdbd;">grey</button>
  <button class="color-btn" data-color="black" style="background-color: #000000;">black</button>
</div>

<script>
  var buttons = document.querySelectorAll('.color-btn');
  var body = document.querySelector('body');
  buttons.forEach(function(btn) {
    btn.addEventListener('click', function() {
      var color = this.getAttribute('data-color');
      body.setAttribute('data-md-color-primary', color);
      localStorage.setItem('user-color-preference', color);
    });
  });
  var savedColor = localStorage.getItem('user-color-preference');
  if (savedColor) { body.setAttribute('data-md-color-primary', savedColor); }
</script>


# Point Transformer

## PTv1

关键词：向量注意力（Vector Attention）、局部邻域（Local Neighborhood）、可学习位置编码；

### 背景

Self-attention 在 NLP 和 2D 图像领域取得了巨大成功，点云本质上是嵌入在度量空间中的几何，具有排列不变性，这与 transformer 的核心操作（自注意力，集合算子（Set Operator））高度契合。

**解决的问题：**

- 以往的点云网络（如 Pointnet++）使用最大池化来聚合信息，可能丢失局部细节；
- 以往的 3D 注意力尝试通常是全局的（计算量过大）或直接沿用标量点积注意力（Scalar Dot-production Attention），未能充分利用 3D 空间信息；

核心创新与网络结构：

1、**Point Transformer Layer (核心层)** 这是网络的基本构建块，其核心是 **向量自注意力 (Vector Self-Attention)** 机制，而非标准的标量注意力。传统的标量注意力，计算每个点的权重，然后对每个点的所有特征都按照同一个权重进行加权，然后累加求和；但是向量自注意力，通过 mlp 计算点的相对权重（使用 `x_k-x_q+p_r` 作为特征）时，把点的特征维度分为 $s$ 组，这样计算某个点的贡献时，它的所有特征不是都具有相同的贡献，特征聚合和表达能力更强。具体计算公式如下：

$$
y_{i} = \sum_{x_{j} \in \mathcal{X}(i)} \rho(\gamma(\varphi(x_{i}) - \psi(x_{j}) + \delta)) \odot (\alpha(x_{j}) + \delta)
$$

其中 $\mathcal{X}(i)$ 表示 $x_i$ 的邻域， $\alpha()$ 表示线性变换 `linear_v`，类似自注意力里面的 $W^V$；$\delta$ 是位置编码，使用当前点和当前点的 KNN 获取的邻居点的坐标差使用 `linear_p` 线性变换获取；$\varphi()$ 表示线性变换 `linear_k`，$\psi()$ 表示线性变换 `linear_q`，相当于自注意力里面的 $W^K$ 和 $W^Q$；$\gamma()$ 是线性变换 `linear_W`，用于计算权重；$\rho()$ 是归一化函数，将权重归一化，比如 softmax 函数；$\odot$ 是哈达玛积，表示逐元素相乘；

其中 $x_i(n,c)$ 是我们当前要更新特征的点，$x_j(n,k,c)$ 是 $x_i$ 使用 KNN 查询获取到的邻居点，所以坐标差的形状是 $(n,k,c)$，权重形状为 $(n,k,c/s)$；在计算的时候，将 $\alpha(x_j)$ 的 $(n,k,c)$ 拆分为 $(n,k,s,c/s)$，也就是将通道维分组，分成 $s$ 组，组内共享注意力权重，组间权重不同，实现更细粒度的控制，组 $s$ 越小，粒度越细。

2、**局部注意力：** 为了处理大规模场景，注意力限制在局部邻域内（通过 k-NN 搜索 $k$ 个最近邻），而不是全局 。

整体架构

如下图所示，整体架构采用了经典的 U-Net 结构（编码器-解码器），包含下采样（Transition Down）和上采样（Transition Up）模块，以及核心的 point transformer block（借鉴了 Botterneck 的结构）

![ptv1 网络整体架构图](https://cdn.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/2025%5C11%5Cimage-20251128212829097.png)

### transition down

下采样模块，实现方法如下图流程所示，主要包括 FPS 最远点采样和 KNN 邻域查询，对于原始输入点坐标（通常是 $x,y,z$）和特征（通常是颜色 RGB 或者法向量），假设原始输入点云点个数为 $N$，首先经过 FPS 下采样选择 $M$ 个点作为特征点，然后以这 $M$ 个特征点为中心，查询邻域内最近的 $K$ 个点，聚合这 $K$ 个点的坐标和特征，这就得到了 Linear 层的输入 $(M,K,C_{in}+3)$，然后经过 Linear 层处理，提升特征维度到 $C_{out}$，然后在 $K$ 这个维度上进行最大池化，得到这 $M$ 个点的新特征 $(M,C_{out})$，然后和这 $M$ 个点的坐标，一起返回。

<img src="https://cdn.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/2025%5C12%5CGenerated%20Image%20December%2025%2C%202025%20-%205_28PM.jpeg" alt="Generated Image December 25, 2025 - 5_28PM" style="zoom: 50%;" />

具体代码实现如下：位于 `pointcept/models/point_transformer/point_transformer_seg.py` 中

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

这里借鉴了 BottnerNeck 的残差连接结构，避免网络层数导致的梯度消失问题。具体网络结构如下：

![point transformer block](https://cdn.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/2025%5C12%5CGenerated%20Image%20December%2025%2C%202025%20-%206_02PM.jpeg)

核心是中间的 point Transformer Layer 层，前面已经介绍了公式和计算流程。下面是实现代码：

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

分为两种情况，第一种是从 encoder5 到 decoder1 的运算，这种情况下点没有变多，只是对特征进行了重组运算

具体来说，将 encoder5 的输出首先在点个数维度做一个全局平均池化（也就是对所有点特征取平均值），然后一个 Linear 层变换一下（不改变特征维度），然后重复原先的点云个数 cnt 次，这样变换之后形状不发生改变，然后再和原始 encoder5 的输出在通道维 concat 起来，拼接起来之后再使用 Linear 提取一次特征，就得到 transitionUp 模块的输出。

第二种情况是 decoder 层之间的变化，此时需要进行跨层融合，具体来说，对于 decoder1 到 decoder2 的变化，decoder1 需要和 encoder4 融合，decoder1 的点稀疏，特征维度高，encoder4 的点密集，特征维度低，这里和 PointNet++的特征广播有点像，我们先把 decoder1 层的高维特征降低到和 encoder4 一致，然后遍历 encoder4 的每一个点，在 decoder1 找距离它最近的 k 个点，然后对这 k 个点的特征按照距离反比加权求和，得到 encoder4 中这个点的新特征；得到 encoder4 中每个点的新特征后，对每个点原来的旧特征进行线性变换（不改变维度），然后将变换后的旧特征和新特征进行逐元素相加（因为通道已经对齐，可以相加），就得到最终输出

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



以上就是 ptv1 所有主要模块的解读了，如果用于分类任务，则不需要 decoder 模块，直接将 encoder5 的输出特征平均池化，再加上一个分类 MLP 头就可以了。如果是分割任务，则需要 decoder 模块，将下采样失去的点，通过 transitionUp 模块逐步添加回来，恢复点数后，再利用每个点的特征进行逐点分类，这就是分割任务。



## PTv2

论文标题为：Point Transformer V2: Grouped Vector Attention and Partion-based Pooling

关键词：分组向量注意力（Grouped Vector Attenion）、分区池化（Partion-based Pooling）、更强的位置编码

### 背景

PTv1 虽然效果好，但存在效率和参数量的缺陷，具体来说：

1. 参数过拟合与深度限制。PTv1 的向量注意力使用全连接层来生成权重，随着通道数增加，参数量剧增，导致过拟合，限制了网络深度；
2. 池化效率低。PTv1 使用的 FPS+KNN 采样方式，既耗时，又导致空间对齐不佳（邻域重叠不可控）；
3. 位置信息利用不足。以前的方法没有充分利用 3D 坐标中的几何知识。

整体架构

ptv2 的整体架构和 ptv1 差不多，都是基于 U-Net 的编码器-解码器架构，只是在向量注意力的计算、可学习位置编码上的利用、以及池化方式等方面，做了优化，所以论文中没有绘制整体网络架构图。

### Grouped Vector Attention

第一个核心改进在于注意力的计算，为了解决参数量爆炸问题，ptv2 将特征通道划分为了 $g$ 个组（但是这一点 pointcept 在 ptv1 的代码中也实现了，在代码中将特征分为了 $s$ 组，但是 ptv1 原始论文中没有提到分组向量注意力，而是使用 MLP 为特征的每一个维度都计算单独的权重，可能 pointcept 仓库写这个代码的时候 ptv2 早就已经出了，就把这个融合到 ptv1 的代码里面了）。

划分为 $g$ 个组后，同一个组内的特征通道共享同一个注意力权重向量，而不是每一个通道一个权重，这大大减少了参数量和过拟合的风险，同时保留了向量注意力的优势。

如下图所示，在计算分组向量注意力的时候，输入给计算权重的 mlp 的特征，不再是 $\gamma(Q,K)$ 的全部特征，而是将 $\gamma(Q,K)$ 和值向量 $v$ 一样进行分组（分为 $g$ 组），然后每个组单独使用 mlp 计算注意力权重（这一点和多头自注意力比较像，论文中也画了图 c 进行对比，区别在于计算注意力权重的方式不同，多头注意力通过 $\gamma(Q,K)$ 的点积计算注意力权重），如果是全通道向量注意力，mlp 参数数量将是 $c^2$，如果分为 $g$ 组，参数数量就是 $(c/g)^2 \times g =c^2/g$，参数减少为原来的 $1/g$，兼顾了向量注意力的优势以及计算效率。

> [!IMPORTANT]
> 作者发现，严格限制“组内只看组内参数 $\gamma(Q,K)$”的 `GroupedLinear` 对最终性能的贡献微乎其微，但会拖慢速度。直接用一个全连接层（让所有通道通过线性组合生成 $g$ 个组的权重）效果一样好，且能利用 PyTorch/CUDA 高度优化的 Dense Layer 算子。
> **优势**：**速度更快，代码更简洁**。

![分组向量注意力](https://cdn.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/2025%5C12%5Cimage-20251226194800308.png)

同样，论文中下面这张图这对比了 ptv2 改进的向量注意力计算方式和池化方式和 ptv1 的一个对比。首先，在注意力的计算上，前面已经说过了，提出了分组来兼顾向量注意力优势和计算效率的平衡，但是位置编码的使用上有不同，ptv2 代码中给了两种选择：

??? note

    ```python
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
    ```

一种是位置编码乘子法（和 ptv1 一样，计算注意力的时候使用 KNN 查询邻域点，然后计算坐标差、注意力权重等），具体来说，将坐标差 pos 进行线性变换，变换到和特征维度一致，然后和 $\gamma(Q,K)$ 进行逐元素乘积，将结果送入到计算注意力权重的 mlp 层，得到注意力权重，值向量 value 不变，没有加位置编码；即

$$
w =\omega(\gamma(K, Q)\odot Linear(\Delta pos))
$$

另一种是位置编码偏置法，具体来说，先把坐标差 pos 进行线性变换，变换到和特征维度一致，然后和 $\gamma(Q,K)$ 进行相加，将相加的结果送入到计算注意力权重的 mlp 层，得到注意力权重，同时值向量 value 也加上变换后的坐标编码，对变化后的值向量进行加权求和；即

$$
w =\omega(\gamma(Q, K)+Linear(\Delta pos)), \quad y = w(value+Linear(\Delta pos))
$$

这一点代码中的实现和论文中貌似有点不一致，论文中图示的效果感觉应该是下面这种：

$$
w =\omega(Linear(\Delta pos)\odot\gamma(Q, K)+Linear(\Delta pos))
$$

然后值向量保持不变。

![ptv2 和 v1 在向量注意力、池化方式上的对比](https://cdn.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/2025%5C12%5Cimage-20251226195426222.png)

然后是池化方式上的改进，前面说过，ptv1 使用 FPS 和 KNN 进行下采样池化，找邻居，聚合邻居特征。但是这种方法的查询集在空间上不对齐，且不同查询集之间的重叠区域不可控，这导致计算效率低且显存占用高。因此 ptv2 使用了基于区域划分的池化，如上图所示，ptv1 的池化和反池化都要通过 KNN 查询最近点，但是 ptv2 通过将点云空间切割成互不重叠的分区（比如均匀网格，即 Grid Pooling）

**操作流程：**

1. **划分（Partition）：** 将整个点云空间 $\mathcal{M}$ 切割成多个互不重叠的子集 $[\mathcal{M}_1, \mathcal{M}_2, ..., \mathcal{M}_{n'}]$。就像把一个大盒子切成很多小方块（Grid）。
2. **融合（Fusion）：** 对每一个小方块内的所有点进行聚合，生成 **一个** 新的代表点（Pooling Point）。

具体来说，对小方块内所有点特征进行线性变换，然后对每一个样本点云进行网格划分，同一个网格中点的坐标采用平均池化，用网格内所有点的质心做代表；同一个网格中点的特征采用最大池化，每一维特征选择所有点中值最大的作为代表，从 $(n,c)$ 到 $(c)$。

