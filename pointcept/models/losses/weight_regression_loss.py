"""
================================================================================
脚本作用：
定义并注册用于体尺和体重预测的回归损失函数。

功能：
实现 L1 Loss (MAE - Mean Absolute Error)，并将其注册到 Pointcept 专属的 LOSSES 注册表中，解决找不到损失函数类的报错。

实现了什么：
提供了一个标准化的 `RegressionL1Loss` 类，使得配置文件的 `criteria=[dict(type="RegressionL1Loss", ...)]` 能够被成功解析。它会对预测的 7 维向量和真实的 7 维标签计算 L1 误差。

怎么实现的：
1. 继承 `torch.nn.Module`。
2. 引入 `from .builder import LOSSES`，并使用 `@LOSSES.register_module()` 装饰器将其推入正确的注册表。
3. 在 `forward` 阶段，通过 `.view(-1, 7)` 对齐张量维度，然后调用底层的 `nn.L1Loss()` 并乘以相应的权重。
================================================================================
"""

import torch
import torch.nn as nn
from .builder import LOSSES

@LOSSES.register_module()
class RegressionL1Loss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        # 确保预测值和标签的 shape 一致，统一拉平成 (N, 7)
        pred = pred.view(-1, 7)
        target = target.view(-1, 7)
        
        # 计算 L1 损失 (平均绝对误差)
        loss = self.criterion(pred, target)
        
        return loss * self.loss_weight
