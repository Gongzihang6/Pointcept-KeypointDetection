"""
================================================================================
脚本作用：
封装 Pointcept 默认分类器，实时计算并打印体尺体重的真实物理误差。

核心机制：
利用 PyTorch 的 register_forward_hook 拦截最后一层线性网络 (cls_head) 的输出预测值。
在不重写底层复杂逻辑的前提下，悄悄把真实的 MAE 误差（厘米/千克）塞进 output_dict，让外层的 Logger 自动打印。
================================================================================
"""
import torch
from .builder import MODELS
from .default import DefaultClassifier

@MODELS.register_module()
class PigBodyRegressor(DefaultClassifier):
    def forward(self, input_dict):
        # 准备一个空列表，用来临时存放网络输出的预测值
        pred_cache = []

        # 黑科技：把钩子挂在网络的最后一层线性头 (cls_head) 上
        def cls_head_hook(module, inputs, output):
            # output 就是网络刚刚预测出来的 7 维数值
            pred_cache.append(output)

        # 1. 把钩子挂在分类头上 (因为这是标准的 PyTorch Module，绝对不会报错)
        handle = self.cls_head.register_forward_hook(cls_head_hook)
        
        # 2. 正常运行 Pointcept 底层的复杂前向传播和 Loss 计算
        output_dict = super().forward(input_dict)
        
        # 3. 运行完毕，拔掉钩子，防止内存泄漏
        handle.remove()
        
        # 4. 如果成功抓到了预测值，并且是在训练/验证模式 (output_dict 是字典)
        if pred_cache and isinstance(output_dict, dict) and "category" in input_dict:
            pred = pred_cache[0].view(-1, 7)
            target = input_dict["category"].view(-1, 7)
            
            with torch.no_grad():
                # 乘回 100，还原为真实的物理尺寸和重量
                pred_real = pred * 100.0
                target_real = target * 100.0
                
                # 计算绝对误差 (MAE) 并按列求平均
                abs_err = torch.abs(pred_real - target_real).mean(dim=0)
                
                # 键名尽量短，保持终端日志整洁好看
                output_dict['e_Len'] = abs_err[0]
                output_dict['e_Wid'] = abs_err[1]
                output_dict['e_Hei'] = abs_err[2]
                output_dict['e_Che'] = abs_err[3]
                output_dict['e_Wai'] = abs_err[4]
                output_dict['e_Hip'] = abs_err[5]
                output_dict['e_Wei'] = abs_err[6]
            
        return output_dict
