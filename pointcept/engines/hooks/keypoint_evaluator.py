"""
pointcept/engines/hooks/keypoint_evaluator.py
"""
import torch
import torch.distributed as dist
import pointcept.utils.comm as comm
from pointcept.engines.hooks.builder import HOOKS
from pointcept.engines.hooks.default import HookBase

@HOOKS.register_module()
class KeypointEvaluator(HookBase):
    def __init__(self):
        pass

    def after_epoch(self):
        # 只有在定义了验证集加载器时才运行
        if self.trainer.val_loader is not None:
            self.eval()

    def eval(self):
        self.trainer.model.eval()
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        
        total_dist = 0.0
        total_samples = 0
        
        # 禁用梯度计算，节省显存
        with torch.no_grad():
            for idx, data_dict in enumerate(self.trainer.val_loader):
                # 1. 数据搬运到 GPU
                for key in data_dict.keys():
                    if isinstance(data_dict[key], torch.Tensor):
                        data_dict[key] = data_dict[key].cuda(non_blocking=True)
                
                # 2. 模型推理
                output_dict = self.trainer.model(data_dict)
                pred = output_dict["pred"]   # (B, 6, 3)
                target = data_dict["target"] # (B, 6, 3)
                
                # 3. 计算距离 (L2 Norm)
                # shape: (B, 6) -> mean -> (B,)
                dist_val = torch.norm(pred - target, p=2, dim=-1).mean(dim=1)
                
                # 4. [可选] 还原真实尺度 (如果在 Dataset 里保存了 scale)
                # 这样打印出来的误差就是真实单位（如毫米）
                if "scale" in data_dict:
                    scale = data_dict["scale"]
                    # 调整形状以支持广播: (B,) -> (B, 1) (如果需要)
                    if scale.ndim == 1: 
                        scale = scale.view(-1)
                    dist_val = dist_val * scale
                
                total_dist += dist_val.sum().item()
                total_samples += dist_val.shape[0]

        # 5. 多卡同步 (Distributed Reduce)
        # [修正] 使用 torch.distributed 直接进行 reduce
        if comm.get_world_size() > 1:
            # 必须先转为 Tensor 并在 GPU 上
            dist_tensor = torch.tensor(total_dist).cuda()
            samples_tensor = torch.tensor(total_samples).cuda()
            
            # In-place 操作
            dist.all_reduce(dist_tensor)
            dist.all_reduce(samples_tensor)
            
            total_dist = dist_tensor.item()
            total_samples = samples_tensor.item()
        
        # 计算平均距离
        mean_dist = total_dist / (total_samples + 1e-6)
        
        # 6. 打印日志
        self.trainer.logger.info(f"Eval Result: Mean Distance = {mean_dist:.4f}")
        
        # 7. 传递给 CheckpointSaver
        # 【核心Trick】因为 Saver 默认认为值越大越好，我们取负数。
        # 距离越小 -> 负数越大 -> 被判定为 Best Model
        self.trainer.comm_info["current_metric_value"] = -mean_dist
        self.trainer.comm_info["current_metric_name"] = "mean_dist"