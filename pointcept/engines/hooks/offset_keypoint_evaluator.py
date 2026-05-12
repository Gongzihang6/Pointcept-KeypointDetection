"""
pointcept/engines/hooks/offset_keypoint_evaluator.py
"""
import torch
import torch.distributed as dist
import pointcept.utils.comm as comm
from pointcept.engines.hooks.builder import HOOKS
from pointcept.engines.hooks.default import HookBase

@HOOKS.register_module()
class OffsetKeypointEvaluator(HookBase):
    def __init__(self, num_keypoints=6):
        self.num_keypoints = num_keypoints

    def after_epoch(self):
        if self.trainer.val_loader is not None:
            self.eval()

    def eval(self):
        self.trainer.model.eval()
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Offset-based Evaluation >>>>>>>>>>>>>>>>")
        
        total_dist = 0.0
        total_samples = 0
        
        # 为了更细致地追踪每个关键点的平均误差
        num_kps = self.num_keypoints
        total_dist_per_kp = [0.0] * num_kps
        samples_per_kp = [0] * num_kps

        with torch.no_grad():
            for idx, data_dict in enumerate(self.trainer.val_loader):
                for key in data_dict.keys():
                    if isinstance(data_dict[key], torch.Tensor):
                        data_dict[key] = data_dict[key].cuda(non_blocking=True)
                
                # 模型推理
                output_dict = self.trainer.model(data_dict)
                pred = output_dict["pred"]       # (N_total, num_kps, 4)
                target = data_dict["target"]     # (N_total, num_kps, 4)
                coord = data_dict["coord"]       # (N_total, 3)
                batch_offsets = data_dict["offset"] # (B,)
                
                scale = data_dict.get("scale", None) # (B,)
                B = len(batch_offsets)
                
                for b in range(B):
                    start = 0 if b == 0 else batch_offsets[b - 1]
                    end = batch_offsets[b]
                    
                    b_coord = coord[start:end]   # (N_b, 3)
                    b_pred = pred[start:end]     # (N_b, num_kps, 4)
                    b_target = target[start:end] # (N_b, num_kps, 4)
                    b_scale = scale[b].item() if scale is not None else 1.0

                    b_dist_sum = 0.0
                    b_valid_kps = 0

                    for k in range(num_kps):
                        # --- 还原 Ground Truth Keypoint ---
                        # target 第四维是 mask (有效掩码)
                        # 我们找出所有 mask 为 1 的点，用它们反推 GT 关键点 (keypoint = point + offset) 
                        gt_mask = b_target[:, k, 3]
                        valid_idx = torch.nonzero(gt_mask > 0).squeeze(-1)
                        if len(valid_idx) == 0:
                            # 假如在这个点云中没有任何点在距离阈值 r 范围内，跳过
                            continue
                            
                        # 取所有 mask=1 的点反推关键点，求平均获得精确的 GT
                        gt_kp = (b_coord[valid_idx] + b_target[valid_idx, k, :3]).mean(dim=0)
                        
                        # --- 还原 Predicted Keypoint ---
                        # 根据预测的 mask 取置信度最高的点作为“基准点”
                        pred_mask = b_pred[:, k, 3]
                        best_idx = torch.argmax(pred_mask)
                        
                        # 预测 keypoint = 该该点的归一化坐标 + 模型给该点回归出的位移参数
                        pred_kp = b_coord[best_idx] + b_pred[best_idx, k, :3]
                        
                        # --- 误差计算并乘回真实尺度 ---
                        dist_val = torch.norm(pred_kp - gt_kp, p=2) * b_scale
                        
                        total_dist_per_kp[k] += dist_val.item()
                        samples_per_kp[k] += 1
                        
                        b_dist_sum += dist_val.item()
                        b_valid_kps += 1
                        
                    # 只记录有有效关键点的点云样本误差
                    if b_valid_kps > 0:
                        total_dist += (b_dist_sum / b_valid_kps)
                        total_samples += 1

        # 多卡同步 (Distributed Reduce)
        if comm.get_world_size() > 1:
            dist_tensor = torch.tensor([total_dist, total_samples] + total_dist_per_kp + samples_per_kp).cuda()
            dist.all_reduce(dist_tensor)
            
            total_dist = dist_tensor[0].item()
            total_samples = dist_tensor[1].item()
            
            for k in range(num_kps):
                total_dist_per_kp[k] = dist_tensor[2 + k].item()
                samples_per_kp[k] = dist_tensor[2 + num_kps + k].item()
        
        # 计算全局平均距离误差
        mean_dist = total_dist / (total_samples + 1e-6)
        
        self.trainer.logger.info(f"Val Result: Mean Distance (Over all active keypoints and samples) = {mean_dist:.4f}")
        for k in range(num_kps):
            k_mean_dist = total_dist_per_kp[k] / (samples_per_kp[k] + 1e-6)
            self.trainer.logger.info(f"  Keypoint {k} Mean Distance: {k_mean_dist:.4f} (Valid Samples Evaluated: {int(samples_per_kp[k])})")
            
            # 同步输出至 TensorBoard/WandB
            if self.trainer.writer is not None:
                self.trainer.writer.add_scalar(f"val/KP_{k}_MeanDist", k_mean_dist, self.trainer.epoch + 1)

        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/MeanDist", mean_dist, self.trainer.epoch + 1)
            
        # 【核心策略】将度量指标变为负数，以此适应 Pointcept SaveBest (以数值越大作为更优)
        self.trainer.comm_info["current_metric_value"] = -mean_dist
        self.trainer.comm_info["current_metric_name"] = "mean_dist"
