"""
evaluate_all_splits.py
Evaluate model on train, val, and test splits and calculate Precision, Recall, IoU, etc.
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F

from pointcept.engines.defaults import default_config_parser, default_setup
from pointcept.engines.launch import launch
from pointcept.engines.test import TESTERS
from pointcept.datasets import build_dataset
import pointcept.utils.comm as comm
from pointcept.utils.logger import get_root_logger

# Import the base tester to inherit from it
from pointcept.engines.test import SemSegTester

class CustomSemSegTester(SemSegTester):
    def __init__(self, cfg, split_name="test", model=None):
        # We temporarily modify the test config to point to the desired split
        # while keeping the test_mode and test_cfg
        original_test_cfg = cfg.data.test.copy()
        
        # We take the split string from the corresponding train/val/test data config if available
        # or fallback to string name.
        if split_name in cfg.data:
            target_split = cfg.data[split_name].get("split", split_name)
        else:
            target_split = split_name
            
        cfg.data.test.split = target_split
        self.split_name = split_name
        
        # Init base tester
        super().__init__(cfg, model=model)
        
        # Restore configuration
        cfg.data.test = original_test_cfg

    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(f">>>>>>>>>>>>>>>> Start Evaluation on Split: {self.split_name.upper()} >>>>>>>>>>>>>>>>")

        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, f"result_{self.split_name}")
        os.makedirs(save_path, exist_ok=True)
        
        comm.synchronize()

        num_classes = self.cfg.data.num_classes
        
        # We accumulate confusion matrix manually for Precision/Recall/IoU
        # rows: target, cols: prediction
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

        for idx, data_dict in enumerate(self.test_loader):
            data_dict = data_dict[0]
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            
            pred = torch.zeros((segment.size, num_classes)).cuda()
            
            fragment_batch_size = 8  # 🚀 核心提速：增大单卡内的片段级测试批量，将速度提升数倍
            num_fragments = len(fragment_list)
            num_batches = (num_fragments + fragment_batch_size - 1) // fragment_batch_size
            
            for i in range(num_batches):
                s_i, e_i = i * fragment_batch_size, min((i + 1) * fragment_batch_size, num_fragments)
                # using internal Pointcept collate_fn
                from pointcept.datasets import collate_fn
                input_dict = collate_fn(fragment_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                
                with torch.no_grad():
                    pred_part_batch = self.model(input_dict)["seg_logits"]
                    pred_part_batch = F.softmax(pred_part_batch, -1)
                
                if self.cfg.empty_cache:
                    torch.cuda.empty_cache()
                    
                bs = e_i - s_i
                for j in range(bs):
                    if "offset" in input_dict:
                        # Find indices belonging to batch j
                        start_idx = 0 if j == 0 else input_dict["offset"][j-1]
                        end_idx = input_dict["offset"][j]
                        idx_part = input_dict["index"][start_idx:end_idx]
                        pred_part = pred_part_batch[start_idx:end_idx]
                        pred[idx_part, :] += pred_part
                    else:
                        idx_part = input_dict["index"]
                        pred[idx_part, :] += pred_part_batch
            
            # Predict labels
            pred = pred.max(1)[1].data.cpu().numpy()
            
            # Mask out ignore index (-1)
            mask = segment != self.cfg.data.ignore_index
            pred = pred[mask]
            target = segment[mask]
            
            # Calculate intersection and union for the current pointcloud
            conf_mat = np.bincount(
                num_classes * target.astype(np.int64) + pred,
                minlength=num_classes**2
            ).reshape(num_classes, num_classes)
            
            confusion_matrix += conf_mat
            
            if (idx + 1) % 10 == 0 or (idx + 1) == len(self.test_loader):
                logger.info(f"[{idx + 1}/{len(self.test_loader)}] Evaluated: {data_name}")

        # Reduce over multiple GPUs if distributed
        if comm.get_world_size() > 1:
            confusion_matrix = torch.tensor(confusion_matrix, device="cuda")
            torch.distributed.all_reduce(confusion_matrix)
            confusion_matrix = confusion_matrix.cpu().numpy()

        if comm.is_main_process():
            # Calculate Metrics
            # TP = diag(conf), FP = sum(col) - TP, FN = sum(row) - TP
            TP = np.diag(confusion_matrix)
            FP = np.sum(confusion_matrix, axis=0) - TP
            FN = np.sum(confusion_matrix, axis=1) - TP
            
            # Deal with division by zero
            eps = 1e-6
            union = TP + FP + FN
            target_count = TP + FN
            pred_count = TP + FP
            
            iou = TP / (union + eps)
            recall = TP / (target_count + eps)
            precision = TP / (pred_count + eps)
            
            # Overall Accuracy
            overall_acc = np.sum(TP) / (np.sum(target_count) + eps)
            
            logger.info(f"======= Final Results for Split: {self.split_name.upper()} =======")
            logger.info("Class-wise Metrics:")
            
            class_names = self.cfg.data.names if "names" in self.cfg.data else [str(i) for i in range(num_classes)]
            for i in range(num_classes):
                logger.info(f"  Class [{i} - {class_names[i]}]: "
                            f"IoU: {iou[i]:.4f} | "
                            f"Recall(Acc): {recall[i]:.4f} | "
                            f"Precision: {precision[i]:.4f}")
            
            logger.info(f"Mean IoU: {np.mean(iou):.4f}")
            logger.info(f"Mean Recall(mAcc): {np.mean(recall):.4f}")
            logger.info(f"Mean Precision: {np.mean(precision):.4f}")
            logger.info(f"Overall Accuracy (OA): {overall_acc:.4f}")
            logger.info("=========================================================\n")
            
        return self.model


def main_worker(cfg):
    cfg = default_setup(cfg)
    
    splits_to_evaluate = ["train", "val", "test"]
    
    loaded_model = None
    for split_index, split in enumerate(splits_to_evaluate):
        tester = CustomSemSegTester(cfg, split_name=split, model=loaded_model)
        loaded_model = tester.test() # Keep the loaded model to reuse across splits


def main():
    parser = argparse.ArgumentParser("evaluate_all_splits")
    parser.add_argument("--config-file", required=True, type=str, help="path to config file")
    parser.add_argument("--weight", required=True, type=str, help="path to model weights")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus")
    parser.add_argument("--num-machines", type=int, default=1, help="number of machines")
    parser.add_argument("--machine-rank", type=int, default=0, help="rank of machine")
    parser.add_argument("--dist-url", default="auto", type=str, help="dist url")
    parser.add_argument("options", nargs=argparse.REMAINDER, help="modify config options, e.g. options model.test_only=True")
    
    args = parser.parse_args()
    
    # 构造 options_dict 而不是直接喂给 config list
    try:
        from pointcept.utils.config import DictAction
        # 如果能调包就把命令变成 ["weight=xxx"] 交给 argparse
    except ImportError:
        pass
    
    from pointcept.engines.defaults import default_config_parser
    cfg = default_config_parser(args.config_file, None) # 这里直接传 None 避开其内部的合并字典 bug
    cfg.weight = args.weight # 直接给解析出来的 config 对象赋值

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )

"""
python tools/evaluate_all_splits.py --config-file configs/pigseg/semseg-ptv3-v1m1-0-base.py --weight exp/PTV3_PigSeg_0511/model/model_best.pth
"""
if __name__ == "__main__":
    main()
