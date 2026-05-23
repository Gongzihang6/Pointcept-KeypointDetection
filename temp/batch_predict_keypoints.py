import os
import glob
import argparse
import numpy as np
import open3d as o3d
import torch
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from pointcept.utils.config import Config, DictAction
from pointcept.models import build_model
from pointcept.datasets.transform import Compose
from pointcept.datasets import point_collate_fn

def get_args():
    parser = argparse.ArgumentParser(description="Batch Predict Keypoints from PCD")
    parser.add_argument("--data-dir", type=str, default=r"F:\Gongzihang\2026\data\KeyPointData", help="时间戳文件夹所在根目录")
    parser.add_argument("--config-file", required=True, help="配置文件路径")
    parser.add_argument("--weights", required=True, help="模型权重文件路径 (.pth)")
    parser.add_argument("--agg-method", choices=["argmax", "weighted"], default="weighted", help="聚合手段")
    parser.add_argument("--mask-thresh", type=float, default=0.5, help="weighted 聚合的 mask 阈值")
    parser.add_argument("--model-name", type=str, default="PTv3", help="用于保存结果的后缀名称标识")
    return parser.parse_args()

def setup_model(cfg, weights_path):
    print(f"=> Building model from config: {cfg.model.type}")
    model = build_model(cfg.model)
    checkpoint = torch.load(weights_path, map_location="cuda")
    state_dict = checkpoint.get("state_dict", checkpoint)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    model.cuda()
    model.eval()
    return model

def compute_curvature(pcd, radius=0.05, max_nn=30):
    """手动计算主曲率"""
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.orient_normals_consistent_tangent_plane(100)
    
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    points = np.asarray(pcd.points)
    curvatures = np.zeros(len(points))
    
    for i in range(len(points)):
        [_, idx, _] = pcd_tree.search_hybrid_vector_3d(points[i], radius, max_nn)
        if len(idx) < 3:
            continue
        neighbors = points[idx, :]
        mean = np.mean(neighbors, axis=0)
        cov = np.cov(neighbors - mean, rowvar=False)
        eigenvalues, _ = np.linalg.eigh(cov)
        eigenvalues = np.sort(eigenvalues)
        if np.sum(eigenvalues) > 1e-6:
            curvatures[i] = eigenvalues[0] / np.sum(eigenvalues)
    return np.asarray(pcd.normals), curvatures

def process_pcd_to_dict(pcd_path, transform):
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    normals, curvatures = compute_curvature(pcd)
    
    coord = points.astype(np.float32)
    feat = np.concatenate([normals, curvatures[:, None]], axis=1).astype(np.float32)
    coord_feat = normals.astype(np.float32)
    
    # 模拟 Dataset 里面的预处理
    centroid = np.mean(coord, axis=0)
    coord -= centroid
    dist = np.sqrt(np.sum(coord**2, axis=1))
    m = np.max(dist) if dist.shape[0] > 0 else 1.0
    if m < 1e-6: m = 1.0
    scale = np.array(m, dtype=np.float32)
    coord = coord / scale
    
    data_dict = dict(coord=coord, feat=feat, coord_feat=coord_feat, centroid=centroid, scale=scale)
    
    if transform is not None:
        data_dict = transform(data_dict)
    
    # 找回可能被 Collect 丢掉的额外信息
    if "scale" not in data_dict:
        data_dict["scale"] = scale
    if "centroid" not in data_dict:
        data_dict["centroid"] = centroid
    
    # # 构造 target 为 dummy tensor，因为我们需要满足网络可能的 target 取值或 collect，但对于 infer 可以不要
    # if "target" not in data_dict:
    #     # Fake target just in case transform Collect needs it
    #     data_dict["target"] = torch.zeros((len(coord), 6, 4))
        
    return data_dict

def main():
    args = get_args()
    cfg = Config.fromfile(args.config_file)
    
    model = setup_model(cfg, args.weights)
    
    # 从配置中提取预处理流程 (测试集)
    transform_cfg = cfg.data.test.transform
    # 如果有 Collect 操作可能会报错缺少 target，我们把 target 从 Collect keys 里面删掉
    for t in transform_cfg:
        if t.get("type", "") == "Collect":
            if "target" in t.get("keys", []):
                t["keys"] = tuple([k for k in t["keys"] if k != "target"])
    transform = Compose(transform_cfg)
    
    num_kps = cfg.model.num_keypoints
    
    # 遍历外部的所有子文件夹
    data_dir = args.data_dir
    # 兼容 Wsl 路径和 Windows 路径转换
    if "\\" in data_dir and os.name == "posix":
        data_dir = data_dir.replace("\\", "/") # 如果传进来的是win路径直接在wsl里面用
    
    subdirs = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    print(f"=> Found {len(subdirs)} timestamp directories in {data_dir}")
    
    for subdir in tqdm(subdirs, desc="Processing PCDs"):
        timestamp = os.path.basename(subdir)
        pcd_path = os.path.join(subdir, f"{timestamp}.pcd")
        
        if not os.path.exists(pcd_path):
            print(f"[Warning] PCD not found: {pcd_path}")
            continue
            
        # [处理数据]
        try:
            data_dict = process_pcd_to_dict(pcd_path, transform)
            batch = point_collate_fn([data_dict])
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.cuda(non_blocking=True)
                
            # [推理]
            with torch.no_grad():
                res = model(batch)
                pred = res["pred"] if isinstance(res, dict) else res
            
            # [反算结果]
            pred_offset = pred[..., :3].cpu().numpy()
            pred_mask = pred[..., 3].cpu().numpy()
            coord = batch["coord"].cpu().numpy()
            scale = batch["scale"].cpu().numpy()[0]
            centroid = batch.get("centroid", torch.zeros((1, 3))).cpu().numpy()[0]
            true_coord = coord * scale + centroid
            
            pred_kps = np.zeros((num_kps, 3))
            for k in range(num_kps):
                k_probs = pred_mask[:, k]
                if args.agg_method == "argmax":
                    best_idx = np.argmax(k_probs)
                    pred_kps[k] = true_coord[best_idx] + pred_offset[best_idx, k] * scale
                else:
                    valid_mask = k_probs > args.mask_thresh
                    if np.any(valid_mask):
                        valid_probs = k_probs[valid_mask]
                        valid_coords = true_coord[valid_mask]
                        valid_offs = pred_offset[valid_mask, k]
                        candidate_kps = valid_coords + valid_offs * scale
                        weights = valid_probs / np.sum(valid_probs)
                        pred_kps[k] = np.sum(candidate_kps * weights[:, None], axis=0)
                    else:
                        best_idx = np.argmax(k_probs)
                        pred_kps[k] = true_coord[best_idx] + pred_offset[best_idx, k] * scale
            
            # [保存结果到 txt]
            output_file = os.path.join(subdir, f"关键点坐标预测结果_{args.model_name}.txt")
            np.savetxt(output_file, pred_kps, fmt="%.6f")
            
        except Exception as e:
            print(f"[Error] Failed to process {pcd_path}: {e}")

"""
python temp/batch_predict_keypoints.py \
    --data-dir "/mnt/f/Gongzihang/2026/data/KeyPointData" \
    --config-file configs/my_dataset/offset_keypoint_octformer.py \
    --weights exp/offset_keypoint_octformer_0512/model/model_best.pth \
    --agg-method weighted \
    --mask-thresh 0.8 \
    --model-name OctFormer_0512
"""
if __name__ == "__main__":
    main()