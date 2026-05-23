"""
Keypoint Detection Inference & Visualization Script
功能：
1. 支持单样本推理：计算误差 + Open3D 可视化（球体=真值，立方体=预测）
2. 支持批量推理：计算整个数据集的平均误差 (Mean) 和标准差 (Std)
3. 架构通用：通过 config 文件自动加载对应的模型架构
"""

import argparse
import os
import sys
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

# 添加项目根目录到 python path，确保能导入 pointcept
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from pointcept.utils.config import Config, DictAction
from pointcept.models import build_model
from pointcept.datasets import build_dataset, point_collate_fn
from pointcept.utils.misc import intersection_and_union, make_dirs
from pointcept.engines.defaults import default_argument_parser
from torch.utils.data import ConcatDataset
def get_args():
    parser = argparse.ArgumentParser(description="Pointcept Keypoint Inference")
    parser.add_argument("--config-file", default="configs/my_dataset/keypoint_ptv3.py", help="配置文件路径")
    parser.add_argument("--options", nargs="+", action=DictAction, help="覆盖配置文件的参数")
    parser.add_argument("--weights", default=None, required=True, help="模型权重文件路径 (.pth)")
    parser.add_argument("--subset", default="val", choices=["train", "val", "test", "all"], help="数据集划分")
    parser.add_argument("--idx", type=int, default=-1, help="单样本索引。如果为 -1，则进行批量推理")
    
    # 可视化参数
    parser.add_argument("--visualize", action="store_true", help="是否开启 Open3D 可视化 (仅单样本模式有效)")
    parser.add_argument("--sphere-radius", type=float, default=10, help="真实关键点(球)的半径")
    parser.add_argument("--cube-size", type=float, default=10, help="预测关键点(正方体)的边长")
    parser.add_argument("--point-size", type=float, default=2.0, help="Open3D 可视化时点云的点大小")
    parser.add_argument("--agg-method", choices=["argmax", "weighted"], default="argmax", help="关键点聚合策略：'argmax'（置信度最高点） 或 'weighted'（置信度大于阈值的点加权平均）")
    parser.add_argument("--mask-thresh", type=float, default=0.5, help="当 agg-method 为 weighted 时的掩码阈值")
    parser.add_argument("--save-dir", default=None, help="结果保存路径 (可选)")
    
    args = parser.parse_args()
    return args

def setup_model(cfg, weights_path):
    """加载模型和权重"""
    print(f"=> Building model from config: {cfg.model.type}")
    model = build_model(cfg.model)
    
    if os.path.isfile(weights_path):
        print(f"=> Loading weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cuda")
        state_dict = checkpoint.get("state_dict", checkpoint)
        # 移除 'module.' 前缀 (如果是 DDP 训练保存的)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=True)
    else:
        raise FileNotFoundError(f"No weights found at {weights_path}")
    
    model.cuda()
    model.eval()
    return model

def create_colored_mesh(geometry_type, center, color, size):
    """创建带颜色的几何体 (球或立方体)"""
    if geometry_type == 'sphere':
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=size)
    elif geometry_type == 'box':
        mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
        # Box 默认原点在角落，需要平移到中心
        mesh.translate(-np.array([size/2, size/2, size/2]))
    
    mesh.translate(center)
    mesh.paint_uniform_color(color)
    return mesh

def visualize_single(coord, pred_kps, target_kps, args, num_kps, active_points_info=None):
    """使用 Open3D 可视化 (支持调整点大小，点云特定高亮着色)"""
    print(f"=> Visualizing... (Point Size: {args.point_size})")
    geometries = []

    # 1. 点云颜色设定
    point_colors = np.ones((coord.shape[0], 3)) * 0.7  # 默认全局灰色
    cmap = plt.get_cmap("jet")
    colors = [cmap(i / (num_kps - 1 if num_kps > 1 else 1))[:3] for i in range(num_kps)]

    # 着色参与预测的关键点区域
    if active_points_info is not None:
        for k, info in active_points_info.items():
            idx = info['indices']
            probs = info['probs']
            if len(idx) > 0:
                base_color = np.array(colors[k])
                probs_arr = np.array(probs)
                # 使用置信度概率来调节颜色亮度 (概率越高，颜色越纯/饱和，越低越靠近灰白)
                # 这会让高置信度的点非常醒目，同时同属一类的点有相同的色相
                intensities = np.clip(probs_arr, 0, 1.0).reshape(-1, 1)
                colored_points = base_color * intensities + np.array([0.7, 0.7, 0.7]) * (1 - intensities)
                point_colors[idx] = colored_points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    geometries.append(pcd)

    # 3. 绘制关键点（真实值：球；预测值：立方体）
    for i in range(num_kps):
        # 真实值：圆球 (Sphere)
        try:
            if target_kps is not None and not np.any(np.isnan(target_kps[i])):
                sphere = create_colored_mesh('sphere', np.asarray(target_kps[i]), colors[i], args.sphere_radius)
                geometries.append(sphere)
        except Exception:
            # 容错：若 target_kps 不可用则忽略
            pass

        # 预测值：正方体 (Cube)
        try:
            if pred_kps is not None and not np.any(np.isnan(pred_kps[i])):
                cube = create_colored_mesh('box', np.asarray(pred_kps[i]), colors[i], args.cube_size)
                geometries.append(cube)
        except Exception:
            pass

    # 4. [修改核心] 使用 Visualizer 来控制渲染选项
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Sample {args.idx} (Sphere=GT, Cube=Pred)", width=1024, height=768)
    
    # 添加所有几何体
    for geom in geometries:
        vis.add_geometry(geom)
        
    # 获取并修改渲染选项
    opt = vis.get_render_option()
    opt.point_size = args.point_size        # [关键] 设置点的大小
    opt.background_color = np.asarray([1, 1, 1]) # [可选] 设置背景为白色，看起来更清晰
    
    vis.run()
    vis.destroy_window()


def inference_single_sample(cfg, model, dataset, args):
    """单样本推理逻辑"""
    idx = args.idx
    if idx >= len(dataset):
        print(f"Error: Index {idx} out of bounds (Dataset size: {len(dataset)})")
        return

    # 1. 获取数据
    data_dict = dataset[idx]
    # Collate: 即使是单个样本，也需要伪造成 batch 为 1 的形式 (增加 batch 维度)
    data_dict = point_collate_fn([data_dict])
    
    # 转移到 GPU
    for key in data_dict:
        if isinstance(data_dict[key], torch.Tensor):
            data_dict[key] = data_dict[key].cuda(non_blocking=True)

    # 2. 推理
    with torch.no_grad():
        result = model(data_dict)
        pred = result["pred"] if isinstance(result, dict) else result

    # 3. 数据处理 (GPU -> CPU -> Numpy)
    # 对于 Offset 模型，pred 的形状为: (N, K, 4) (x, y, z, mask_prob)
    # 模型输出的点数
    num_kps = cfg.model.num_keypoints
    
    # 提取 coord, target_offset 和 target_mask
    coord = data_dict["coord"].cpu().numpy()  # (N, 3)
    target = data_dict["target"] # (N, K, 4)
    target_offset = target[..., :3].cpu().numpy()
    target_mask = target[..., 3].cpu().numpy()
    
    pred_offset = pred[..., :3].cpu().numpy()  # (N, K, 3)
    pred_mask = pred[..., 3].cpu().numpy()     # (N, K)

    # =============== 聚合点云偏移来得到关键点位置 ===============
    # 对于每个关键点 k，找到预测 mask 概率最大的那个点，或者用置信度超过阈值的点的预测坐标加权平均。
    # 这里我们采用一种简单的策略：取 mask_prob 最大的前 top_M 个点，用它们的坐标 + offset 作为预测的关键点坐标
    pred_kps = np.zeros((num_kps, 3))
    target_kps = np.zeros((num_kps, 3))
    
    # 获取归一化的 scale 和 centroid 进行逆变换还原绝对坐标
    if "scale" in data_dict:
        scale = data_dict["scale"].cpu().numpy()[0] # (1,) -> scalar
    else:
        scale = 1.0
        
    if "centroid" in data_dict:
        centroid = data_dict["centroid"].cpu().numpy()[0]
    else:
        centroid = np.zeros(3)

    # 逆归一化 coord
    true_coord = coord * scale + centroid
    
    # 传递参与计算的点索引及其权重以供可视化
    active_points_info = {}

    for k in range(num_kps):
        # ---- 计算预测关键点 ----
        # 获取第 k 个关键点的所有点的置信度
        k_probs = pred_mask[:, k]
        
        if args.agg_method == "argmax":
            best_idx_pred = np.argmax(k_probs)
            pred_kp_coord = true_coord[best_idx_pred] + pred_offset[best_idx_pred, k] * scale
            active_points_info[k] = {'indices': [best_idx_pred], 'probs': [k_probs[best_idx_pred]]}
        else: # weighted
            valid_mask = k_probs > args.mask_thresh
            if np.any(valid_mask):
                valid_indices_pred = np.where(valid_mask)[0]
                valid_probs = k_probs[valid_mask]
                valid_coords = true_coord[valid_mask]
                valid_offsets = pred_offset[valid_mask, k]
                # 计算候选点的预测关键点世界坐标
                candidate_kps = valid_coords + valid_offsets * scale
                # 通过概率归一化权重
                weights = valid_probs / np.sum(valid_probs)
                pred_kp_coord = np.sum(candidate_kps * weights[:, np.newaxis], axis=0)
                active_points_info[k] = {'indices': valid_indices_pred, 'probs': valid_probs}
            else:
                # 降级退回到 argmax
                best_idx_pred = np.argmax(k_probs)
                pred_kp_coord = true_coord[best_idx_pred] + pred_offset[best_idx_pred, k] * scale
                active_points_info[k] = {'indices': [best_idx_pred], 'probs': [k_probs[best_idx_pred]]}
                
        pred_kps[k] = pred_kp_coord
        
        # ---- 计算真实关键点 ----
        # 同样，在 GT mask 中找值为 1 的点（随便一个即可，因为它们回归的终点都一样）
        valid_indices = np.where(target_mask[:, k] > 0.5)[0]
        if len(valid_indices) > 0:
            best_idx_gt = valid_indices[0]
            target_kp_coord = true_coord[best_idx_gt] + target_offset[best_idx_gt, k] * scale
            target_kps[k] = target_kp_coord
        else:
            # 数据集中该关键点没有有效前景点！跳过或设为0
            target_kps[k] = pred_kps[k] # 用预测值充当防止画图时爆炸，或者可以填 nan

    # 计算距离误差
    dist = np.linalg.norm(pred_kps - target_kps, axis=-1)

    print(f"\n[{args.subset} set, item {idx}] ({data_dict.get('name', ['Unknown'])[0]})")
    for i in range(num_kps):
        print(f"  KP {i}: Erro = {dist[i]:.4f}mm | Pred = {pred_kps[i].round(4)} | GT = {target_kps[i].round(4)}")
    print(f"  --> Mean Error = {dist.mean():.4f}mm")

    # 可视化 (需要传真实空间的 coord)
    if args.visualize:
        visualize_single(true_coord, pred_kps, target_kps, args, num_kps, active_points_info)

def plot_batch_errors(all_errors, num_kps):
    """
    绘制关键点误差散点图
    layout: 2行3列 (针对6个关键点)
    """
    import matplotlib.pyplot as plt
    
    # 设置绘图风格
    plt.style.use('ggplot')
    
    # 创建画布，2行3列
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10))
    fig.suptitle('Keypoint Prediction Errors (Batch Inference)', fontsize=16)
    
    # 展平 axes 方便索引
    axes = axes.flatten()
    
    # 样本序号 (X轴)
    x = np.arange(all_errors.shape[0])
    
    for i in range(num_kps):
        if i >= len(axes): break # 防止关键点数量超过子图数量
        
        ax = axes[i]
        y = all_errors[:, i] # 第 i 个关键点的所有样本误差
        
        # 统计指标
        mean_val = np.mean(y)
        std_val = np.std(y)
        upper_limit = mean_val + 2 * std_val
        
        # 1. 绘制散点
        ax.scatter(x, y, alpha=0.6, s=10, c='blue', label='Sample Error')
        
        # 2. 绘制平均值虚线 (红色)
        ax.axhline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
        
        # 3. 绘制 2*标准差 虚线 (绿色)
        ax.axhline(upper_limit, color='green', linestyle='--', linewidth=2, label=f'Mean+2Std: {upper_limit:.4f}')
        
        # 标签设置
        ax.set_title(f'Keypoint {i}', fontsize=12)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Error (mm)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, which='both', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局防止重叠
    plt.show() # 弹出窗口

def inference_batch(cfg, model, dataset, args):
    """批量推理逻辑"""
    print(f"=> Start Batch Inference on [{args.subset}] set...")
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=cfg.data.batch_size if hasattr(cfg.data, "batch_size") else 1,
        shuffle=False, 
        num_workers=cfg.num_worker, 
        collate_fn=point_collate_fn,
        pin_memory=True
    )

    num_kps = cfg.model.num_keypoints
    all_errors = [] # 存储所有样本所有关键点的误差

    model.eval()
    with torch.no_grad():
        for i, data_dict in enumerate(tqdm(dataloader)):
            # GPU
            for key in data_dict:
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda(non_blocking=True)
            
            # Forward
            result = model(data_dict)
            pred = result["pred"] if isinstance(result, dict) else result
            
            target = data_dict["target"]
            
            # 提取 coord, target_offset 和 target_mask
            coord = data_dict["coord"]  # (N, 3)
            offset = data_dict["offset"]
            
            # (N, K, 4)
            pred_offset = pred[..., :3]
            pred_mask = pred[..., 3]
            target_offset = target[..., :3]
            target_mask = target[..., 3]
            
            # 由于是 batch 送入的，我们需要按样本对点云进行切割
            b = torch.zeros(offset[-1], dtype=torch.long, device=offset.device)
            if len(offset) > 1:
                b[offset[:-1]] = 1
            batch_idx = torch.cumsum(b, dim=0)
            
            num_samples = len(offset)
            
            # 从 data_dict 中取 scale (B,) 和 centroid (B, 3)
            scale = data_dict.get("scale", torch.ones(num_samples, device=coord.device))
            centroid = data_dict.get("centroid", torch.zeros(num_samples, 3, device=coord.device))
            
            pred_kps_batch = torch.zeros(num_samples, num_kps, 3, device=coord.device)
            target_kps_batch = torch.zeros(num_samples, num_kps, 3, device=coord.device)
            
            for b_idx in range(num_samples):
                # 获取该样本的 Mask
                sample_mask = (batch_idx == b_idx)
                
                # 获取当前样本的点、预测掩码、预测偏移
                s_coord = coord[sample_mask]
                s_pred_mask = pred_mask[sample_mask]
                s_pred_off = pred_offset[sample_mask]
                
                s_target_mask = target_mask[sample_mask]
                s_target_off = target_offset[sample_mask]
                
                s_scale = scale[b_idx]
                s_centroid = centroid[b_idx]
                
                s_true_coord = s_coord * s_scale + s_centroid
                
                for k in range(num_kps):
                    k_probs = s_pred_mask[:, k]
                    if len(k_probs) == 0:
                        continue
                        
                    if args.agg_method == "argmax":
                        best_idx_pred = torch.argmax(k_probs)
                        pred_kp_coord = s_true_coord[best_idx_pred] + s_pred_off[best_idx_pred, k] * s_scale
                    else: # weighted
                        valid_mask = k_probs > args.mask_thresh
                        if torch.any(valid_mask):
                            valid_probs = k_probs[valid_mask]
                            valid_coords = s_true_coord[valid_mask]
                            valid_offsets = s_pred_off[valid_mask, k]
                            
                            candidate_kps = valid_coords + valid_offsets * s_scale
                            weights = valid_probs / torch.sum(valid_probs)
                            pred_kp_coord = torch.sum(candidate_kps * weights.unsqueeze(1), dim=0)
                        else:
                            best_idx_pred = torch.argmax(k_probs)
                            pred_kp_coord = s_true_coord[best_idx_pred] + s_pred_off[best_idx_pred, k] * s_scale
                            
                    pred_kps_batch[b_idx, k] = pred_kp_coord
                    
                    valid_indices = torch.where(s_target_mask[:, k] > 0.5)[0]
                    if len(valid_indices) > 0:
                        best_idx_gt = valid_indices[0]
                        target_kp_coord = s_true_coord[best_idx_gt] + s_target_off[best_idx_gt, k] * s_scale
                        target_kps_batch[b_idx, k] = target_kp_coord
                    else:
                        target_kps_batch[b_idx, k] = pred_kp_coord
            
            # 计算欧氏距离
            dist = torch.norm(pred_kps_batch - target_kps_batch, p=2, dim=-1) # (B, K)
            all_errors.append(dist.cpu().numpy())

    # Concatenate all batches: (Total_Samples, K)
    all_errors = np.concatenate(all_errors, axis=0)
    
    # Statistics
    mean_per_kp = np.mean(all_errors, axis=0)
    std_per_kp = np.std(all_errors, axis=0)
    total_mean = np.mean(all_errors)

    print("\n====== Batch Inference Statistics ======")
    print(f"Total Samples: {all_errors.shape[0]}")
    print("-" * 65)
    print(f"{'Keypoint ID':<15} | {'Mean Error':<20} | {'Std Dev':<20}")
    print("-" * 65)
    for i in range(num_kps):
        print(f"KP {i:<12} | {mean_per_kp[i]:.5f}            | {std_per_kp[i]:.5f}")
    print("-" * 65)
    print(f"{'OVERALL':<15} | {total_mean:.5f}")
    print("-" * 65)

    # [新增] 调用绘图函数
    print("=> Plotting error distribution...")
    plot_batch_errors(all_errors, num_kps)

def main():
    args = get_args()
    
    # 1. 加载配置
    cfg = Config.fromfile(args.config_file)
    if args.options:
        cfg.merge_from_dict(args.options)
    
    # 2. 构建模型
    model = setup_model(cfg, args.weights)
    
    # 3. 构建数据集
    if args.subset == "all":
        dataset_list = []
        # 遍历常用的三个数据集划分
        for split in ["train", "val", "test"]:
            if split in cfg.data:
                dataset_cfg = cfg.data[split]
                dataset_cfg.data_root = cfg.data_root # 确保 data_root 被正确传递
                dataset_list.append(build_dataset(dataset_cfg))
        
        if len(dataset_list) == 0:
            raise ValueError("No valid dataset configuration found for 'train', 'val', or 'test'.")
            
        # 使用 ConcatDataset 合并多个数据集
        dataset = ConcatDataset(dataset_list)
        print(f"=> Loaded {len(dataset)} samples from ALL (train+val+test) sets.")
        
    else:
        # 原有逻辑：仅加载单个子集
        if args.subset not in cfg.data:
            raise ValueError(f"Subset {args.subset} not found in config.data")
        
        dataset_cfg = cfg.data[args.subset]
        dataset_cfg.data_root = cfg.data_root 
        dataset = build_dataset(dataset_cfg)
        print(f"=> Loaded {len(dataset)} samples from {args.subset} set.")

    # 4. 执行推理
    if args.idx != -1:
        # 单样本模式
        inference_single_sample(cfg, model, dataset, args)
    else:
        # 批量模式
        inference_batch(cfg, model, dataset, args)


"""
## 基于 OffsetKeypointPTV3 模型的推理示例
python tools/infer_offset.py \
    --config-file configs/my_dataset/offset_keypoint_ptv3.py \
    --weights exp/offset_keypoint_ptv3_0512/model/model_best.pth \
    --subset all \
    --idx 1 \
    --visualize
python tools/infer_offset.py \
    --config-file configs/my_dataset/offset_keypoint_ptv3.py \
    --weights exp/offset_keypoint_ptv3_0512/model/model_best.pth \
    --subset all \
    --idx -1 \
    --visualize \
    --agg-method weighted \
    --mask-thresh 0.8 
#全样本误差 24.47934
#训练集误差 13.83127
#验证集误差 50.20841
#测试集误差 48.25328
### 全样本推理
====== Batch Inference Statistics ======
Total Samples: 152
-----------------------------------------------------------------
Keypoint ID     | Mean Error           | Std Dev             
-----------------------------------------------------------------
KP 0            | 27.62905            | 13.80553
KP 1            | 36.48778            | 25.14210
KP 2            | 32.77974            | 30.66247
KP 3            | 33.92687            | 25.01572
KP 4            | 29.42157            | 24.30774
KP 5            | 28.05849            | 14.97811
-----------------------------------------------------------------
OVERALL         | 31.38391
-----------------------------------------------------------------
### 训练集推理
====== Batch Inference Statistics ======
Total Samples: 106
-----------------------------------------------------------------
Keypoint ID     | Mean Error           | Std Dev             
-----------------------------------------------------------------
KP 0            | 23.01698            | 10.17486
KP 1            | 26.01616            | 12.66112
KP 2            | 18.54890            | 9.07661
KP 3            | 24.08994            | 12.70826
KP 4            | 19.71862            | 8.26604
KP 5            | 25.08293            | 11.07103
-----------------------------------------------------------------
OVERALL         | 22.74559
-----------------------------------------------------------------
### 验证集推理
====== Batch Inference Statistics ======
Total Samples: 30
-----------------------------------------------------------------
Keypoint ID     | Mean Error           | Std Dev             
-----------------------------------------------------------------
KP 0            | 40.01009            | 13.97123
KP 1            | 64.34909            | 26.55335
KP 2            | 66.30247            | 40.15957
KP 3            | 58.62809            | 34.18132
KP 4            | 54.97828            | 30.42426
KP 5            | 39.37823            | 19.12898
-----------------------------------------------------------------
OVERALL         | 53.94105
-----------------------------------------------------------------
### 测试集推理
====== Batch Inference Statistics ======
Total Samples: 16
-----------------------------------------------------------------
Keypoint ID     | Mean Error           | Std Dev             
-----------------------------------------------------------------
KP 0            | 38.89864            | 18.73992
KP 1            | 54.32608            | 29.74505
KP 2            | 60.49099            | 28.61563
KP 3            | 51.96609            | 29.44759
KP 4            | 51.00819            | 26.54192
KP 5            | 35.85108            | 16.25771
-----------------------------------------------------------------
OVERALL         | 48.75685
-----------------------------------------------------------------


## 基于 OffsetKeypointSwin3D 模型的推理示例
python tools/infer_offset.py \
    --config-file configs/my_dataset/offset_keypoint_swin3d.py \
    --weights exp/offset_keypoint_swin3d_0512/model/model_best.pth \
    --subset all \
    --idx -1 \
    --visualize \
    --agg-method weighted \
    --mask-thresh 0.8
### 全样本推理
====== Batch Inference Statistics ======
Total Samples: 152
-----------------------------------------------------------------
Keypoint ID     | Mean Error           | Std Dev             
-----------------------------------------------------------------
KP 0            | 57.87032            | 33.15565
KP 1            | 37.99462            | 26.44286
KP 2            | 41.14744            | 32.10968
KP 3            | 40.26004            | 25.71033
KP 4            | 45.39358            | 26.83938
KP 5            | 40.85643            | 22.84280
-----------------------------------------------------------------
OVERALL         | 43.92041
-----------------------------------------------------------------
### 训练集推理
====== Batch Inference Statistics ======
Total Samples: 106
-----------------------------------------------------------------
Keypoint ID     | Mean Error           | Std Dev             
-----------------------------------------------------------------
KP 0            | 49.95384            | 30.73011
KP 1            | 29.07417            | 15.12297
KP 2            | 30.49793            | 14.66652
KP 3            | 34.24857            | 19.06088
KP 4            | 37.83554            | 18.01805
KP 5            | 40.78370            | 21.49480
-----------------------------------------------------------------
OVERALL         | 37.06563
-----------------------------------------------------------------
### 验证集推理
====== Batch Inference Statistics ======
Total Samples: 30
-----------------------------------------------------------------
Keypoint ID     | Mean Error           | Std Dev             
-----------------------------------------------------------------
KP 0            | 73.82295            | 32.46471
KP 1            | 66.05435            | 35.00121
KP 2            | 75.60102            | 50.09143
KP 3            | 57.01134            | 34.45636
KP 4            | 61.50737            | 32.32346
KP 5            | 46.60015            | 28.23231
-----------------------------------------------------------------
OVERALL         | 63.43287
-----------------------------------------------------------------
### 测试集推理
====== Batch Inference Statistics ======
Total Samples: 16
-----------------------------------------------------------------
Keypoint ID     | Mean Error           | Std Dev             
-----------------------------------------------------------------
KP 0            | 70.69192            | 33.18789
KP 1            | 52.62598            | 30.70875
KP 2            | 60.98167            | 25.91590
KP 3            | 54.33746            | 31.59367
KP 4            | 59.30550            | 28.79784
KP 5            | 40.57346            | 18.18356
-----------------------------------------------------------------
OVERALL         | 56.41934
-----------------------------------------------------------------

## 基于OctFormer模型的推理示例
python tools/infer_offset.py \
    --config-file configs/my_dataset/offset_keypoint_octformer.py \
    --weights exp/offset_keypoint_octformer_0512/model/model_best.pth \
    --subset train \
    --idx 25 \
    --visualize \
    --agg-method weighted \
    --mask-thresh 0.8
====== Batch Inference Statistics ======
Total Samples: 152
-----------------------------------------------------------------
Keypoint ID     | Mean Error           | Std Dev             
-----------------------------------------------------------------
KP 0            | 14.72275            | 20.08362
KP 1            | 24.54355            | 28.27561
KP 2            | 26.55748            | 33.80216
KP 3            | 20.92421            | 23.73089
KP 4            | 20.50109            | 24.36789
KP 5            | 22.45653            | 17.22614
-----------------------------------------------------------------
OVERALL         | 21.61760
-----------------------------------------------------------------
### 训练集推理
====== Batch Inference Statistics ======
Total Samples: 106
-----------------------------------------------------------------
Keypoint ID     | Mean Error           | Std Dev             
-----------------------------------------------------------------
KP 0            | 4.90193            | 2.22768
KP 1            | 10.18732            | 5.54086
KP 2            | 10.16163            | 3.90413
KP 3            | 9.77647            | 4.62208
KP 4            | 7.93249            | 3.73575
KP 5            | 14.24255            | 7.28289
-----------------------------------------------------------------
OVERALL         | 9.53373
-----------------------------------------------------------------
"""
if __name__ == "__main__":
    main()
