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
from datetime import datetime

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
    parser.add_argument("--sphere-radius", type=float, default=20, help="真实关键点(球)的半径")
    parser.add_argument("--cube-size", type=float, default=30, help="预测关键点(正方体)的边长")
    parser.add_argument("--cube-wire-radius", type=float, default=1.5, help="预测关键点线框正方体的圆柱线半径")
    parser.add_argument("--point-size", type=float, default=2.0, help="Open3D 可视化时点云的点大小")
    parser.add_argument("--agg-method", choices=["argmax", "weighted"], default="argmax", help="关键点聚合策略：'argmax'（置信度最高点） 或 'weighted'（置信度大于阈值的点加权平均）")
    parser.add_argument("--mask-thresh", type=float, default=0.5, help="当 agg-method 为 weighted 时的掩码阈值")
    parser.add_argument("--save-dir", default=None, help="结果保存路径 (可选)")
    parser.add_argument("--save-keypoints", action="store_true", help="是否保存预测关键点和真实关键点坐标")
    parser.add_argument("--save-name", default=None, help="保存坐标的 txt 文件名；默认自动生成")
    
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

def get_sample_names(data_dict, start_idx, batch_size):
    """从 batch 中提取样本名，失败时回退到 sample_序号。"""
    names = data_dict.get("name", None)
    fallback = [f"sample_{start_idx + i}" for i in range(batch_size)]
    if names is None:
        return fallback
    if isinstance(names, np.ndarray):
        names = names.tolist()
    if isinstance(names, torch.Tensor):
        names = names.detach().cpu().tolist()
    if isinstance(names, (list, tuple)):
        names = [str(name) for name in names]
        if len(names) == batch_size:
            return names
        return names + fallback[len(names):]
    if batch_size == 1:
        return [str(names)]
    return fallback

def sanitize_filename(name):
    keep = []
    for ch in str(name):
        keep.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "_")
    return "".join(keep).strip("_") or "keypoints"

def build_keypoint_save_path(args, cfg, mode):
    if not args.save_keypoints:
        return None
    if args.save_dir is None:
        raise ValueError("--save-keypoints requires --save-dir to specify the output directory.")
    os.makedirs(args.save_dir, exist_ok=True)
    if args.save_name:
        filename = args.save_name
    else:
        model_name = sanitize_filename(cfg.model.type)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{args.subset}_{mode}_{timestamp}_keypoints.txt"
    if not filename.endswith(".txt"):
        filename += ".txt"
    return os.path.join(args.save_dir, sanitize_filename(filename))

def save_keypoint_records(records, save_path, args, cfg, prediction_type):
    if save_path is None:
        return
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("# Keypoint prediction export\n")
        f.write(f"# prediction_type={prediction_type}\n")
        f.write(f"# config_file={args.config_file}\n")
        f.write(f"# weights={args.weights}\n")
        f.write(f"# subset={args.subset}\n")
        f.write(f"# model_type={cfg.model.type}\n")
        f.write(f"# agg_method={args.agg_method}\n")
        f.write(f"# mask_thresh={args.mask_thresh}\n")
        f.write("# columns are tab-separated\n")
        f.write("sample_index\tsample_id\tkeypoint_id\tpred_x\tpred_y\tpred_z\tgt_x\tgt_y\tgt_z\terror\n")
        for record in records:
            pred = np.asarray(record["pred"], dtype=np.float64)
            target = np.asarray(record["target"], dtype=np.float64)
            errors = np.linalg.norm(pred - target, axis=-1)
            for kp_idx, (pred_xyz, target_xyz, error) in enumerate(zip(pred, target, errors)):
                f.write(
                    f"{record['sample_index']}\t{record['sample_id']}\t{kp_idx}\t"
                    f"{pred_xyz[0]:.8f}\t{pred_xyz[1]:.8f}\t{pred_xyz[2]:.8f}\t"
                    f"{target_xyz[0]:.8f}\t{target_xyz[1]:.8f}\t{target_xyz[2]:.8f}\t"
                    f"{error:.8f}\n"
                )
    print(f"=> Saved keypoint predictions and GT to: {save_path}")

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


def is_valid_point(point):
    point = np.asarray(point)
    return point.shape == (3,) and np.all(np.isfinite(point))


def align_cylinder(p1, p2, radius, color):
    """
    生成连接两点 (p1, p2) 的具有实体厚度的圆柱体
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    v = p2 - p1
    length = np.linalg.norm(v)
    if length < 1e-8:
        return None
    
    # 创建基础圆柱体 (默认沿 Z 轴)
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    
    # 计算旋转矩阵，将 Z 轴对齐到向量 v 的方向
    z_axis = np.array([0, 0, 1])
    dir_vec = v / length
    axis = np.cross(z_axis, dir_vec)
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm < 1e-6:
        # 平行的情况 (同向或反向)
        if np.dot(z_axis, dir_vec) < 0:
            R = cylinder.get_rotation_matrix_from_xyz((np.pi, 0, 0))
            cylinder.rotate(R, center=(0,0,0))
    else:
        # 一般情况，计算旋转轴和旋转角
        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(z_axis, dir_vec), -1.0, 1.0))
        R = cylinder.get_rotation_matrix_from_axis_angle(axis * angle)
        cylinder.rotate(R, center=(0,0,0))
    
    # 平移到两点的中点
    midpoint = (p1 + p2) / 2.0
    cylinder.translate(midpoint)
    cylinder.paint_uniform_color(color)
    cylinder.compute_vertex_normals()
    
    return cylinder

def create_thick_wireframe_box(center, size, color, cylinder_radius=1.5):
    """
    创建由 12 根圆柱体拼接而成的实体线框正方体
    """
    half_s = size / 2.0
    # 定义正方体的 8 个顶点相对坐标，并平移到中心
    points = np.array([
        [-half_s, -half_s, -half_s], [half_s, -half_s, -half_s],
        [-half_s, half_s, -half_s],  [half_s, half_s, -half_s],
        [-half_s, -half_s, half_s],  [half_s, -half_s, half_s],
        [-half_s, half_s, half_s],   [half_s, half_s, half_s]
    ]) + center
    
    # 12 条边的顶点索引
    edges = [
        [0, 1], [0, 2], [1, 3], [2, 3], # 底面
        [4, 5], [4, 6], [5, 7], [6, 7], # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7]  # 侧面
    ]
    
    # 创建一个空的 TriangleMesh 用于合并所有圆柱体
    thick_wireframe = o3d.geometry.TriangleMesh()
    
    for edge in edges:
        p1, p2 = points[edge[0]], points[edge[1]]
        cylinder = align_cylinder(p1, p2, radius=cylinder_radius, color=color)
        if cylinder is not None:
            thick_wireframe += cylinder # 合并网格
        
    thick_wireframe.compute_vertex_normals()
    return thick_wireframe

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
        color = np.array(colors[i])
        # 真实值：圆球 (Sphere)
        try:
            if target_kps is not None and not np.any(np.isnan(target_kps[i])):
                sphere = create_colored_mesh('sphere', np.asarray(target_kps[i]), colors[i], args.sphere_radius)
                geometries.append(sphere)
        except Exception:
            # 容错：若 target_kps 不可用则忽略
            pass

        # 预测值 (Pred)：具有实体厚度的加粗线框
        try:
            if pred_kps is not None and is_valid_point(pred_kps[i]):
                thick_cube = create_thick_wireframe_box(
                    np.asarray(pred_kps[i]),
                    args.cube_size,
                    color,
                    cylinder_radius=args.cube_wire_radius,
                )
                geometries.append(thick_cube)
        except Exception as exc:
            print(f"=> Warning: failed to draw predicted wireframe cube for KP {i}: {exc}")
        # [新增] 误差辅助线：只有当 GT 和 Pred 都存在时才画线
        if target_kps is not None and pred_kps is not None and is_valid_point(target_kps[i]) and is_valid_point(pred_kps[i]):
            line_points = [target_kps[i], pred_kps[i]]
            lines = [[0, 1]]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(line_points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            # 连线统一用鲜艳的红色或对比色，方便观察偏移方向
            line_set.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]]) 
            geometries.append(line_set)

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
    opt.line_width = 3.0 # 设置线框的粗细
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
            # 缺失 GT 时保留 NaN，避免把预测值当成 GT 产生人为的零误差。
            target_kps[k] = np.nan

    # 计算距离误差
    dist = np.linalg.norm(pred_kps - target_kps, axis=-1)

    print(f"\n[{args.subset} set, item {idx}] ({data_dict.get('name', ['Unknown'])[0]})")
    for i in range(num_kps):
        print(f"  KP {i}: Erro = {dist[i]:.4f}mm | Pred = {pred_kps[i].round(4)} | GT = {target_kps[i].round(4)}")
    print(f"  --> Mean Error = {np.nanmean(dist):.4f}mm")

    if args.save_keypoints:
        sample_name = get_sample_names(data_dict, idx, 1)[0]
        save_path = build_keypoint_save_path(args, cfg, "single")
        save_keypoint_records(
            [dict(sample_index=idx, sample_id=sample_name, pred=pred_kps, target=target_kps)],
            save_path,
            args,
            cfg,
            prediction_type="offset",
        )

    # 可视化 (需要传真实空间的 coord)
    if args.visualize:
        visualize_single(true_coord, pred_kps, target_kps, args, num_kps, active_points_info)

def plot_batch_errors(all_errors, num_kps):
    """
    绘制关键点误差散点图
    layout: 2行3列 (针对6个关键点)
    """
    import matplotlib.pyplot as plt

    # 可视化前随机打乱样本顺序，避免原始数据顺序影响散点分布观感。
    if all_errors.shape[0] > 1:
        all_errors = all_errors[np.random.permutation(all_errors.shape[0])]
    
    # 设置绘图风格
    plt.style.use('ggplot')
    
    # 创建画布，2行3列
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10))
    fig.suptitle('Keypoint Prediction Errors (Batch Inference)', fontsize=16)
    
    # 展平 axes 方便索引
    axes = axes.flatten()
    
    # 打乱后的样本序号 (X轴)
    x = np.arange(all_errors.shape[0])
    
    for i in range(num_kps):
        if i >= len(axes): break # 防止关键点数量超过子图数量
        
        ax = axes[i]
        y = all_errors[:, i] # 第 i 个关键点的所有样本误差
        
        # 统计指标
        mean_val = np.nanmean(y)
        std_val = np.nanstd(y)
        upper_limit = mean_val + 2 * std_val
        
        # 1. 绘制散点
        valid = np.isfinite(y)
        ax.scatter(x[valid], y[valid], alpha=0.6, s=10, c='blue', label='Sample Error')
        
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
    # --- 新增功能：保存高分辨率图片 ---
    save_path = "results/batch_keypoint_errors.svg"
    dpi=1200
    if save_path is not None:
        # bbox_inches='tight' 用于去除图片周围多余的白边
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"图表已成功保存至: {save_path} (DPI: {dpi})")
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
    keypoint_records = []
    sample_cursor = 0

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
            target_kps_batch = torch.full((num_samples, num_kps, 3), float("nan"), device=coord.device)
            
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
            
            # 计算欧氏距离
            dist = torch.norm(pred_kps_batch - target_kps_batch, p=2, dim=-1) # (B, K)
            all_errors.append(dist.cpu().numpy())

            if args.save_keypoints:
                sample_names = get_sample_names(data_dict, sample_cursor, num_samples)
                pred_kps_np = pred_kps_batch.cpu().numpy()
                target_kps_np = target_kps_batch.cpu().numpy()
                for local_idx in range(num_samples):
                    keypoint_records.append(
                        dict(
                            sample_index=sample_cursor + local_idx,
                            sample_id=sample_names[local_idx],
                            pred=pred_kps_np[local_idx],
                            target=target_kps_np[local_idx],
                        )
                    )
            sample_cursor += num_samples

    # Concatenate all batches: (Total_Samples, K)
    all_errors = np.concatenate(all_errors, axis=0)
    
    # Statistics
    mean_per_kp = np.nanmean(all_errors, axis=0)
    std_per_kp = np.nanstd(all_errors, axis=0)
    total_mean = np.nanmean(all_errors)

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

    if args.save_keypoints:
        save_path = build_keypoint_save_path(args, cfg, "batch")
        save_keypoint_records(keypoint_records, save_path, args, cfg, prediction_type="offset")

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
    --mask-thresh 0.8 \
    --save-keypoints \
    --save-dir outputs/keypoints \
    --save-name  ptv3_offset.txt
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
    --mask-thresh 0.8 \
    --save-keypoints \
    --save-dir outputs/keypoints \
    --save-name  swin3d_offset.txt
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

## 基于OctFormer模型的推理示例（R=300)
python tools/infer_offset.py \
    --config-file configs/my_dataset/offset_keypoint_octformer.py \
    --weights exp/offset_keypoint_octformer_0512/model/model_best.pth \
    --subset all \
    --idx -1 \
    --visualize \
    --agg-method weighted \
    --mask-thresh 0.5 \
    --save-keypoints \
    --save-dir outputs/keypoints \
    --save-name  octformer_offset.txt
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

## octformer设置半径R=100
python tools/infer_offset.py \
    --config-file configs/my_dataset/offset_keypoint_octformer.py \
    --weights exp/offset_keypoint_octformer_0529/model/model_best.pth \
    --subset all \
    --idx -1 \
    --visualize \
    --agg-method weighted \
    --mask-thresh 0.8
====== Batch Inference Statistics ======
Total Samples: 152
-----------------------------------------------------------------
Keypoint ID     | Mean Error           | Std Dev             
-----------------------------------------------------------------
KP 0            | 426.04031            | 252.91177
KP 1            | 177.47498            | 171.32773
KP 2            | 172.42622            | 174.92284
KP 3            | 351.86066            | 203.70782
KP 4            | 514.02325            | 277.95557
KP 5            | 15.55502            | 18.01666
-----------------------------------------------------------------
OVERALL         | 276.23010
-----------------------------------------------------------------

## octformer设置半径R=200(虽然效果好，但是过拟合明显)
python tools/infer_offset.py \
    --config-file configs/my_dataset/offset_keypoint_octformer.py \
    --weights exp/offset_keypoint_octformer_0529v1/model/model_best.pth \
    --subset all \
    --idx 60 \
    --visualize \
    --agg-method weighted \
    --mask-thresh 0.8
====== Batch Inference Statistics ======
Total Samples: 152
-----------------------------------------------------------------
Keypoint ID     | Mean Error           | Std Dev             
-----------------------------------------------------------------
KP 0            | 13.30709            | 19.14559
KP 1            | 19.94459            | 31.65851
KP 2            | 22.07256            | 36.70895
KP 3            | 18.97061            | 25.74864
KP 4            | 17.50271            | 27.29051
KP 5            | 14.86118            | 18.07594
-----------------------------------------------------------------
OVERALL         | 17.77645
-----------------------------------------------------------------

## octformer设置半径R=400
python tools/infer_offset.py \
    --config-file configs/my_dataset/offset_keypoint_octformer.py \
    --weights exp/offset_keypoint_octformer_0530/model/model_best.pth \
    --subset all \
    --idx -1 \
    --visualize \
    --agg-method weighted \
    --mask-thresh 0.8
====== Batch Inference Statistics ======
Total Samples: 152
-----------------------------------------------------------------
Keypoint ID     | Mean Error           | Std Dev             
-----------------------------------------------------------------
KP 0            | 22.99101            | 21.97149
KP 1            | 26.61954            | 27.52293
KP 2            | 28.30157            | 33.79631
KP 3            | 25.39908            | 24.98058
KP 4            | 26.98810            | 24.12887
KP 5            | 23.97778            | 18.13627
-----------------------------------------------------------------
OVERALL         | 25.71285
-----------------------------------------------------------------
"""
if __name__ == "__main__":
    main()
