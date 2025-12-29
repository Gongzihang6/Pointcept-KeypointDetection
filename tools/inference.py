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

def get_args():
    parser = argparse.ArgumentParser(description="Pointcept Keypoint Inference")
    parser.add_argument("--config-file", default="configs/my_dataset/keypoint_ptv3.py", help="配置文件路径")
    parser.add_argument("--options", nargs="+", action=DictAction, help="覆盖配置文件的参数")
    parser.add_argument("--weights", default=None, required=True, help="模型权重文件路径 (.pth)")
    parser.add_argument("--subset", default="val", choices=["train", "val", "test"], help="数据集划分")
    parser.add_argument("--idx", type=int, default=-1, help="单样本索引。如果为 -1，则进行批量推理")
    
    # 可视化参数
    parser.add_argument("--visualize", action="store_true", help="是否开启 Open3D 可视化 (仅单样本模式有效)")
    parser.add_argument("--sphere-radius", type=float, default=0.05, help="真实关键点(球)的半径")
    parser.add_argument("--cube-size", type=float, default=0.08, help="预测关键点(正方体)的边长")
    parser.add_argument("--point-size", type=float, default=2.0, help="Open3D 可视化时点云的点大小")
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

def visualize_single(coord, pred_kps, target_kps, args, num_kps):
    """使用 Open3D 可视化 (支持调整点大小)"""
    print(f"=> Visualizing... (Point Size: {args.point_size})")
    geometries = []

    # 1. 点云 (灰色)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.paint_uniform_color([0.7, 0.7, 0.7]) # 灰色点云
    geometries.append(pcd)

    # 2. 关键点颜色映射
    cmap = plt.get_cmap("jet")
    colors = [cmap(i / (num_kps - 1 if num_kps > 1 else 1))[:3] for i in range(num_kps)]

    # 3. 绘制关键点
    for i in range(num_kps):
        # 真实值：圆球 (Sphere)
        if target_kps is not None:
            sphere = create_colored_mesh('sphere', target_kps[i], colors[i], args.sphere_radius)
            geometries.append(sphere)
        
        # 预测值：正方体 (Cube)
        cube = create_colored_mesh('box', pred_kps[i], colors[i], args.cube_size)
        geometries.append(cube)

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
        # 兼容不同的返回格式 (有的模型返回字典，有的返回 Tensor)
        pred = result["pred"] if isinstance(result, dict) else result

    # 3. 数据后处理 (GPU -> CPU -> Numpy)
    # pred shape: (1, K, 3) -> (K, 3)
    # target shape: (1, K, 3) -> (K, 3)
    num_kps = cfg.model.num_keypoints
    pred = pred.view(-1, num_kps, 3).cpu().numpy()[0]
    
    target = None
    if "target" in data_dict:
        target = data_dict["target"].view(-1, num_kps, 3).cpu().numpy()[0]

    # 获取点云坐标用于可视化 (优先用原始 coord，如果没有则用 grid_coord * grid_size)
    coord = data_dict["coord"].cpu().numpy()
    
    # 4. 计算误差 (逆归一化)
    scale = 1.0
    if "scale" in data_dict:
        scale = data_dict["scale"].cpu().numpy()[0] # (1,) -> scalar
    elif "grid_size" in data_dict:
        # 如果没有 scale 只有 grid_size，且 target 是体素坐标，则用 grid_size
        scale = data_dict["grid_size"]
        if isinstance(scale, torch.Tensor): scale = scale.item()

    print(f"\n====== Inference Result [Sample IDX: {idx}] ======")
    print(f"Scale Factor: {scale}")

    if target is not None:
        # 计算欧氏距离
        # 注意：pred 和 target 目前通常是在归一化坐标系下
        diff = np.linalg.norm(pred - target, axis=-1) # (K,)
        
        # 逆归一化到原始物理尺度
        real_diff = diff * scale 

        print("-" * 40)
        print(f"{'Keypoint ID':<15} | {'Error (Original Scale)':<25}")
        print("-" * 40)
        for i in range(num_kps):
            print(f"KP {i:<12} | {real_diff[i]:.4f}")
        print("-" * 40)
        print(f"Mean Error      | {np.mean(real_diff):.4f}")
        print("-" * 40)
    
    # 5. 可视化
    if args.visualize:
        # 坐标通常也需要缩放以便可视化正确（如果 coord 也是归一化的）
        # 这里我们直接画归一化空间下的，或者全部乘 scale
        # 为了方便观察相对位置，直接画归一化空间下的即可，只要 scale 一致
        visualize_single(coord, pred, target, args, num_kps)


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
        ax.set_ylabel('Error (m)')
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
            
            # Reshape (B, K, 3)
            pred = pred.view(-1, num_kps, 3)
            target = target.view(-1, num_kps, 3)
            
            # Calc Distance in Normalized Space
            dist = torch.norm(pred - target, p=2, dim=-1) # (B, K)
            
            # Inverse Normalization
            if "scale" in data_dict:
                scale = data_dict["scale"] # (B,)
                if scale.ndim == 1: scale = scale.view(-1, 1)
                dist = dist * scale
            elif "grid_size" in data_dict:
                # Fallback logic
                g = data_dict["grid_size"]
                dist = dist * g

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
    # 注意：只构建 args.subset 指定的那一部分 (train/val/test)
    if args.subset not in cfg.data:
        raise ValueError(f"Subset {args.subset} not found in config.data")
    
    dataset_cfg = cfg.data[args.subset]
    dataset_cfg.data_root = cfg.data_root # 确保 data_root 被正确传递
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

## 基于 Pointcept-OctFormer 模型的推理脚本
python tools/inference.py \
    --config-file configs/my_dataset/keypoint_octformer.py \
    --weights exp/keypoint_octformer/model/model_best.pth \
    --subset test \
    --idx 1 \
    --visualize \
    --sphere-radius 0.02 \
    --cube-size 0.02
====== Batch Inference Statistics ======
Total Samples: 28
-----------------------------------------------------------------
Keypoint ID     | Mean Error           | Std Dev             
-----------------------------------------------------------------
KP 0            | 33.27346            | 16.75899
KP 1            | 26.27490            | 15.12749
KP 2            | 31.13291            | 19.68741
KP 3            | 27.60023            | 18.44568
KP 4            | 32.59894            | 17.00893
KP 5            | 38.96297            | 19.05381
-----------------------------------------------------------------
OVERALL         | 31.64057
-----------------------------------------------------------------

## 基于 Pointcept-PTv1 模型的推理脚本
export PYTHONPATH=.
python tools/inference.py \
    --config-file configs/my_dataset/keypoint_ptv1.py \
    --weights exp/keypoint_ptv1/model/model_best.pth \
    --subset test \
    --idx 10 \
    --visualize \
    --sphere-radius 0.02 \
    --cube-size 0.02
====== Batch Inference Statistics ======
Total Samples: 28
-----------------------------------------------------------------
Keypoint ID     | Mean Error           | Std Dev             
-----------------------------------------------------------------
KP 0            | 26.13216            | 11.06119
KP 1            | 24.93169            | 17.54638
KP 2            | 31.74557            | 21.16514
KP 3            | 24.36304            | 16.39521
KP 4            | 22.96082            | 11.83866
KP 5            | 32.36544            | 11.43148
-----------------------------------------------------------------
OVERALL         | 27.08312
-----------------------------------------------------------------

## 基于 Pointcept-PTv2 模型的推理脚本
python tools/inference.py \
    --config-file configs/my_dataset/keypoint_ptv2.py \
    --weights exp/keypoint_ptv2/model/model_best.pth \
    --subset train \
    --idx -1 \
    --visualize \
    --sphere-radius 0.02 \
    --cube-size 0.02
====== Batch Inference Statistics ======
Total Samples: 28
-----------------------------------------------------------------
Keypoint ID     | Mean Error           | Std Dev             
-----------------------------------------------------------------
KP 0            | 27.17042            | 15.76558
KP 1            | 21.63920            | 18.46684
KP 2            | 30.25765            | 20.59952
KP 3            | 25.22526            | 18.56907
KP 4            | 25.64672            | 13.56527
KP 5            | 31.95988            | 13.46944
-----------------------------------------------------------------
OVERALL         | 26.98319
-----------------------------------------------------------------

## 基于 Pointcept-PTv3 模型的推理脚本
python tools/inference.py \
    --config-file configs/my_dataset/keypoint_ptv3.py \
    --weights exp/keypoint_ptv3/model/model_best.pth \
    --subset test \
    --idx -1 \
    --visualize \
    --sphere-radius 0.02 \
    --cube-size 0.02
====== Batch Inference Statistics ======
Total Samples: 28
-----------------------------------------------------------------
Keypoint ID     | Mean Error           | Std Dev             
-----------------------------------------------------------------
KP 0            | 22.65371            | 13.36939
KP 1            | 21.20184            | 13.95168
KP 2            | 30.23071            | 20.58898
KP 3            | 27.92994            | 20.37176
KP 4            | 32.97409            | 18.85650
KP 5            | 34.39021            | 21.52718
-----------------------------------------------------------------
OVERALL         | 28.23009
-----------------------------------------------------------------


## 基于 Swin3D 模型的推理脚本
python tools/inference.py \
    --config-file configs/my_dataset/keypoint_swin3d.py \
    --weights exp/keypoint_swin3d/model/model_best.pth \
    --subset test \
    --idx 10 \
    --visualize \
    --sphere-radius 0.02 \
    --cube-size 0.02
====== Batch Inference Statistics ======
Total Samples: 28
-----------------------------------------------------------------
Keypoint ID     | Mean Error           | Std Dev             
-----------------------------------------------------------------
KP 0            | 19.49621            | 11.82266
KP 1            | 21.12213            | 17.46186
KP 2            | 31.11461            | 21.69423
KP 3            | 25.54762            | 19.42591
KP 4            | 23.38527            | 13.12805
KP 5            | 25.49153            | 12.77503
-----------------------------------------------------------------
OVERALL         | 24.35956
-----------------------------------------------------------------

## 基于 StratifiedTransformer 模型的推理脚本
python tools/inference.py \
    --config-file configs/my_dataset/keypoint_stratified_transformer.py \
    --weights exp/keypoint_stratified_transformer/model/model_best.pth \
    --subset test \
    --idx 10 \
    --visualize \
    --sphere-radius 0.02 \
    --cube-size 0.02
====== Batch Inference Statistics ======
Total Samples: 28
-----------------------------------------------------------------
Keypoint ID     | Mean Error           | Std Dev             
-----------------------------------------------------------------
KP 0            | 22.32730            | 11.79639
KP 1            | 21.55352            | 17.45925
KP 2            | 30.36640            | 19.86914
KP 3            | 25.84641            | 18.55916
KP 4            | 23.82463            | 13.31656
KP 5            | 25.93475            | 10.36481
-----------------------------------------------------------------
OVERALL         | 24.97550
-----------------------------------------------------------------

## 基于OA-CNNS的推理脚本
export PYTHONPATH=.
source .venv/bin/activate
python tools/inference.py \
    --config-file configs/my_dataset/keypoint_oa_cnns.py \
    --weights exp/keypoint_oa_cnns/model/model_best.pth \
    --subset test \
    --idx 10 \
    --visualize \
    --sphere-radius 0.02 \
    --cube-size 0.02


## 基于ptv3_plus的推理脚本
python tools/inference.py \
    --config-file configs/my_dataset/keypoint_ptv3_plus.py \
    --weights exp/keypoint_ptv3_plus/model/model_best.pth \
    --subset all \
    --idx -1 \
    --visualize \
    --sphere-radius 0.02 \
    --cube-size 0.02

====== Batch Inference Statistics ======
Total Samples: 28
-----------------------------------------------------------------
Keypoint ID     | Mean Error           | Std Dev             
-----------------------------------------------------------------
KP 0            | 27.34802            | 21.18443
KP 1            | 24.23499            | 20.26222
KP 2            | 33.68962            | 21.03918
KP 3            | 26.90525            | 18.14697
KP 4            | 29.70839            | 17.92249
KP 5            | 31.32251            | 21.76398
-----------------------------------------------------------------
OVERALL         | 28.86813
-----------------------------------------------------------------
"""
if __name__ == "__main__":
    main()
