"""
Keypoint Detection Inference & Visualization Script
功能：
1. 支持单样本推理：计算误差 + Open3D 可视化
2. 支持批量推理：计算整个数据集的平均误差 (Mean) 和标准差 (Std)
3. 架构通用：通过 config 文件自动加载对应的模型架构
4. 支持 subset='all'，一次性推理 train+val+test 所有数据
5. [修复] 智能提取样本文件名（支持任意字段名的文件路径），解决 SampleID 为 sample_0 的问题

日期：2025-12-09 (Fixed Name Extraction V2)
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import sys
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime

# 添加项目根目录到 python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from pointcept.utils.config import Config, DictAction
from pointcept.models import build_model
from pointcept.datasets import build_dataset, point_collate_fn

def get_args():
    parser = argparse.ArgumentParser(description="Pointcept Keypoint Inference")
    parser.add_argument("--config-file", default="configs/my_dataset/keypoint_ptv3.py", help="配置文件路径")
    parser.add_argument("--options", nargs="+", action=DictAction, help="覆盖配置文件的参数")
    parser.add_argument("--weights", default=None, required=True, help="模型权重文件路径 (.pth)")
    parser.add_argument("--subset", default="val", choices=["train", "val", "test", "all"], help="数据集划分")
    parser.add_argument("--idx", type=int, default=-1, help="单样本索引。如果为 -1，则进行批量推理")
    
    # 可视化参数
    parser.add_argument("--visualize", action="store_true", help="是否开启 Open3D 可视化")
    parser.add_argument("--sphere-radius", type=float, default=0.05, help="真实关键点(球)的半径")
    parser.add_argument("--cube-size", type=float, default=0.08, help="预测关键点(正方体)的边长")
    parser.add_argument("--point-size", type=float, default=2.0, help="Open3D 点大小")
    parser.add_argument("--save-dir", default=None, help="结果保存路径")
    
    args = parser.parse_args()
    return args

def setup_model(cfg, weights_path):
    print(f"=> Building model from config: {cfg.model.type}")
    model = build_model(cfg.model)
    
    if os.path.isfile(weights_path):
        print(f"=> Loading weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cuda")
        state_dict = checkpoint.get("state_dict", checkpoint)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=True)
    else:
        raise FileNotFoundError(f"No weights found at {weights_path}")
    
    model.cuda()
    model.eval()
    return model

def create_colored_mesh(geometry_type, center, color, size):
    if geometry_type == 'sphere':
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=size)
    elif geometry_type == 'box':
        mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
        mesh.translate(-np.array([size/2, size/2, size/2]))
    
    mesh.translate(center)
    mesh.paint_uniform_color(color)
    return mesh

def visualize_single(coord, pred_kps, target_kps, args, num_kps):
    print(f"=> Visualizing... (Point Size: {args.point_size})")
    geometries = []
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    geometries.append(pcd)

    cmap = plt.get_cmap("jet")
    colors = [cmap(i / (num_kps - 1 if num_kps > 1 else 1))[:3] for i in range(num_kps)]

    for i in range(num_kps):
        if target_kps is not None:
            sphere = create_colored_mesh('sphere', target_kps[i], colors[i], args.sphere_radius)
            geometries.append(sphere)
        cube = create_colored_mesh('box', pred_kps[i], colors[i], args.cube_size)
        geometries.append(cube)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Sample {args.idx}", width=1024, height=768)
    for geom in geometries:
        vis.add_geometry(geom)
    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.asarray([1, 1, 1])
    vis.run()
    vis.destroy_window()

def debug_dataset_structure(dataset):
    """
    [新增] 打印数据集的第一条数据结构，帮助定位文件名在哪里
    """
    print("\n" + "="*20 + " DEBUG DATASET INFO " + "="*20)
    try:
        # 如果是 ConcatDataset，取第一个子集
        if isinstance(dataset, torch.utils.data.ConcatDataset):
            ds = dataset.datasets[0]
        else:
            ds = dataset

        if hasattr(ds, 'data_list'):
            sample_info = ds.data_list[0]
            print(f"Dataset has 'data_list'. Type of item: {type(sample_info)}")
            print(f"Content of first item in data_list: {sample_info}")
        elif hasattr(ds, 'files'):
             print(f"Dataset has 'files'. First item: {ds.files[0]}")
        elif hasattr(ds, 'filenames'):
             print(f"Dataset has 'filenames'. First item: {ds.filenames[0]}")
        else:
            print("Dataset has NO known list attribute (data_list/files/filenames).")
            print(f"Dataset dir(): {dir(ds)}")
    except Exception as e:
        print(f"Error inspecting dataset: {e}")
    print("="*60 + "\n")

def get_sample_name(dataset, idx):
    """
    [增强版] 递归获取样本名称，支持智能搜索文件名
    """
    # 1. 处理 ConcatDataset (subset='all')
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        for i, size in enumerate(dataset.cumulative_sizes):
            if idx < size:
                prev_size = dataset.cumulative_sizes[i-1] if i > 0 else 0
                return get_sample_name(dataset.datasets[i], idx - prev_size)
    
    # 2. 获取数据项信息
    info = None
    if hasattr(dataset, 'data_list'):
        info = dataset.data_list[idx]
    elif hasattr(dataset, 'files'):
        info = dataset.files[idx]
    elif hasattr(dataset, 'filenames'):
        info = dataset.filenames[idx]
    
    # 3. 解析文件名
    if info is not None:
        # 情况 A: 直接是字符串路径
        if isinstance(info, str):
            filename = os.path.basename(info)
            return os.path.splitext(filename)[0]
        
        # 情况 B: 是字典，搜索可能是文件名的字段
        if isinstance(info, dict):
            # 优先查找显式的 name 字段
            if 'name' in info: return str(info['name'])
            if 'scene_id' in info: return str(info['scene_id'])
            
            # 智能搜索：查找值是字符串且包含常见后缀的字段
            valid_extensions = ('.ply', '.bin', '.pth', '.txt', '.xyz', '.las')
            for k, v in info.items():
                if isinstance(v, str) and v.endswith(valid_extensions):
                    filename = os.path.basename(v)
                    return os.path.splitext(filename)[0]
            
            # 如果没找到后缀，尝试找 'coord' 字段对应的原本路径（有的数据集 coord 存的是路径）
            if 'coord' in info and isinstance(info['coord'], str):
                return os.path.splitext(os.path.basename(info['coord']))[0]

        # 情况 C: 元组
        if isinstance(info, (list, tuple)) and len(info) > 0 and isinstance(info[0], str):
            return os.path.splitext(os.path.basename(info[0]))[0]

    # 4. Fallback
    return f"sample_{idx}"

def inference_single_sample(cfg, model, dataset, args):
    """单样本推理"""
    idx = args.idx
    if idx >= len(dataset):
        print(f"Error: Index {idx} out of bounds")
        return

    data_dict = dataset[idx]
    data_dict = point_collate_fn([data_dict])
    
    for key in data_dict:
        if isinstance(data_dict[key], torch.Tensor):
            data_dict[key] = data_dict[key].cuda(non_blocking=True)

    with torch.no_grad():
        result = model(data_dict)
        pred = result["pred"] if isinstance(result, dict) else result

    num_kps = cfg.model.num_keypoints
    pred = pred.view(-1, num_kps, 3).cpu().numpy()[0]
    
    target = None
    if "target" in data_dict:
        target = data_dict["target"].view(-1, num_kps, 3).cpu().numpy()[0]

    coord = data_dict["coord"].cpu().numpy()
    
    scale = 1.0
    if "scale" in data_dict:
        scale = data_dict["scale"].cpu().numpy()[0]
    elif "grid_size" in data_dict:
        scale = data_dict["grid_size"]
        if isinstance(scale, torch.Tensor): scale = scale.item()

    sample_name = get_sample_name(dataset, idx)

    print(f"\n====== Inference Result [Sample: {sample_name}] ======")
    print(f"Scale Factor: {scale}")

    if target is not None:
        diff = np.linalg.norm(pred - target, axis=-1)
        real_diff = diff * scale 

        print("-" * 40)
        print(f"{'Keypoint ID':<15} | {'Error (Original Scale)':<25}")
        print("-" * 40)
        for i in range(num_kps):
            print(f"KP {i:<12} | {real_diff[i]:.4f}")
        print("-" * 40)
        print(f"Mean Error      | {np.mean(real_diff):.4f}")
        print("-" * 40)
    
    if args.visualize:
        visualize_single(coord, pred, target, args, num_kps)

def plot_batch_errors(all_errors, num_kps):
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    
    cols = 3
    rows = (num_kps + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    fig.suptitle('Keypoint Prediction Errors (Batch Inference)', fontsize=16)
    
    axes = axes.flatten() if num_kps > 1 else [axes]
    x = np.arange(all_errors.shape[0])
    
    for i in range(num_kps):
        if i >= len(axes): break
        ax = axes[i]
        y = all_errors[:, i]
        
        # 1. 计算统计量
        mean_val = np.mean(y)
        std_val = np.std(y)
        
        # 2. 定义可视化上限 (均值 + 3倍标准差)
        # 加上 max 是为了防止数据极其集中时(std~0)显示范围太小，至少保留一点高度
        vis_upper_limit = mean_val + 2.5 * std_val
        
        # 如果数据中存在极端离群值，std 也会变大，所以可以额外加一个保护逻辑（可选）：
        # 例如：不让上限超过最大值的 95 分位数的 1.5 倍，或者直接使用你的 3sigma 逻辑。
        # 这里严格按照你的要求：Mean + 3 * Std
        
        # 绘制散点
        ax.scatter(x, y, alpha=0.6, s=10, c='blue', label='Sample Error')
        
        # 绘制辅助线
        ax.axhline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax.axhline(mean_val + 2 * std_val, color='green', linestyle='--', label=f'Mean+2Std')
        
        # 3. [核心修改] 强制设置 Y 轴范围
        # 底部固定为 0 (因为误差是非负的)
        # 顶部设置为 3倍标准差 + 10% 的余量，让图好看一点
        current_ymax = vis_upper_limit * 1.1
        ax.set_ylim(bottom=-0.02 * current_ymax, top=current_ymax)
        
        ax.set_title(f'Keypoint {i}')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Error (m)')
        ax.legend(loc='upper right', fontsize='small') # 图例稍微改小一点防止遮挡
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def inference_batch(cfg, model, dataset, args):
    """批量推理逻辑"""
    print(f"=> Start Batch Inference on [{args.subset}] set...")
    
    # [新增] 打印 Debug 信息
    debug_dataset_structure(dataset)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=cfg.data.batch_size if hasattr(cfg.data, "batch_size") else 1,
        shuffle=False, 
        num_workers=cfg.num_worker, 
        collate_fn=point_collate_fn,
        pin_memory=True
    )

    num_kps = cfg.model.num_keypoints
    all_errors = []     
    all_pred_coords = [] 
    
    model.eval()
    with torch.no_grad():
        for i, data_dict in enumerate(tqdm(dataloader)):
            for key in data_dict:
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda(non_blocking=True)
            
            result = model(data_dict)
            pred = result["pred"] if isinstance(result, dict) else result
            
            target = data_dict["target"]
            pred = pred.view(-1, num_kps, 3)
            target = target.view(-1, num_kps, 3)
            
            scale_factor = 1.0
            if "scale" in data_dict:
                scale = data_dict["scale"]
                if scale.ndim == 1: 
                    scale_factor = scale.view(-1, 1)
                    scale_tensor = scale.view(-1, 1, 1)
                else:
                    scale_factor = scale
                    scale_tensor = scale.unsqueeze(-1)
            elif "grid_size" in data_dict:
                g = data_dict["grid_size"]
                scale_factor = g
                scale_tensor = g
            
            dist = torch.norm(pred - target, p=2, dim=-1)
            dist = dist * scale_factor
            all_errors.append(dist.cpu().numpy())

            pred_physical = pred * scale_tensor
            all_pred_coords.append(pred_physical.cpu().numpy())

    all_errors = np.concatenate(all_errors, axis=0)
    all_pred_coords = np.concatenate(all_pred_coords, axis=0)
    
    print("=> Extracting sample names from dataset...")
    all_sample_names = []
    for idx in range(len(dataset)):
        name = get_sample_name(dataset, idx)
        all_sample_names.append(name)

    assert len(all_sample_names) == all_pred_coords.shape[0], \
        f"Mismatch: {len(all_sample_names)} names vs {all_pred_coords.shape[0]} predictions"

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

    if args.save_dir is not None:
        save_dir = args.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = cfg.model.type
        txt_filename = f"{model_name}_{timestamp}_预测关键点坐标.txt"
        txt_path = os.path.join(save_dir, txt_filename)

        print(f"=> Exporting predicted coordinates to: {txt_path}")
        
        with open(txt_path, "w") as f:
            header = "SampleID"
            for k in range(num_kps):
                header += f" KP{k}_x KP{k}_y KP{k}_z"
            f.write(header + "\n")
            
            for i in range(all_pred_coords.shape[0]):
                line = f"{all_sample_names[i]}" 
                for kp_idx in range(num_kps):
                    coords = all_pred_coords[i, kp_idx]
                    line += f" {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}"
                f.write(line + "\n")
                
        print("=> Export done.")
    else:
        print("=> --save-dir not specified, skipping txt export.")
    print("=> Plotting error distribution...")
    plot_batch_errors(all_errors, num_kps)

def main():
    args = get_args()
    cfg = Config.fromfile(args.config_file)
    if args.options:
        cfg.merge_from_dict(args.options)
    
    model = setup_model(cfg, args.weights)
    
    if args.subset == "all":
        assert args.idx == -1, "Error: subset='all' only supports batch inference."
        datasets = []
        splits = ["train", "val", "test"]
        print(f"=> Merging datasets: {splits}...")
        for split in splits:
            if split in cfg.data:
                dataset_cfg = cfg.data[split]
                dataset_cfg.data_root = cfg.data_root
                ds = build_dataset(dataset_cfg)
                datasets.append(ds)
                print(f"   -> Loaded {split}: {len(ds)} samples")
        
        if len(datasets) == 0:
            raise ValueError("No datasets found.")
        dataset = torch.utils.data.ConcatDataset(datasets)
    else:
        if args.subset not in cfg.data:
            raise ValueError(f"Subset {args.subset} not found in config.data")
        dataset_cfg = cfg.data[args.subset]
        dataset_cfg.data_root = cfg.data_root
        dataset = build_dataset(dataset_cfg)
        print(f"=> Loaded {len(dataset)} samples from {args.subset} set.")

    if args.idx != -1:
        inference_single_sample(cfg, model, dataset, args)
    else:
        inference_batch(cfg, model, dataset, args)
"""
## 基于Swin3D模型的批量推理脚本
export PYTHONPATH=.
python tools/batch_infer_export_txt.py \
    --config-file configs/my_dataset/keypoint_swin3d.py \
    --weights exp/keypoint_swin3d/model/model_best.pth \
    --subset all \
    --idx -1 \
    --visualize \
    --sphere-radius 0.02 \
    --cube-size 0.02 \
    --save-dir 预测关键点坐标/Swin3D

## 基于StratifiedTransformer模型的批量推理脚本
export PYTHONPATH=.
python tools/batch_infer_export_txt.py \
    --config-file configs/my_dataset/keypoint_stratified_transformer.py \
    --weights exp/keypoint_stratified_transformer/model/model_best.pth \
    --subset all \
    --idx -1 \
    --visualize \
    --sphere-radius 0.02 \
    --cube-size 0.02 \
    --save-dir 预测关键点坐标/StratifiedTransformer
"""
if __name__ == "__main__":
    main()