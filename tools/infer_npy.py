"""
===============================================================================
代码作用：
单文件点云语义分割推理脚本，包含完整预处理、模型推理，并将预测结果保存为 .ply 格式文件，
同时新增了单独提取并保存目标主体（猪）点云的功能。

功能/实现了什么：
1. 读取指定的 .npy 点云文件。
2. 执行与训练环境完全一致的预处理（NaN清理、离群点过滤、体素下采样、中心平移）。
3. 加载训练好的模型权重（如 Swin3D）进行前向推理，获取每个点的类别预测。
4. 导出两个可视化文件：
   - 完整的彩色场景点云（背景红色，猪蓝色）。
   - 仅包含预测标签为 1（猪主体）的点云文件，剔除了所有背景杂点。

怎么实现的：
- 预处理：利用 numpy 进行距离计算、布尔掩码过滤和网格离散化取唯一值。
- 推理：将处理好的数据构造成 Pointcept 所需的 batch=1 的字典格式送入 GPU 运算。
- 结果保存：利用 numpy 的布尔索引 `pig_coords = original_coord[preds == 1]` 提取出目标点，
  然后利用 `open3d.io.write_point_cloud` 将完整点云和过滤后的纯目标点云分别写入磁盘。
===============================================================================
"""

import argparse
import os
import sys
import numpy as np
import torch
import open3d as o3d

# 确保能正确导入 Pointcept 的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from pointcept.utils.config import Config
from pointcept.models import build_model

def get_args():
    parser = argparse.ArgumentParser(description="Pig Semantic Segmentation Inference")
    parser.add_argument("--config-file", default="configs/pigseg/semseg-swin3d-v1m1-0-base.py", help="训练使用的配置文件")
    parser.add_argument("--weights", default="exp/autodl_weights/swin3d/model_best.pth", help="最佳权重文件路径")
    parser.add_argument("--npy-file", required=True, help="需要推理的 .npy 文件绝对或相对路径")
    parser.add_argument("--voxel-size", type=float, default=20.0, help="下采样网格尺寸(mm)")
    return parser.parse_args()

def load_and_preprocess_data(npy_path, voxel_size=20.0):
    """手动模拟 PigDataset 和 Pipeline 的预处理过程，确保输入特征与训练时一模一样"""
    print(f"=> Loading data from: {npy_path}")
    data = np.load(npy_path)
    
    coord = data[:, 0:3].astype(np.float32)
    normal = data[:, 3:6].astype(np.float32)
    curvature = data[:, 6:7].astype(np.float32)

    # 1. 第 0 道防线：致命 NaN 清洗
    valid_nan = ~(np.isnan(normal).any(axis=1) | np.isnan(curvature).any(axis=1) | np.isnan(coord).any(axis=1))
    coord, normal, curvature = coord[valid_nan], normal[valid_nan], curvature[valid_nan]

    if coord.shape[0] == 0:
        raise ValueError("Data is empty after NaN filtering!")

    # 2. 第 1 道防线：过滤极远飞点 (截断 5000mm)
    median_coord = np.median(coord, axis=0)
    coord = coord - median_coord
    dist = np.linalg.norm(coord, axis=1)
    valid_mask = dist < 5000.0
    coord, normal, curvature = coord[valid_mask], normal[valid_mask], curvature[valid_mask]

    # 3. 第 2 道防线：体素预下采样 (20mm)
    discrete_coords = np.floor(coord / voxel_size).astype(np.int32)
    _, unique_indices = np.unique(discrete_coords, axis=0, return_index=True)
    coord = coord[unique_indices]
    normal = normal[unique_indices]
    curvature = curvature[unique_indices]

    # 4. CenterShift (模拟 train_pipeline 中的中心平移)
    coord[:, 0] -= coord[:, 0].mean()
    coord[:, 1] -= coord[:, 1].mean()
    coord[:, 2] -= coord[:, 2].min()  # Z轴通常平移至最小值

    # 重新计算平移后的 grid_coord
    grid_coord = np.floor(coord / voxel_size).astype(np.int32)

    # 5. 特征拼接 (模拟 Collect: feat_keys=("normal", "curvature"))
    feat = np.concatenate([normal, curvature], axis=1).astype(np.float32)

    # 6. 构造模型所需的输入字典
    num_points = coord.shape[0]
    data_dict = {
        "coord": torch.tensor(coord).cuda(),
        "grid_coord": torch.tensor(grid_coord).cuda(),
        "feat": torch.tensor(feat).cuda(),
        "coord_feat": torch.tensor(feat).cuda(),
        "offset": torch.tensor([num_points], dtype=torch.int32).cuda(),
        "batch": torch.zeros(num_points, dtype=torch.long).cuda()
    }
    
    print(f"=> Preprocessing finished. Points after voxelization: {num_points}")
    return data_dict, coord

def main():
    args = get_args()
    
    # 1. 初始化模型
    print(f"=> Loading config: {args.config_file}")
    cfg = Config.fromfile(args.config_file)
    model = build_model(cfg.model)
    
    print(f"=> Loading weights: {args.weights}")
    # 添加 weights_only=True 消除 PyTorch 安全警告
    checkpoint = torch.load(args.weights, map_location="cuda", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.cuda()
    model.eval()

    # 2. 数据读取与处理
    data_dict, original_coord = load_and_preprocess_data(args.npy_file, args.voxel_size)

    # 3. 执行推理
    print("=> Running inference...")
    with torch.no_grad():
        output = model(data_dict)
        if isinstance(output, dict):
            logits = output.get("seg_logits", output.get("pred"))
        else:
            logits = output
            
        # 将 logits 转换为 0~1 的概率分布
        probs = torch.softmax(logits, dim=1) 
        
        # 提取模型认为是猪(类别 1)的概率
        pig_probs = probs[:, 1]
        
        # 【关键】自定义阈值！原本是 0.5，现在降低到 0.2。
        # 意思是：只要模型认为有 20% 的可能是一头猪，我们就宁杀错不放过，把它判为猪！
        threshold = 0.5
        preds = (pig_probs > threshold).int().cpu().numpy()

    # =========================================================================
    # 4. 保存文件阶段 (适配无头服务器，不直接弹窗)
    # =========================================================================
    
    # --- 任务 A: 保存完整的红蓝上色点云 ---
    print("=> Saving full colored point cloud...")
    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(original_coord)

    colors = np.zeros_like(original_coord)
    colors[preds == 0] = [1.0, 0.0, 0.0] # 背景红
    colors[preds == 1] = [0.0, 0.0, 1.0] # 猪主体蓝
    pcd_full.colors = o3d.utility.Vector3dVector(colors)

    output_full_path = args.npy_file.replace('.npy', '_pred_full.ply')
    o3d.io.write_point_cloud(output_full_path, pcd_full)
    print(f"  -> 完整场景已保存: {output_full_path}")

    # --- 任务 B: 仅保留并保存猪主体(标签为1)的点云 ---
    print("=> Extracting and saving Pig-only point cloud...")
    pig_mask = (preds == 1)
    pig_coords = original_coord[pig_mask]

    if len(pig_coords) > 0:
        pcd_pig = o3d.geometry.PointCloud()
        pcd_pig.points = o3d.utility.Vector3dVector(pig_coords)
        # 将提取出的猪统一涂成蓝色，或者你也可以不涂色保留原始坐标
        pcd_pig.paint_uniform_color([0.0, 0.0, 1.0])
        
        output_pig_path = args.npy_file.replace('.npy', '_pig_only.ply')
        o3d.io.write_point_cloud(output_pig_path, pcd_pig)
        print(f"  -> 猪主体点云已提取并保存: {output_pig_path} (共 {len(pig_coords)} 个点)")
    else:
        print("  -> [警告] 模型没有在当前场景中预测出任何猪主体(Label=1)的点！")

    print("\n=====================================================")
    print(" 推理与提取完成！请将 .ply 文件下载到本地查看。")
    print("=====================================================")

"""
python tools/infer_npy.py --npy-file body_npy_output/train/20260329_105410_942.npy
"""
if __name__ == "__main__":
    main()