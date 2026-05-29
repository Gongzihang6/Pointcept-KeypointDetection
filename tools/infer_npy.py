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
from tools.pigseg_shared import build_input_dict, parse_npy_file, preprocess_like_infer_npy

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected, use true/false.")

def get_args():
    parser = argparse.ArgumentParser(description="Pig Semantic Segmentation Inference")
    parser.add_argument("--config-file", default="configs/pigseg/semseg-ptv3-v1m1-0-base.py", help="训练使用的配置文件")
    parser.add_argument("--weights", default="exp/PTV3_PigSeg_0511/model/model_best.pth", help="最佳权重文件路径")
    parser.add_argument("--npy-file", required=True, help="需要推理的 .npy 文件绝对或相对路径")
    parser.add_argument("--voxel-size", type=float, default=None, help="体素尺寸；默认跟随配置文件中的 voxel_size")
    parser.add_argument("--max-nn", type=int, default=30, help="当输入只有 xyz 时，估计法向量/曲率使用的 KNN")
    parser.add_argument("--outlier-distance", type=float, default=5000.0, help="飞点过滤阈值，单位与输入坐标一致")
    parser.add_argument("--pig-label", type=int, default=1, help="猪类别标签编号")
    parser.add_argument(
        "--pig-prob-thresh",
        type=float,
        default=-1.0,
        help="默认使用 argmax；若设为 >=0，则按猪类别概率阈值二值化",
    )
    parser.add_argument(
        "--save-point-clouds",
        type=str2bool,
        default=True,
        help="是否将完整点云和猪主体点云保存到本地(true/false)",
    )
    parser.add_argument(
        "--visualize",
        type=str2bool,
        default=False,
        help="是否弹出Open3D窗口可视化分割结果(true/false)",
    )
    return parser.parse_args()

def resolve_voxel_size(cfg, cli_value):
    if cli_value is not None:
        return float(cli_value)
    if hasattr(cfg, "voxel_size"):
        return float(cfg.voxel_size)
    if hasattr(cfg, "data") and "test" in cfg.data and "test_cfg" in cfg.data.test:
        return float(cfg.data.test.test_cfg.voxelize.grid_size)
    raise ValueError("无法从配置文件解析 voxel_size，请显式传入 --voxel-size")

def build_colored_point_cloud(coord, preds):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)

    colors = np.zeros_like(coord, dtype=np.float64)
    colors[preds == 0] = [1.0, 0.0, 0.0]  # 背景红
    colors[preds == 1] = [0.0, 0.0, 1.0]  # 猪主体蓝
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def save_prediction_outputs(npy_file, original_coord, preds):
    print("=> Saving predicted point clouds...")

    pcd_full = build_colored_point_cloud(original_coord, preds)
    output_full_path = npy_file.replace(".npy", "_pred_full.ply")
    o3d.io.write_point_cloud(output_full_path, pcd_full)
    print(f"  -> 完整场景已保存: {output_full_path}")

    pig_mask = preds == 1
    pig_coords = original_coord[pig_mask]
    if len(pig_coords) > 0:
        pcd_pig = o3d.geometry.PointCloud()
        pcd_pig.points = o3d.utility.Vector3dVector(pig_coords)
        pcd_pig.paint_uniform_color([0.0, 0.0, 1.0])

        output_pig_path = npy_file.replace(".npy", "_pig_only.ply")
        o3d.io.write_point_cloud(output_pig_path, pcd_pig)
        print(f"  -> 猪主体点云已提取并保存: {output_pig_path} (共 {len(pig_coords)} 个点)")
    else:
        print("  -> [警告] 模型没有在当前场景中预测出任何猪主体(Label=1)的点，跳过保存猪主体点云。")

def visualize_prediction(original_coord, preds):
    print("=> Visualizing segmentation result...")
    pcd_vis = build_colored_point_cloud(original_coord, preds)
    try:
        o3d.visualization.draw_geometries(
            [pcd_vis],
            window_name="Pig Segmentation Result",
            width=1280,
            height=720,
        )
    except Exception as exc:
        print(f"  -> [警告] 可视化失败，当前环境可能不支持图形界面: {exc}")

def main():
    args = get_args()
    
    # 1. 初始化模型
    print(f"=> Loading config: {args.config_file}")
    cfg = Config.fromfile(args.config_file)
    voxel_size = resolve_voxel_size(cfg, args.voxel_size)
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
    print(f"=> Loading data from: {args.npy_file}")
    point_cloud_data = parse_npy_file(args.npy_file)
    processed, preprocess_meta = preprocess_like_infer_npy(
        point_cloud_data=point_cloud_data,
        max_nn=args.max_nn,
        outlier_distance=args.outlier_distance,
        voxel_size=voxel_size,
    )
    data_dict = build_input_dict(
        coord=processed["coord"],
        feat=processed["feat"],
        grid_coord=processed["grid_coord"],
        device=torch.device("cuda"),
    )
    original_coord = processed["return_coord"]
    print(
        "=> Preprocessing finished. "
        f"source_points={preprocess_meta['source_points']} "
        f"valid_input_points={preprocess_meta['valid_input_points']} "
        f"feature_valid_points={preprocess_meta['feature_valid_points']} "
        f"valid_points={preprocess_meta['valid_points']} "
        f"voxel_points={preprocess_meta['voxel_points']} "
        f"feature_source={preprocess_meta['feature_source']} "
        f"voxel_size={voxel_size}"
    )

    # 3. 执行推理
    print("=> Running inference...")
    with torch.no_grad():
        output = model(data_dict)
        if isinstance(output, dict):
            logits = output.get("seg_logits", output.get("pred"))
        else:
            logits = output
            
        probs = torch.softmax(logits, dim=1) 
        pig_probs = probs[:, args.pig_label]

        if args.pig_prob_thresh >= 0.0:
            preds = (pig_probs > args.pig_prob_thresh).int().cpu().numpy()
            print(f"=> Prediction mode: threshold, pig_prob_thresh={args.pig_prob_thresh}")
        else:
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            print("=> Prediction mode: argmax")
        pig_positive = int((preds == args.pig_label).sum())
        print(
            "=> Prediction stats: "
            f"pig_prob_min={float(pig_probs.min().item()):.6f} "
            f"pig_prob_mean={float(pig_probs.mean().item()):.6f} "
            f"pig_prob_max={float(pig_probs.max().item()):.6f} "
            f"pig_positive={pig_positive} "
            f"total_points={int(preds.shape[0])}"
        )

    if args.save_point_clouds:
        save_prediction_outputs(args.npy_file, original_coord, preds)
        print("\n=====================================================")
        print(" 推理与提取完成！请将 .ply 文件下载到本地查看。")
        print("=====================================================")
    else:
        print("=> Skip saving point clouds because --save-point-clouds=false")

    if args.visualize:
        visualize_prediction(original_coord, preds)
    else:
        print("=> Skip visualization because --visualize=false")



"""
## 基于Swin3D的语义分割推理
python tools/infer_npy.py \
    --weights exp/Swin3D_PigSeg_0512/model/model_best.pth \
    --config-file configs/pigseg/semseg-swin3d-v1m1-0-base.py \
    --npy-file body_npy_output/train/20260329_105410_942.npy \
    --save-point-clouds true \
    --visualize false

"""
if __name__ == "__main__":
    main()
