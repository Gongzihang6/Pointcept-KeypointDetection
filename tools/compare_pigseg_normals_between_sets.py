import argparse
import os
from typing import Dict, Tuple

import numpy as np
import open3d as o3d


def estimate_normals_and_curvature(coord: np.ndarray, knn: int) -> Tuple[np.ndarray, np.ndarray]:
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError("coord 必须是形状为 [N, 3] 的数组。")
    if coord.shape[0] < 3:
        raise ValueError("点数少于 3，无法估计法向量和曲率。")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord.astype(np.float64))

    search_param = o3d.geometry.KDTreeSearchParamKNN(knn=int(knn))
    pcd.estimate_normals(search_param=search_param)
    pcd.orient_normals_towards_camera_location(
        camera_location=np.array([0.0, 0.0, 0.0], dtype=np.float64)
    )

    normals = np.asarray(pcd.normals, dtype=np.float32)

    pcd.estimate_covariances(search_param=search_param)
    covariances = np.asarray(pcd.covariances, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(covariances)
    curvature = (
        eigenvalues[:, 0] / (np.sum(eigenvalues, axis=1) + 1e-8)
    ).astype(np.float32).reshape(-1, 1)

    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
    curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
    return normals, curvature


def summarize_array(name: str, values: np.ndarray) -> Dict[str, float]:
    flat = values.reshape(-1).astype(np.float64)
    return {
        f"{name}_min": float(np.min(flat)),
        f"{name}_p01": float(np.percentile(flat, 1)),
        f"{name}_p50": float(np.percentile(flat, 50)),
        f"{name}_p99": float(np.percentile(flat, 99)),
        f"{name}_max": float(np.max(flat)),
        f"{name}_mean": float(np.mean(flat)),
        f"{name}_std": float(np.std(flat)),
    }


def compare_distribution(source_name: str, target_name: str, source: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    return {
        f"{source_name}_mean": float(np.mean(source)),
        f"{target_name}_mean": float(np.mean(target)),
        "mean_abs_gap": float(abs(np.mean(source) - np.mean(target))),
        f"{source_name}_std": float(np.std(source)),
        f"{target_name}_std": float(np.std(target)),
        "std_abs_gap": float(abs(np.std(source) - np.std(target))),
        f"{source_name}_p01": float(np.percentile(source, 1)),
        f"{target_name}_p01": float(np.percentile(target, 1)),
        "p01_abs_gap": float(abs(np.percentile(source, 1) - np.percentile(target, 1))),
        f"{source_name}_p50": float(np.percentile(source, 50)),
        f"{target_name}_p50": float(np.percentile(target, 50)),
        "p50_abs_gap": float(abs(np.percentile(source, 50) - np.percentile(target, 50))),
        f"{source_name}_p99": float(np.percentile(source, 99)),
        f"{target_name}_p99": float(np.percentile(target, 99)),
        "p99_abs_gap": float(abs(np.percentile(source, 99) - np.percentile(target, 99))),
    }


def nearest_neighbor_metrics(
    source_coord: np.ndarray,
    source_normal: np.ndarray,
    target_coord: np.ndarray,
    target_normal: np.ndarray,
) -> Dict[str, float]:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(target_coord.astype(np.float64))
    tree = o3d.geometry.KDTreeFlann(pcd)

    nn_dist = np.zeros(source_coord.shape[0], dtype=np.float64)
    nn_cos = np.zeros(source_coord.shape[0], dtype=np.float64)

    for index, point in enumerate(source_coord):
        _, nn_idx, nn_dist2 = tree.search_knn_vector_3d(point.astype(np.float64), 1)
        target_index = int(nn_idx[0])
        nn_dist[index] = float(np.sqrt(nn_dist2[0]))

        src_normal = source_normal[index]
        tgt_normal = target_normal[target_index]
        denom = max(float(np.linalg.norm(src_normal) * np.linalg.norm(tgt_normal)), 1e-8)
        cosine = float(np.clip(np.dot(src_normal, tgt_normal) / denom, -1.0, 1.0))
        nn_cos[index] = cosine

    angle_deg = np.degrees(np.arccos(np.clip(nn_cos, -1.0, 1.0)))
    return {
        "nn_dist_mean": float(np.mean(nn_dist)),
        "nn_dist_p50": float(np.percentile(nn_dist, 50)),
        "nn_dist_p95": float(np.percentile(nn_dist, 95)),
        "nn_dist_max": float(np.max(nn_dist)),
        "nn_dist_le_1mm_ratio": float(np.mean(nn_dist <= 1.0)),
        "nn_dist_le_5mm_ratio": float(np.mean(nn_dist <= 5.0)),
        "nn_dist_le_10mm_ratio": float(np.mean(nn_dist <= 10.0)),
        "nn_normal_cosine_mean": float(np.mean(nn_cos)),
        "nn_normal_abs_cosine_mean": float(np.mean(np.abs(nn_cos))),
        "nn_normal_angle_deg_mean": float(np.mean(angle_deg)),
        "nn_normal_angle_deg_p95": float(np.percentile(angle_deg, 95)),
        "nn_normal_flip_ratio": float(np.mean(nn_cos < 0.0)),
    }


def format_metrics(title: str, metrics: Dict[str, float]) -> str:
    lines = [title]
    for key in sorted(metrics):
        lines.append(f"  {key}: {metrics[key]:.8f}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare stored train normals/curvature with regenerated normals/curvature from another xyz point set."
    )
    parser.add_argument("--train-npy", required=True, help="训练集 .npy 文件路径，至少包含 7 列")
    parser.add_argument("--target-xyz", required=True, help="Qt 导出的 xyz .npy 文件路径")
    parser.add_argument("--knn", type=int, default=30, help="重算法向量和曲率时使用的 KNN")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_npy = os.path.abspath(args.train_npy)
    target_xyz_file = os.path.abspath(args.target_xyz)
    if not os.path.isfile(train_npy):
        raise FileNotFoundError(f"训练样本不存在: {train_npy}")
    if not os.path.isfile(target_xyz_file):
        raise FileNotFoundError(f"目标 xyz 文件不存在: {target_xyz_file}")

    train_data = np.load(train_npy, allow_pickle=True)
    if train_data.ndim != 2 or train_data.shape[1] < 7:
        raise ValueError("训练 .npy 至少需要 7 列: x, y, z, nx, ny, nz, curvature")

    train_coord = train_data[:, 0:3].astype(np.float32)
    train_normal = train_data[:, 3:6].astype(np.float32)
    train_curvature = train_data[:, 6:7].astype(np.float32)

    target_xyz = np.load(target_xyz_file, allow_pickle=True)
    if target_xyz.ndim != 2 or target_xyz.shape[1] < 3:
        raise ValueError("目标 xyz 文件必须是至少 3 列的二维数组。")
    target_coord = target_xyz[:, 0:3].astype(np.float32)
    target_normal, target_curvature = estimate_normals_and_curvature(target_coord, args.knn)

    train_valid = ~(
        np.isnan(train_coord).any(axis=1)
        | np.isnan(train_normal).any(axis=1)
        | np.isnan(train_curvature).any(axis=1)
    )
    train_coord = train_coord[train_valid]
    train_normal = train_normal[train_valid]
    train_curvature = train_curvature[train_valid]

    target_valid = ~(
        np.isnan(target_coord).any(axis=1)
        | np.isnan(target_normal).any(axis=1)
        | np.isnan(target_curvature).any(axis=1)
    )
    target_coord = target_coord[target_valid]
    target_normal = target_normal[target_valid]
    target_curvature = target_curvature[target_valid]

    print(f"train_npy: {train_npy}")
    print(f"target_xyz: {target_xyz_file}")
    print(f"train_points: {train_coord.shape[0]}")
    print(f"target_points: {target_coord.shape[0]}")
    print(f"knn: {args.knn}")
    print()

    print(format_metrics("Train Normal Stats", summarize_array("train_normal", train_normal)))
    print()
    print(format_metrics("Target Regenerated Normal Stats", summarize_array("target_normal", target_normal)))
    print()
    print(format_metrics("Train Curvature Stats", summarize_array("train_curvature", train_curvature)))
    print()
    print(format_metrics("Target Regenerated Curvature Stats", summarize_array("target_curvature", target_curvature)))
    print()
    print(
        format_metrics(
            "Normal Distribution Gap",
            compare_distribution("train_normal", "target_normal", train_normal, target_normal),
        )
    )
    print()
    print(
        format_metrics(
            "Curvature Distribution Gap",
            compare_distribution(
                "train_curvature",
                "target_curvature",
                train_curvature.reshape(-1),
                target_curvature.reshape(-1),
            ),
        )
    )
    print()
    print(
        format_metrics(
            "Nearest Neighbor Alignment",
            nearest_neighbor_metrics(train_coord, train_normal, target_coord, target_normal),
        )
    )


if __name__ == "__main__":
    main()
