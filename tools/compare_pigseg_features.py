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


def compare_normals(stored: np.ndarray, regenerated: np.ndarray) -> Dict[str, float]:
    diff = regenerated - stored
    abs_diff = np.abs(diff)

    stored_norm = np.linalg.norm(stored, axis=1)
    regen_norm = np.linalg.norm(regenerated, axis=1)
    denom = np.clip(stored_norm * regen_norm, a_min=1e-8, a_max=None)
    cosine = np.sum(stored * regenerated, axis=1) / denom
    cosine = np.clip(cosine, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cosine))

    metrics = {
        "normal_abs_diff_mean_x": float(abs_diff[:, 0].mean()),
        "normal_abs_diff_mean_y": float(abs_diff[:, 1].mean()),
        "normal_abs_diff_mean_z": float(abs_diff[:, 2].mean()),
        "normal_abs_diff_max_x": float(abs_diff[:, 0].max()),
        "normal_abs_diff_max_y": float(abs_diff[:, 1].max()),
        "normal_abs_diff_max_z": float(abs_diff[:, 2].max()),
        "normal_cosine_mean": float(cosine.mean()),
        "normal_cosine_min": float(cosine.min()),
        "normal_abs_cosine_mean": float(np.abs(cosine).mean()),
        "normal_angle_deg_mean": float(angle_deg.mean()),
        "normal_angle_deg_p95": float(np.percentile(angle_deg, 95)),
        "normal_angle_deg_max": float(angle_deg.max()),
        "normal_flip_ratio": float(np.mean(cosine < 0.0)),
    }
    return metrics


def compare_curvature(stored: np.ndarray, regenerated: np.ndarray) -> Dict[str, float]:
    diff = regenerated.reshape(-1) - stored.reshape(-1)
    abs_diff = np.abs(diff)
    denom = np.clip(np.abs(stored.reshape(-1)), a_min=1e-8, a_max=None)
    rel_diff = abs_diff / denom
    stored_flat = stored.reshape(-1)
    regenerated_flat = regenerated.reshape(-1)
    if np.std(stored_flat) < 1e-12 or np.std(regenerated_flat) < 1e-12:
        corrcoef = 0.0
    else:
        corrcoef = float(np.corrcoef(stored_flat, regenerated_flat)[0, 1])

    metrics = {
        "curvature_abs_diff_mean": float(abs_diff.mean()),
        "curvature_abs_diff_p95": float(np.percentile(abs_diff, 95)),
        "curvature_abs_diff_max": float(abs_diff.max()),
        "curvature_rel_diff_mean": float(rel_diff.mean()),
        "curvature_rel_diff_p95": float(np.percentile(rel_diff, 95)),
        "curvature_rel_diff_max": float(rel_diff.max()),
        "curvature_corrcoef": corrcoef,
    }
    return metrics


def format_metrics(title: str, metrics: Dict[str, float]) -> str:
    lines = [title]
    for key in sorted(metrics):
        lines.append(f"  {key}: {metrics[key]:.8f}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare stored PigSeg features with regenerated features.")
    parser.add_argument("--npy-file", required=True, help="训练集中的单个 .npy 文件路径")
    parser.add_argument("--knn", type=int, default=30, help="重算法向量和曲率时使用的 KNN")
    parser.add_argument(
        "--save-regenerated-npy",
        default="",
        help="可选，将重生成的 [x,y,z,nx,ny,nz,curvature] 保存到指定 .npy 路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npy_file = os.path.abspath(args.npy_file)
    if not os.path.isfile(npy_file):
        raise FileNotFoundError(f".npy 文件不存在: {npy_file}")

    data = np.load(npy_file, allow_pickle=True)
    if data.ndim != 2 or data.shape[1] < 7:
        raise ValueError("输入 .npy 至少需要 7 列: x, y, z, nx, ny, nz, curvature")

    coord = data[:, 0:3].astype(np.float32)
    stored_normal = data[:, 3:6].astype(np.float32)
    stored_curvature = data[:, 6:7].astype(np.float32)

    valid = ~(
        np.isnan(coord).any(axis=1)
        | np.isnan(stored_normal).any(axis=1)
        | np.isnan(stored_curvature).any(axis=1)
    )
    coord = coord[valid]
    stored_normal = stored_normal[valid]
    stored_curvature = stored_curvature[valid]

    regenerated_normal, regenerated_curvature = estimate_normals_and_curvature(coord, args.knn)

    print(f"npy_file: {npy_file}")
    print(f"valid_points: {coord.shape[0]}")
    print(f"knn: {args.knn}")
    print()

    print(format_metrics("Stored Normal Stats", summarize_array("stored_normal", stored_normal)))
    print()
    print(format_metrics("Regenerated Normal Stats", summarize_array("regenerated_normal", regenerated_normal)))
    print()
    print(format_metrics("Stored Curvature Stats", summarize_array("stored_curvature", stored_curvature)))
    print()
    print(format_metrics("Regenerated Curvature Stats", summarize_array("regenerated_curvature", regenerated_curvature)))
    print()
    print(format_metrics("Normal Comparison", compare_normals(stored_normal, regenerated_normal)))
    print()
    print(format_metrics("Curvature Comparison", compare_curvature(stored_curvature, regenerated_curvature)))

    if args.save_regenerated_npy:
        out_path = os.path.abspath(args.save_regenerated_npy)
        regenerated = np.hstack([coord, regenerated_normal, regenerated_curvature]).astype(np.float32)
        np.save(out_path, regenerated)
        print()
        print(f"saved_regenerated_npy: {out_path}")


if __name__ == "__main__":
    main()
