from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import open3d as o3d
import torch


@dataclass
class ParsedPointCloudData:
    coord: np.ndarray
    normal: Optional[np.ndarray]
    curvature: Optional[np.ndarray]
    source_points: int
    valid_input_points: int
    feature_source: str


def parse_point_cloud_buffer(content: bytes, num_channels: int) -> ParsedPointCloudData:
    if num_channels < 3:
        raise ValueError("num_channels 不能小于 3，至少需要 xyz 三列坐标。")
    if num_channels != 3 and num_channels < 7:
        raise ValueError("num_channels 只能为 3 或 >= 7；若提供特征，请至少包含 xyz + normal + curvature。")

    raw = np.frombuffer(content, dtype=np.float32)
    if raw.size == 0:
        raise ValueError("收到的点云二进制为空。")
    if raw.size % num_channels != 0:
        raise ValueError(f"二进制长度无法按 num_channels={num_channels} 整除，请检查数据格式。")

    points = raw.reshape(-1, num_channels)
    coord = points[:, :3].astype(np.float32)
    valid = np.isfinite(coord).all(axis=1)

    normal = None
    curvature = None
    feature_source = "estimated"
    if num_channels >= 7:
        normal = points[:, 3:6].astype(np.float32)
        curvature = points[:, 6:7].astype(np.float32)
        valid &= np.isfinite(normal).all(axis=1)
        valid &= np.isfinite(curvature).all(axis=1)
        feature_source = "provided"

    coord = coord[valid]
    if normal is not None:
        normal = normal[valid]
    if curvature is not None:
        curvature = curvature[valid]
    if coord.shape[0] == 0:
        raise ValueError("点云在清理 NaN / Inf 后为空。")

    return ParsedPointCloudData(
        coord=coord,
        normal=normal,
        curvature=curvature,
        source_points=int(points.shape[0]),
        valid_input_points=int(coord.shape[0]),
        feature_source=feature_source,
    )


def estimate_normals_and_curvature(coord: np.ndarray, max_nn: int) -> Tuple[np.ndarray, np.ndarray]:
    if coord.shape[0] < 3:
        raise ValueError("有效点数小于 3，无法估计法向量和曲率。")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord.astype(np.float64))
    search_param = o3d.geometry.KDTreeSearchParamKNN(knn=int(max_nn))
    pcd.estimate_normals(search_param=search_param)
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0], dtype=np.float64))

    normals = np.asarray(pcd.normals, dtype=np.float32)
    pcd.estimate_covariances(search_param=search_param)
    covariances = np.asarray(pcd.covariances, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(covariances)
    curvature = (eigenvalues[:, 0] / (np.sum(eigenvalues, axis=1) + 1e-8)).astype(np.float32).reshape(-1, 1)

    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
    curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
    return normals, curvature


def center_shift_apply_z(coord: np.ndarray) -> np.ndarray:
    x_min, y_min, z_min = coord.min(axis=0)
    x_max, y_max, _ = coord.max(axis=0)
    shift = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, z_min], dtype=np.float32)
    return coord - shift


def preprocess_like_infer_npy(
    point_cloud_data: ParsedPointCloudData,
    max_nn: int,
    outlier_distance: float,
    voxel_size: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    coord_raw = point_cloud_data.coord.astype(np.float32)
    if point_cloud_data.normal is not None and point_cloud_data.curvature is not None:
        normals_raw = point_cloud_data.normal.astype(np.float32)
        curvature_raw = point_cloud_data.curvature.astype(np.float32)
        feature_source = "provided"
    else:
        normals_raw, curvature_raw = estimate_normals_and_curvature(coord=coord_raw, max_nn=max_nn)
        feature_source = "estimated"

    valid_nan = ~(
        np.isnan(normals_raw).any(axis=1)
        | np.isnan(curvature_raw).any(axis=1)
        | np.isnan(coord_raw).any(axis=1)
    )
    coord = coord_raw[valid_nan].astype(np.float32)
    original_coord = coord.copy()
    normal = normals_raw[valid_nan].astype(np.float32)
    curvature = curvature_raw[valid_nan].astype(np.float32)
    if coord.shape[0] == 0:
        raise ValueError("Data is empty after NaN filtering!")

    median_coord = np.median(coord, axis=0)
    coord = coord - median_coord
    dist = np.linalg.norm(coord, axis=1)
    valid_mask = dist < outlier_distance
    coord = coord[valid_mask]
    original_coord = original_coord[valid_mask]
    normal = normal[valid_mask]
    curvature = curvature[valid_mask]
    if coord.shape[0] == 0:
        raise ValueError("点云在离群点过滤后为空。")

    coord = center_shift_apply_z(coord)
    discrete_coords = np.floor(coord / voxel_size).astype(np.int32)
    _, unique_indices = np.unique(discrete_coords, axis=0, return_index=True)

    coord = coord[unique_indices].astype(np.float32)
    original_coord = original_coord[unique_indices].astype(np.float32)
    normal = normal[unique_indices].astype(np.float32)
    curvature = curvature[unique_indices].astype(np.float32)
    if coord.shape[0] == 0:
        raise ValueError("点云在体素化后为空。")

    grid_coord = np.floor(coord / voxel_size).astype(np.int32)
    grid_coord -= grid_coord.min(0)

    coord = center_shift_apply_z(coord)
    feat = np.concatenate([normal, curvature], axis=1).astype(np.float32)

    processed = {
        "coord": coord,
        "grid_coord": grid_coord,
        "feat": feat,
        "coord_feat": feat.copy(),
        "return_coord": original_coord.copy(),
    }
    meta = {
        "source_points": int(point_cloud_data.source_points),
        "valid_input_points": int(point_cloud_data.valid_input_points),
        "feature_valid_points": int(valid_nan.sum()),
        "valid_points": int(valid_mask.sum()),
        "voxel_points": int(coord.shape[0]),
        "feature_source": feature_source,
    }
    return processed, meta


def parse_npy_file(npy_path: str) -> ParsedPointCloudData:
    data = np.load(npy_path, allow_pickle=True)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("输入 .npy 至少需要 3 列: x, y, z")
    coord = data[:, 0:3].astype(np.float32)
    normal = None
    curvature = None
    feature_source = "estimated"
    if data.shape[1] >= 7:
        normal = data[:, 3:6].astype(np.float32)
        curvature = data[:, 6:7].astype(np.float32)
        feature_source = "provided"
    return ParsedPointCloudData(
        coord=coord,
        normal=normal,
        curvature=curvature,
        source_points=int(data.shape[0]),
        valid_input_points=int(data.shape[0]),
        feature_source=feature_source,
    )


def build_input_dict(
    coord: np.ndarray,
    feat: np.ndarray,
    grid_coord: np.ndarray,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    num_points = coord.shape[0]
    normals = feat[:, :3].astype(np.float32)
    curvature = feat[:, 3:4].astype(np.float32)
    return {
        "coord": torch.from_numpy(coord.astype(np.float32)).to(device),
        "feat": torch.from_numpy(feat.astype(np.float32)).to(device),
        "coord_feat": torch.from_numpy(feat.astype(np.float32)).to(device),
        "normal": torch.from_numpy(normals).to(device),
        "curvature": torch.from_numpy(curvature).to(device),
        "grid_coord": torch.from_numpy(grid_coord.astype(np.int32)).to(device),
        "offset": torch.tensor([num_points], dtype=torch.int32, device=device),
        "batch": torch.zeros(num_points, dtype=torch.long, device=device),
    }


def summarize_points_for_console(points: np.ndarray) -> str:
    if points.shape[0] == 0:
        return "[PigSegQt] 分割结果中没有猪点云。"

    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    return (
        "[PigSegQt] 猪点云包围盒范围 | "
        f"min=({xyz_min[0]:.4f}, {xyz_min[1]:.4f}, {xyz_min[2]:.4f}) | "
        f"max=({xyz_max[0]:.4f}, {xyz_max[1]:.4f}, {xyz_max[2]:.4f})"
    )
