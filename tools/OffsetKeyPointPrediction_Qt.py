"""
作用：
提供基于 FastAPI 的基元 offset 关键点检测推理接口，供 Qt 程序通过 HTTP 调用。

功能：
1. 启动时按指定配置文件和权重文件加载 Pointcept 模型，支持 Swin3D / OctFormer / PTv3 等现有 offset 模型切换。
2. 接收 Qt 发送的完整猪三维点云，默认只要求输入 xyz 坐标。
3. 在服务端完成预处理，包括：
   - 非法点清洗；
   - 法向量估计；
   - 曲率计算；
   - 坐标中心化与归一化；
   - 体素下采样 / grid_coord 构造；
   - 将特征组装成 Pointcept 模型需要的输入格式。
4. 执行 offset 关键点推理，并将归一化空间下的预测坐标逆变换回输入点云原始坐标系。
5. 返回关键点坐标二进制流给 Qt，便于直接反序列化为 float32 数组。

怎么实现的：
1. 使用 FastAPI 暴露 `/predict` 接口，Qt 通过 multipart/form-data 上传点云二进制。
2. 服务启动时读取配置文件，自动推断 grid_size，并加载对应权重。
3. 预处理阶段用 Open3D 估计法向量，再基于局部协方差矩阵计算曲率。
4. 推理阶段统一构造包含 `coord`、`feat`、`grid_coord`、`offset` 等字段的 `data_dict`，
   兼容本仓库中常见的 offset 模型输入格式。
5. 对模型输出的逐点 offset + mask 进行聚合，恢复得到最终关键点世界坐标。

Qt 调用约定：
1. 上传字段名：`file`
2. 额外表单字段：
   - `num_channels`：每个点的通道数，默认 3。若 Qt 发送的是 xyz，则保持默认即可；
     若发送的是 xyzrgb / xyznormal 等，也只会取前 3 列作为坐标。
   - `agg_method`：`weighted` 或 `argmax`，默认 `weighted`
   - `mask_thresh`：`weighted` 模式下的掩码阈值，默认 0.5
3. 返回值：`application/octet-stream`，内容为 `(K, 3)` 的 float32 关键点坐标。

运行示例：
python tools/OffsetKeyPointPrediction_Qt.py \
    --config-file configs/my_dataset/offset_keypoint_octformer.py \
    --weights exp/offset_keypoint_octformer_0512/model/model_best.pth \
    --host 0.0.0.0 \
    --port 8001
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import open3d as o3d
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pointcept.models import build_model  # noqa: E402
from pointcept.utils.config import Config  # noqa: E402


app = FastAPI(title="Offset Keypoint Prediction For Qt")


@dataclass
class ServerSettings:
    config_file: str
    weights: str
    host: str
    port: int
    device: str
    normal_radius: float
    curvature_radius: float
    max_nn: int
    mask_thresh: float
    agg_method: str


SERVER_SETTINGS: Optional[ServerSettings] = None
INFERENCE_SERVICE = None


def parse_args() -> ServerSettings:
    parser = argparse.ArgumentParser(description="Offset keypoint FastAPI service for Qt")
    parser.add_argument(
        "--config-file",
        default="configs/my_dataset/offset_keypoint_octformer.py",
        help="训练时使用的配置文件路径",
    )
    parser.add_argument(
        "--weights",
        default="exp/offset_keypoint_octformer_0512/model/model_best.pth",
        help="模型权重路径",
    )
    parser.add_argument("--host", default="0.0.0.0", help="服务监听地址")
    parser.add_argument("--port", type=int, default=8001, help="服务监听端口")
    parser.add_argument("--device", default="cuda", help="推理设备，例如 cuda 或 cpu")
    parser.add_argument(
        "--normal-radius",
        type=float,
        default=35.0,
        help="法向量估计半径，单位与输入点云坐标一致，默认按毫米场景设置",
    )
    parser.add_argument(
        "--curvature-radius",
        type=float,
        default=35.0,
        help="曲率计算半径，单位与输入点云坐标一致，默认按毫米场景设置",
    )
    parser.add_argument(
        "--max-nn",
        type=int,
        default=30,
        help="法向量/曲率局部邻域最多搜索的点数",
    )
    parser.add_argument(
        "--mask-thresh",
        type=float,
        default=0.5,
        help="weighted 聚合时使用的 mask 阈值",
    )
    parser.add_argument(
        "--agg-method",
        choices=["weighted", "argmax"],
        default="weighted",
        help="关键点聚合策略",
    )
    args = parser.parse_args()
    return ServerSettings(
        config_file=args.config_file,
        weights=args.weights,
        host=args.host,
        port=args.port,
        device=args.device,
        normal_radius=args.normal_radius,
        curvature_radius=args.curvature_radius,
        max_nn=args.max_nn,
        mask_thresh=args.mask_thresh,
        agg_method=args.agg_method,
    )


def remove_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key.replace("module.", ""): value for key, value in state_dict.items()}


def infer_grid_size_from_cfg(cfg: Config) -> float:
    # 优先读取 offset 配置里显式定义的 grid_size_val。
    if hasattr(cfg, "grid_size_val"):
        return float(cfg.grid_size_val)

    model_cfg = getattr(cfg, "model", {})
    if "backbone_conf" in model_cfg and "base_grid_size" in model_cfg["backbone_conf"]:
        return float(model_cfg["backbone_conf"]["base_grid_size"])
    if "backbone" in model_cfg and "base_grid_size" in model_cfg["backbone"]:
        return float(model_cfg["backbone"]["base_grid_size"])

    # 再从数据流水线里读取 GridSample 的 grid_size。
    data_cfg = getattr(cfg, "data", {})
    for split in ("train", "val", "test"):
        if split not in data_cfg:
            continue
        for transform in data_cfg[split].get("transform", []):
            if transform.get("type") == "GridSample" and "grid_size" in transform:
                return float(transform["grid_size"])

    raise ValueError("无法从配置文件中推断 grid_size，请检查 config 中的 GridSample 或 backbone_conf 设置。")


def detect_model_family(cfg: Config) -> str:
    model_type = str(cfg.model.get("type", "")).lower()
    backbone_type = ""
    if "backbone_conf" in cfg.model:
        backbone_type = str(cfg.model["backbone_conf"].get("type", "")).lower()
    elif "backbone" in cfg.model:
        backbone_type = str(cfg.model["backbone"].get("type", "")).lower()

    signature = f"{model_type} {backbone_type}"
    if "oct" in signature:
        return "octformer"
    if "swin3d" in signature or "swin" in signature:
        return "swin3d"
    if "pt-v3" in signature or "ptv3" in signature:
        return "ptv3"
    return "generic"


def parse_point_cloud_buffer(content: bytes, num_channels: int) -> np.ndarray:
    if num_channels < 3:
        raise ValueError("num_channels 不能小于 3，至少需要 xyz 三列坐标。")
    raw = np.frombuffer(content, dtype=np.float32)
    if raw.size == 0:
        raise ValueError("收到的点云二进制为空。")
    if raw.size % num_channels != 0:
        raise ValueError(
            f"二进制长度无法按 num_channels={num_channels} 整除，请检查 Qt 发送的数据格式。"
        )
    points = raw.reshape(-1, num_channels)
    coord = points[:, :3].astype(np.float32)
    valid = np.isfinite(coord).all(axis=1)
    coord = coord[valid]
    if coord.shape[0] == 0:
        raise ValueError("点云在清理 NaN / Inf 后为空。")
    return coord


def estimate_normals_and_curvature(
    coord: np.ndarray,
    normal_radius: float,
    curvature_radius: float,
    max_nn: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if coord.shape[0] < 3:
        raise ValueError("有效点数小于 3，无法估计法向量和曲率。")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord.astype(np.float64))

    # 法向量估计直接交给 Open3D；如果点数充足，再做一致性定向以减少法向抖动。
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(normal_radius),
            max_nn=int(max_nn),
        )
    )
    if coord.shape[0] >= max(10, max_nn):
        try:
            pcd.orient_normals_consistent_tangent_plane(min(100, coord.shape[0] - 1))
        except RuntimeError:
            # 某些退化点云无法做一致性定向，此时保留 estimate_normals 的结果即可。
            pass

    normals = np.asarray(pcd.normals, dtype=np.float32)
    tree = o3d.geometry.KDTreeFlann(pcd)
    curvature = np.zeros((coord.shape[0], 1), dtype=np.float32)

    # 曲率采用局部协方差矩阵最小特征值占比，和仓库临时脚本中的做法保持一致。
    for index, point in enumerate(coord):
        _, neighbor_idx, _ = tree.search_hybrid_vector_3d(
            point.astype(np.float64),
            float(curvature_radius),
            int(max_nn),
        )
        if len(neighbor_idx) < 3:
            continue
        neighborhood = coord[np.asarray(neighbor_idx, dtype=np.int64)]
        centered = neighborhood - neighborhood.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / max(len(neighborhood) - 1, 1)
        eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)
        denom = float(eigenvalues.sum())
        if denom > 1e-12:
            curvature[index, 0] = float(eigenvalues[0] / denom)

    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
    curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
    return normals, curvature


def normalize_offset_points(coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    centroid = coord.mean(axis=0).astype(np.float32)
    centered = coord - centroid
    dist = np.linalg.norm(centered, axis=1)
    scale = float(dist.max()) if dist.shape[0] > 0 else 1.0
    if scale < 1e-6:
        scale = 1.0
    normalized = centered / scale
    return normalized.astype(np.float32), centroid, np.float32(scale)


def voxelize_points(
    coord: np.ndarray,
    feat: np.ndarray,
    grid_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if coord.shape[0] == 0:
        raise ValueError("点云为空，无法体素化。")

    discrete = np.floor(coord / grid_size).astype(np.int32)
    discrete -= discrete.min(axis=0, keepdims=True)

    # 训练流水线里的 GridSample(mode='train') 会保留每个体素的一个点。
    # 这里为了推理稳定性，固定取每个体素的首个点，并保留原顺序。
    _, unique_indices = np.unique(discrete, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)

    sampled_coord = coord[unique_indices].astype(np.float32)
    sampled_feat = feat[unique_indices].astype(np.float32)
    sampled_grid = np.floor(sampled_coord / grid_size).astype(np.int32)
    sampled_grid -= sampled_grid.min(axis=0, keepdims=True)
    return sampled_coord, sampled_feat, sampled_grid


def format_keypoints_for_console(keypoints: np.ndarray) -> str:
    lines = ["[OffsetQt] 关键点坐标(原始坐标系):"]
    for keypoint_id, point in enumerate(keypoints):
        lines.append(
            f"  KP{keypoint_id}: x={point[0]:.4f}, y={point[1]:.4f}, z={point[2]:.4f}"
        )
    return "\n".join(lines)


class OffsetKeypointInferenceService:
    def __init__(self, settings: ServerSettings):
        self.settings = settings
        self.cfg = Config.fromfile(settings.config_file)
        self.device = torch.device(
            settings.device if settings.device == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self.grid_size = infer_grid_size_from_cfg(self.cfg)
        self.model_family = detect_model_family(self.cfg)
        self.num_keypoints = int(self.cfg.model.get("num_keypoints", 6))
        self.model = self._load_model(settings.weights)

    def _load_model(self, weights_path: str) -> torch.nn.Module:
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")

        print(f"[OffsetQt] 正在构建模型: {self.cfg.model.type}")
        model = build_model(self.cfg.model)
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = remove_module_prefix(state_dict)
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        print(
            f"[OffsetQt] 模型加载完成 | family={self.model_family} | "
            f"grid_size={self.grid_size} | device={self.device}"
        )
        return model

    def _build_input_dict(
        self,
        coord: np.ndarray,
        normals: np.ndarray,
        curvature: np.ndarray,
        grid_coord: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        feat = np.concatenate([normals, curvature], axis=1).astype(np.float32)
        num_points = coord.shape[0]

        data_dict = {
            "coord": torch.from_numpy(coord).to(self.device),
            "feat": torch.from_numpy(feat).to(self.device),
            "coord_feat": torch.from_numpy(feat).to(self.device),
            "normal": torch.from_numpy(normals.astype(np.float32)).to(self.device),
            "curvature": torch.from_numpy(curvature.astype(np.float32)).to(self.device),
            "grid_coord": torch.from_numpy(grid_coord.astype(np.int32)).to(self.device),
            "offset": torch.tensor([num_points], dtype=torch.int32, device=self.device),
            "batch": torch.zeros(num_points, dtype=torch.long, device=self.device),
            "grid_size": torch.tensor(self.grid_size, dtype=torch.float32, device=self.device),
        }
        return data_dict

    def _aggregate_keypoints(
        self,
        pred_tensor: torch.Tensor,
        coord_normalized: np.ndarray,
        centroid: np.ndarray,
        scale: float,
        agg_method: str,
        mask_thresh: float,
    ) -> np.ndarray:
        pred = pred_tensor.detach().cpu().numpy()
        pred_offset = pred[..., :3]
        pred_mask = pred[..., 3]

        coord_world = coord_normalized * scale + centroid
        keypoints = np.zeros((self.num_keypoints, 3), dtype=np.float32)

        for keypoint_id in range(self.num_keypoints):
            probs = pred_mask[:, keypoint_id]
            if probs.shape[0] == 0:
                continue

            if agg_method == "weighted":
                valid = probs > mask_thresh
                if np.any(valid):
                    candidate_points = coord_world[valid] + pred_offset[valid, keypoint_id] * scale
                    weights = probs[valid]
                    weights_sum = float(weights.sum())
                    if weights_sum > 1e-12:
                        weights = weights / weights_sum
                        keypoints[keypoint_id] = np.sum(
                            candidate_points * weights[:, None],
                            axis=0,
                        ).astype(np.float32)
                        continue

            best_index = int(np.argmax(probs))
            keypoints[keypoint_id] = (
                coord_world[best_index] + pred_offset[best_index, keypoint_id] * scale
            ).astype(np.float32)

        return keypoints

    def predict(
        self,
        coord_raw: np.ndarray,
        agg_method: str,
        mask_thresh: float,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        raw_point_count = int(coord_raw.shape[0])
        normals_raw, curvature_raw = estimate_normals_and_curvature(
            coord=coord_raw,
            normal_radius=self.settings.normal_radius,
            curvature_radius=self.settings.curvature_radius,
            max_nn=self.settings.max_nn,
        )

        coord_normalized, centroid, scale = normalize_offset_points(coord_raw)
        sampled_coord, sampled_feat, sampled_grid = voxelize_points(
            coord=coord_normalized,
            feat=np.concatenate([normals_raw, curvature_raw], axis=1).astype(np.float32),
            grid_size=self.grid_size,
        )

        sampled_normals = sampled_feat[:, :3].astype(np.float32)
        sampled_curvature = sampled_feat[:, 3:4].astype(np.float32)
        data_dict = self._build_input_dict(
            coord=sampled_coord,
            normals=sampled_normals,
            curvature=sampled_curvature,
            grid_coord=sampled_grid,
        )

        with torch.no_grad():
            output = self.model(data_dict)
            pred_tensor = output["pred"] if isinstance(output, dict) and "pred" in output else output
            if not torch.is_tensor(pred_tensor):
                raise RuntimeError("模型没有返回可解析的预测张量，请检查 offset 模型 forward 输出格式。")

        keypoints = self._aggregate_keypoints(
            pred_tensor=pred_tensor,
            coord_normalized=sampled_coord,
            centroid=centroid,
            scale=float(scale),
            agg_method=agg_method,
            mask_thresh=mask_thresh,
        )
        meta = {
            "source_points": raw_point_count,
            "used_points": int(sampled_coord.shape[0]),
            "num_keypoints": int(keypoints.shape[0]),
            "normal_radius": float(self.settings.normal_radius),
            "curvature_radius": float(self.settings.curvature_radius),
            "scale": float(scale),
        }
        return keypoints, meta


@app.on_event("startup")
def load_model_on_startup() -> None:
    global INFERENCE_SERVICE
    if SERVER_SETTINGS is None:
        raise RuntimeError("服务启动参数未初始化。")
    INFERENCE_SERVICE = OffsetKeypointInferenceService(SERVER_SETTINGS)


@app.get("/health")
def health_check() -> Dict[str, object]:
    if INFERENCE_SERVICE is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成。")
    return {
        "status": "ok",
        "config_file": SERVER_SETTINGS.config_file,
        "weights": SERVER_SETTINGS.weights,
        "device": str(INFERENCE_SERVICE.device),
        "grid_size": INFERENCE_SERVICE.grid_size,
        "model_family": INFERENCE_SERVICE.model_family,
        "num_keypoints": INFERENCE_SERVICE.num_keypoints,
    }


@app.post("/predict")
async def predict_pointcloud(
    file: UploadFile = File(...),
    num_channels: int = Form(3),
    agg_method: str = Form(""),
    mask_thresh: float = Form(-1.0),
) -> Response:
    if INFERENCE_SERVICE is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成。")

    try:
        content = await file.read()
        coord_raw = parse_point_cloud_buffer(content, num_channels=num_channels)

        if agg_method == "":
            agg_method = SERVER_SETTINGS.agg_method
        if agg_method not in {"weighted", "argmax"}:
            raise ValueError("agg_method 只支持 weighted 或 argmax。")
        if mask_thresh < 0:
            mask_thresh = SERVER_SETTINGS.mask_thresh

        keypoints, meta = INFERENCE_SERVICE.predict(
            coord_raw=coord_raw,
            agg_method=agg_method,
            mask_thresh=mask_thresh,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"推理失败: {exc}") from exc

    headers = {
        "X-Keypoint-Count": str(meta["num_keypoints"]),
        "X-Source-Point-Count": str(meta["source_points"]),
        "X-Used-Point-Count": str(meta["used_points"]),
    }
    print(
        "[OffsetQt] 本次请求预处理由脚本完成 | "
        f"输入点数={meta['source_points']} | "
        f"体素化后点数={meta['used_points']} | "
        f"normal_radius={meta['normal_radius']} | "
        f"curvature_radius={meta['curvature_radius']} | "
        f"scale={meta['scale']:.6f} | "
        f"agg_method={agg_method} | mask_thresh={mask_thresh}"
    )
    print(format_keypoints_for_console(keypoints))
    return Response(
        content=keypoints.astype(np.float32).tobytes(),
        media_type="application/octet-stream",
        headers=headers,
    )


if __name__ == "__main__":
    SERVER_SETTINGS = parse_args()
    uvicorn.run(app, host=SERVER_SETTINGS.host, port=SERVER_SETTINGS.port)
