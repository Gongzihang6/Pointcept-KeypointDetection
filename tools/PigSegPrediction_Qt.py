"""
作用：
提供基于 FastAPI 的猪体点云语义分割推理接口，供 Qt 程序调用。

功能：
1. 启动时按指定配置文件和权重文件加载 Pointcept 语义分割模型。
2. 接收 Qt 发送的完整场景点云：
   - 若 `num_channels >= 7`，直接使用 `xyz + normal + curvature`；
   - 若 `num_channels == 3`，再由服务端现场估计法向量和曲率。
3. 在服务端完成预处理，其中“坐标预处理流程”严格照抄 `infer_npy.py`：
   - NaN 清洗；
   - 中值去中心；
   - 飞点过滤；
   - 第一次 CenterShift；
   - 体素化；
   - 重算 grid_coord；
   - 第二次 CenterShift。
4. 执行语义分割推理，并返回原始坐标系下只包含猪主体的干净点云。
5. 在控制台输出猪类别概率统计，方便判断阈值或预处理是否异常。

说明：
1. 若 Qt 直接传入 7 通道特征，本脚本优先复用原始 `normal + curvature`，避免和训练分布偏离。
2. 只有在纯 xyz 输入时，才回退到与训练数据构建脚本一致的特征估计逻辑。
3. 为了严格对齐 `infer_npy.py`，体素尺寸使用命令行参数 `--voxel-size`，默认 20.0。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pointcept.models import build_model  # noqa: E402
from pointcept.utils.config import Config  # noqa: E402
from tools.pigseg_shared import build_input_dict, parse_point_cloud_buffer, preprocess_like_infer_npy, summarize_points_for_console


app = FastAPI(title="Pig Segmentation Prediction For Qt")


@dataclass
class ServerSettings:
    config_file: str
    weights: str
    host: str
    port: int
    device: str
    voxel_size: float
    max_nn: int
    outlier_distance: float
    pig_label: int
    pig_prob_thresh: float
    debug_save_input_dir: str


SERVER_SETTINGS: Optional[ServerSettings] = None
INFERENCE_SERVICE = None


def parse_args() -> ServerSettings:
    parser = argparse.ArgumentParser(description="Pig segmentation FastAPI service for Qt")
    parser.add_argument(
        "--config-file",
        default="configs/pigseg/semseg-ptv3-v1m1-0-base.py",
        help="训练时使用的配置文件路径",
    )
    parser.add_argument(
        "--weights",
        default="exp/PTV3_PigSeg_0511/model/model_best.pth",
        help="模型权重路径",
    )
    parser.add_argument("--host", default="0.0.0.0", help="服务监听地址")
    parser.add_argument("--port", type=int, default=8002, help="服务监听端口")
    parser.add_argument("--device", default="cuda", help="推理设备，例如 cuda 或 cpu")
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=20.0,
        help="完全对齐 infer_npy.py 的体素尺寸，默认 20.0",
    )
    parser.add_argument(
        "--max-nn",
        type=int,
        default=30,
        help="法向量和曲率估计统一使用的 KNN 邻域点数，默认 30",
    )
    parser.add_argument(
        "--outlier-distance",
        type=float,
        default=5000.0,
        help="和 infer_npy.py 一致的飞点过滤阈值，默认 5000.0",
    )
    parser.add_argument("--pig-label", type=int, default=1, help="猪类别的标签编号")
    parser.add_argument(
        "--pig-prob-thresh",
        type=float,
        default=-1.0,
        help="默认使用 argmax；若 >= 0 则按该阈值筛选猪类别概率",
    )
    parser.add_argument(
        "--debug-save-input-dir",
        default="debug_qt_inputs",
        help="可选，将每次请求解析后的原始 xyz 保存为 .npy，便于和训练集样本逐点对比",
    )
    args = parser.parse_args()
    return ServerSettings(
        config_file=args.config_file,
        weights=args.weights,
        host=args.host,
        port=args.port,
        device=args.device,
        voxel_size=args.voxel_size,
        max_nn=args.max_nn,
        outlier_distance=args.outlier_distance,
        pig_label=args.pig_label,
        pig_prob_thresh=args.pig_prob_thresh,
        debug_save_input_dir=os.path.abspath(args.debug_save_input_dir) if args.debug_save_input_dir else "",
    )


def remove_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key.replace("module.", ""): value for key, value in state_dict.items()}


def detect_model_family(cfg: Config) -> str:
    model_type = str(cfg.model.get("type", "")).lower()
    backbone_type = ""
    if "backbone" in cfg.model:
        backbone_type = str(cfg.model["backbone"].get("type", "")).lower()
    elif "backbone_conf" in cfg.model:
        backbone_type = str(cfg.model["backbone_conf"].get("type", "")).lower()

    signature = f"{model_type} {backbone_type}"
    if "oct" in signature:
        return "octformer"
    if "swin3d" in signature or "swin" in signature:
        return "swin3d"
    if "pt-v3" in signature or "ptv3" in signature:
        return "ptv3"
    return "generic"


def save_debug_input_npy(
    settings: ServerSettings,
    point_cloud_data,
) -> str:
    if not settings.debug_save_input_dir:
        return ""

    os.makedirs(settings.debug_save_input_dir, exist_ok=True)
    num_channels = 7 if point_cloud_data.normal is not None and point_cloud_data.curvature is not None else 3
    file_name = f"qt_input_points_{point_cloud_data.valid_input_points:06d}_c{num_channels}.npy"
    output_path = os.path.join(settings.debug_save_input_dir, file_name)

    suffix = 1
    while os.path.exists(output_path):
        file_name = f"qt_input_points_{point_cloud_data.valid_input_points:06d}_c{num_channels}_{suffix:03d}.npy"
        output_path = os.path.join(settings.debug_save_input_dir, file_name)
        suffix += 1

    if num_channels == 7:
        data = np.concatenate(
            [
                point_cloud_data.coord.astype(np.float32),
                point_cloud_data.normal.astype(np.float32),
                point_cloud_data.curvature.astype(np.float32),
            ],
            axis=1,
        )
    else:
        data = point_cloud_data.coord.astype(np.float32)
    np.save(output_path, data)
    return output_path


class PigSegInferenceService:
    def __init__(self, settings: ServerSettings):
        self.settings = settings
        self.cfg = Config.fromfile(settings.config_file)
        self.device = torch.device(
            settings.device if settings.device == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self.voxel_size = self._resolve_voxel_size()
        self.model_family = detect_model_family(self.cfg)
        self.model = self._load_model(settings.weights)

    def _resolve_voxel_size(self) -> float:
        if hasattr(self.cfg, "voxel_size"):
            return float(self.cfg.voxel_size)
        if hasattr(self.cfg, "data") and "test" in self.cfg.data and "test_cfg" in self.cfg.data.test:
            return float(self.cfg.data.test.test_cfg.voxelize.grid_size)
        return float(self.settings.voxel_size)

    def _load_model(self, weights_path: str) -> torch.nn.Module:
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")

        print(f"[PigSegQt] 正在构建模型: {self.cfg.model.type}")
        model = build_model(self.cfg.model)
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = remove_module_prefix(state_dict)
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        print(
            f"[PigSegQt] 模型加载完成 | family={self.model_family} | "
            f"voxel_size={self.voxel_size} | device={self.device}"
        )
        return model

    def _build_input_dict(
        self,
        coord: np.ndarray,
        feat: np.ndarray,
        grid_coord: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        return build_input_dict(coord=coord, feat=feat, grid_coord=grid_coord, device=self.device)

    def _predict_voxel_labels(
        self,
        data_dict: Dict[str, torch.Tensor],
        pig_label: int,
        pig_prob_thresh: float,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        with torch.no_grad():
            output = self.model(data_dict)
            if isinstance(output, dict):
                logits = output.get("seg_logits", output.get("pred"))
            else:
                logits = output

        if logits is None or not torch.is_tensor(logits):
            raise RuntimeError("模型没有返回可解析的分割 logits，请检查语义分割模型 forward 输出格式。")

        probs = torch.softmax(logits, dim=1)
        if pig_label >= probs.shape[1]:
            raise ValueError(f"pig_label={pig_label} 超出模型类别数 {probs.shape[1]}。")

        pig_probs = probs[:, pig_label]
        if pig_prob_thresh >= 0.0:
            preds = (pig_probs > pig_prob_thresh).long()
        else:
            preds = torch.argmax(logits, dim=1)

        prob_stats = {
            "pig_prob_min": float(pig_probs.min().item()),
            "pig_prob_max": float(pig_probs.max().item()),
            "pig_prob_mean": float(pig_probs.mean().item()),
            "voxel_pig_positive": int((preds == pig_label).sum().item()),
        }
        return preds.detach().cpu().numpy().astype(np.int32), prob_stats

    def predict(
        self,
        point_cloud_data: ParsedPointCloudData,
        pig_label: int,
        pig_prob_thresh: float,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        processed, preprocess_meta = preprocess_like_infer_npy(
            point_cloud_data=point_cloud_data,
            max_nn=self.settings.max_nn,
            outlier_distance=self.settings.outlier_distance,
            voxel_size=self.voxel_size,
        )

        data_dict = self._build_input_dict(
            coord=processed["coord"],
            feat=processed["feat"],
            grid_coord=processed["grid_coord"],
        )
        voxel_preds, prob_stats = self._predict_voxel_labels(
            data_dict=data_dict,
            pig_label=pig_label,
            pig_prob_thresh=pig_prob_thresh,
        )

        pig_mask = voxel_preds == pig_label
        pig_points = processed["return_coord"][pig_mask].astype(np.float32)
        meta = {
            "source_points": preprocess_meta["source_points"],
            "valid_input_points": preprocess_meta["valid_input_points"],
            "feature_valid_points": preprocess_meta["feature_valid_points"],
            "valid_points": preprocess_meta["valid_points"],
            "voxel_points": preprocess_meta["voxel_points"],
            "pig_points": int(pig_points.shape[0]),
            "voxel_size": float(self.voxel_size),
            "max_nn": int(self.settings.max_nn),
            "outlier_distance": float(self.settings.outlier_distance),
            "feature_source": preprocess_meta["feature_source"],
            "pig_prob_min": prob_stats["pig_prob_min"],
            "pig_prob_max": prob_stats["pig_prob_max"],
            "pig_prob_mean": prob_stats["pig_prob_mean"],
            "voxel_pig_positive": prob_stats["voxel_pig_positive"],
        }
        return pig_points, meta


@app.on_event("startup")
def load_model_on_startup() -> None:
    global INFERENCE_SERVICE
    if SERVER_SETTINGS is None:
        raise RuntimeError("服务启动参数未初始化。")
    INFERENCE_SERVICE = PigSegInferenceService(SERVER_SETTINGS)


@app.get("/health")
def health_check() -> Dict[str, object]:
    if INFERENCE_SERVICE is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成。")
    return {
        "status": "ok",
        "config_file": SERVER_SETTINGS.config_file,
        "weights": SERVER_SETTINGS.weights,
        "device": str(INFERENCE_SERVICE.device),
        "voxel_size": INFERENCE_SERVICE.voxel_size,
        "model_family": INFERENCE_SERVICE.model_family,
    }


@app.post("/predict")
async def predict_pointcloud(
    file: UploadFile = File(...),
    num_channels: int = Form(3),
    pig_label: int = Form(-1),
    pig_prob_thresh: float = Form(-999.0),
) -> Response:
    if INFERENCE_SERVICE is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成。")

    try:
        content = await file.read()
        point_cloud_data = parse_point_cloud_buffer(content, num_channels=num_channels)
        debug_input_path = save_debug_input_npy(SERVER_SETTINGS, point_cloud_data)

        if pig_label < 0:
            pig_label = SERVER_SETTINGS.pig_label
        if pig_prob_thresh <= -999.0:
            pig_prob_thresh = SERVER_SETTINGS.pig_prob_thresh

        pig_points, meta = INFERENCE_SERVICE.predict(
            point_cloud_data=point_cloud_data,
            pig_label=pig_label,
            pig_prob_thresh=pig_prob_thresh,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"推理失败: {exc}") from exc

    headers = {
        "X-Source-Point-Count": str(meta["source_points"]),
        "X-Valid-Point-Count": str(meta["valid_points"]),
        "X-Voxel-Point-Count": str(meta["voxel_points"]),
        "X-Pig-Point-Count": str(meta["pig_points"]),
    }
    print(
        "[PigSegQt] 本次请求预处理由脚本完成 | "
        f"输入点数={meta['source_points']} | "
        f"输入有效点数={meta['valid_input_points']} | "
        f"特征有效点数={meta['feature_valid_points']} | "
        f"离群点过滤后点数={meta['valid_points']} | "
        f"体素化后点数={meta['voxel_points']} | "
        f"voxel_size={meta['voxel_size']} | "
        f"knn={meta['max_nn']} | "
        f"outlier_distance={meta['outlier_distance']} | "
        f"feature_source={meta['feature_source']} | "
        f"pig_label={pig_label} | pig_prob_thresh={pig_prob_thresh} | "
        f"prediction_mode={'threshold' if pig_prob_thresh >= 0.0 else 'argmax'}"
    )
    print(
        "[PigSegQt] 猪类别概率统计 | "
        f"pig_prob_min={meta['pig_prob_min']:.6f} | "
        f"pig_prob_mean={meta['pig_prob_mean']:.6f} | "
        f"pig_prob_max={meta['pig_prob_max']:.6f} | "
        f"voxel_pig_positive={meta['voxel_pig_positive']}"
    )
    print(f"[PigSegQt] 分割得到的猪点云数量: {meta['pig_points']}")
    print(summarize_points_for_console(pig_points))
    if debug_input_path:
        print(f"[PigSegQt] 已保存本次请求的原始输入 .npy: {debug_input_path}")
    return Response(
        content=pig_points.astype(np.float32).tobytes(),
        media_type="application/octet-stream",
        headers=headers,
    )

"""
uv run tools/PigSegPrediction_Qt.py \
  --debug-save-input-dir debug_qt_inputs

"""
if __name__ == "__main__":
    SERVER_SETTINGS = parse_args()
    uvicorn.run(app, host=SERVER_SETTINGS.host, port=SERVER_SETTINGS.port)
