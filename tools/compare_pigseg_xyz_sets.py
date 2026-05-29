import argparse
import os
from typing import Dict

import numpy as np


def load_xyz(file_path: str, num_channels: int) -> np.ndarray:
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if file_path.endswith(".npy"):
        data = np.load(file_path, allow_pickle=True)
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(f"{file_path} 不是至少包含 3 列坐标的二维数组。")
        return data[:, :3].astype(np.float32)

    raw = np.fromfile(file_path, dtype=np.float32)
    if raw.size == 0:
        raise ValueError(f"{file_path} 为空，无法解析点云。")
    if num_channels < 3:
        raise ValueError("读取原始二进制时，num_channels 不能小于 3。")
    if raw.size % num_channels != 0:
        raise ValueError(
            f"{file_path} 的 float32 元素个数无法被 num_channels={num_channels} 整除。"
        )
    data = raw.reshape(-1, num_channels)
    return data[:, :3].astype(np.float32)


def summarize_xyz(name: str, xyz: np.ndarray) -> Dict[str, float]:
    return {
        f"{name}_points": float(xyz.shape[0]),
        f"{name}_x_min": float(xyz[:, 0].min()),
        f"{name}_x_max": float(xyz[:, 0].max()),
        f"{name}_y_min": float(xyz[:, 1].min()),
        f"{name}_y_max": float(xyz[:, 1].max()),
        f"{name}_z_min": float(xyz[:, 2].min()),
        f"{name}_z_max": float(xyz[:, 2].max()),
    }


def rows_to_structured(rows: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(rows)
    dtype = np.dtype([("x", contiguous.dtype), ("y", contiguous.dtype), ("z", contiguous.dtype)])
    return contiguous.view(dtype).reshape(-1)


def sort_rows(rows: np.ndarray) -> np.ndarray:
    order = np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))
    return rows[order]


def unique_count(rows: np.ndarray) -> int:
    return int(np.unique(rows_to_structured(rows)).shape[0])


def quantize_rows(rows: np.ndarray, tolerance: float) -> np.ndarray:
    if tolerance <= 0:
        raise ValueError("tolerance 必须大于 0。")
    return np.round(rows / tolerance).astype(np.int64)


def compare_in_same_order(source: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    common = min(source.shape[0], target.shape[0])
    if common == 0:
        return {
            "same_order_equal": 0.0,
            "same_order_exact_match_ratio": 0.0,
            "same_order_max_abs_diff": 0.0,
            "same_order_mean_abs_diff": 0.0,
        }

    src = source[:common]
    tgt = target[:common]
    point_equal_mask = np.all(src == tgt, axis=1)
    abs_diff = np.abs(src - tgt)
    return {
        "same_order_equal": float(source.shape == target.shape and np.array_equal(source, target)),
        "same_order_exact_match_ratio": float(np.mean(point_equal_mask)),
        "same_order_max_abs_diff": float(abs_diff.max()),
        "same_order_mean_abs_diff": float(abs_diff.mean()),
    }


def compare_exact_set(source: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    if source.shape[0] != target.shape[0]:
        return {
            "exact_set_equal": 0.0,
            "exact_set_size_equal": 0.0,
        }

    source_sorted = sort_rows(source)
    target_sorted = sort_rows(target)
    return {
        "exact_set_equal": float(np.array_equal(source_sorted, target_sorted)),
        "exact_set_size_equal": 1.0,
    }


def compare_quantized_set(source: np.ndarray, target: np.ndarray, tolerance: float) -> Dict[str, float]:
    source_q = quantize_rows(source, tolerance)
    target_q = quantize_rows(target, tolerance)

    source_unique, source_counts = np.unique(rows_to_structured(source_q), return_counts=True)
    target_unique, target_counts = np.unique(rows_to_structured(target_q), return_counts=True)

    same_unique = np.array_equal(source_unique, target_unique)
    same_counts = np.array_equal(source_counts, target_counts)

    source_unique_count = source_unique.shape[0]
    target_unique_count = target_unique.shape[0]
    source_total_count = int(source_q.shape[0])
    target_total_count = int(target_q.shape[0])

    return {
        "quantized_set_equal": float(same_unique and same_counts),
        "quantized_unique_count_source": float(source_unique_count),
        "quantized_unique_count_target": float(target_unique_count),
        "quantized_total_count_source": float(source_total_count),
        "quantized_total_count_target": float(target_total_count),
    }


def format_metrics(title: str, metrics: Dict[str, float]) -> str:
    lines = [title]
    for key in sorted(metrics):
        value = metrics[key]
        if abs(value - round(value)) < 1e-12:
            lines.append(f"  {key}: {int(round(value))}")
        else:
            lines.append(f"  {key}: {value:.8f}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare whether two point clouds are the same set of xyz points.")
    parser.add_argument("--source", required=True, help="第一份点云文件路径，支持 .npy 或原始 float32 二进制")
    parser.add_argument("--target", required=True, help="第二份点云文件路径，支持 .npy 或原始 float32 二进制")
    parser.add_argument(
        "--source-num-channels",
        type=int,
        default=3,
        help="当 source 不是 .npy 时，每个点的通道数，默认 3",
    )
    parser.add_argument(
        "--target-num-channels",
        type=int,
        default=3,
        help="当 target 不是 .npy 时，每个点的通道数，默认 3",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="比较量化后点集合是否一致的坐标容差，默认 1e-4",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = load_xyz(args.source, args.source_num_channels)
    target = load_xyz(args.target, args.target_num_channels)

    print(f"source: {os.path.abspath(args.source)}")
    print(f"target: {os.path.abspath(args.target)}")
    print(f"tolerance: {args.tolerance}")
    print()

    print(format_metrics("Source Summary", summarize_xyz("source", source)))
    print()
    print(format_metrics("Target Summary", summarize_xyz("target", target)))
    print()
    print(format_metrics("Same Order Comparison", compare_in_same_order(source, target)))
    print()
    print(format_metrics("Exact Set Comparison", compare_exact_set(source, target)))
    print()
    print(
        format_metrics(
            "Quantized Set Comparison",
            compare_quantized_set(source, target, args.tolerance),
        )
    )
    print()
    print(
        format_metrics(
            "Duplicate Summary",
            {
                "source_unique_points": float(unique_count(source)),
                "target_unique_points": float(unique_count(target)),
                "source_duplicate_points": float(source.shape[0] - unique_count(source)),
                "target_duplicate_points": float(target.shape[0] - unique_count(target)),
            },
        )
    )


if __name__ == "__main__":
    main()
