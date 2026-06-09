import argparse
import os

import numpy as np
import open3d as o3d


DEFAULT_COLORS = np.array(
    [
        [0.90, 0.10, 0.10],
        [0.10, 0.70, 0.10],
        [0.10, 0.40, 0.95],
        [0.95, 0.70, 0.10],
        [0.70, 0.20, 0.85],
        [0.10, 0.80, 0.80],
    ],
    dtype=np.float64,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize keypoint offset labels saved as (N, 6, 4), where the last "
            "channel is [dx, dy, dz, mask]."
        )
    )
    parser.add_argument(
        "--offset-file",
        type=str,
        help="Path to *_keypoint_offset.npy",
    )
    parser.add_argument(
        "--pointcloud-file",
        type=str,
        default=None,
        help="Optional path to the corresponding point cloud .npy. If omitted, infer from dataset layout.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="Point size in Open3D visualization.",
    )
    parser.add_argument(
        "--keypoint-radius",
        type=float,
        default=None,
        help="Sphere radius for each inferred keypoint. Auto-estimated when omitted.",
    )
    parser.add_argument(
        "--arrow-cylinder-radius",
        type=float,
        default=None,
        help="Arrow cylinder radius. Auto-estimated when omitted.",
    )
    parser.add_argument(
        "--arrow-cone-radius",
        type=float,
        default=None,
        help="Arrow cone radius. Auto-estimated when omitted.",
    )
    parser.add_argument(
        "--arrow-cone-ratio",
        type=float,
        default=0.1,
        help="Cone length ratio in each arrow.",
    )
    parser.add_argument(
        "--max-arrows-per-kp",
        type=int,
        default=100,
        help="Maximum number of mask=1 arrows to draw for each keypoint. Set <=0 to disable arrows.",
    )
    parser.add_argument(
        "--arrow-style",
        type=str,
        choices=("line", "mesh"),
        default="line",
        help=(
            "Arrow rendering style. 'line' batches all arrows into Open3D LineSet and is much faster; "
            "'mesh' keeps the original 3D arrow meshes but is slow for many arrows."
        ),
    )
    parser.add_argument(
        "--arrow-sample-mode",
        type=str,
        choices=("random", "stride"),
        default="random",
        help="How to choose arrows when valid points exceed --max-arrows-per-kp.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed used by --arrow-sample-mode random and point downsampling.",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=2.0,
        help="Line width when --arrow-style line is used. Some Open3D backends may ignore this value.",
    )
    parser.add_argument(
        "--line-arrowhead-ratio",
        type=float,
        default=0.08,
        help=(
            "Arrowhead length ratio for --arrow-style line. "
            "Set <=0 to draw plain line segments without arrowheads."
        ),
    )
    parser.add_argument(
        "--keypoint-index",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of keypoint indices to visualize, e.g. --keypoint-index 0 3 5",
    )
    parser.add_argument(
        "--hide-background",
        action="store_true",
        help="Only show the mask=1 source points instead of the full point cloud.",
    )
    parser.add_argument(
        "--point-sample-rate",
        type=float,
        default=1.0,
        help=(
            "Randomly keep this fraction of background points. "
            "Only applies when --hide-background is not set."
        ),
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=200000,
        help=(
            "Maximum number of background points to render. "
            "Use <=0 for no cap. Only applies when --hide-background is not set."
        ),
    )
    parser.add_argument(
        "--active-point-sample-rate",
        type=float,
        default=1.0,
        help=(
            "Randomly keep this fraction of mask=1 source points when --hide-background is set. "
            "This does not affect keypoint inference or arrow sampling."
        ),
    )
    parser.add_argument(
        "--max-active-points",
        type=int,
        default=50000,
        help=(
            "Maximum number of mask=1 source points to render when --hide-background is set. "
            "Use <=0 for no cap."
        ),
    )
    return parser.parse_args()


def infer_pointcloud_path(offset_file):
    offset_dir = os.path.dirname(os.path.abspath(offset_file))
    filename = os.path.basename(offset_file)

    if filename.endswith("_keypoint_offset.npy"):
        stem = filename[: -len("_keypoint_offset.npy")]
    else:
        stem = os.path.splitext(filename)[0]

    split_dir = os.path.dirname(offset_dir)
    candidate = os.path.join(split_dir, "pointclouds", f"{stem}.npy")
    return candidate


def load_pointcloud(pointcloud_file):
    if not os.path.exists(pointcloud_file):
        raise FileNotFoundError(f"Point cloud file not found: {pointcloud_file}")

    data = np.load(pointcloud_file)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(
            f"Point cloud should have shape (N, C>=3), but got {data.shape} from {pointcloud_file}"
        )
    # Open3D 最终会把点坐标转换成 Vector3dVector；这里先保留原始浮点精度，
    # 避免在读入阶段就把整个点云强制拷贝成 float64，减少大点云的内存压力。
    return data[:, :3]


def load_offset(offset_file):
    if not os.path.exists(offset_file):
        raise FileNotFoundError(f"Offset file not found: {offset_file}")

    data = np.load(offset_file)
    if data.ndim != 3 or data.shape[2] != 4:
        raise ValueError(
            f"Offset label should have shape (N, K, 4), but got {data.shape} from {offset_file}"
        )
    # offset 标签通常是 float32，形状为 (N, K, 4)。不要在这里 astype(float64)，
    # 否则一个大文件会立刻产生 2 倍内存占用；需要给 Open3D 的小片段再局部转换。
    return data


def get_bbox_scale(coords):
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    diag = np.linalg.norm(bbox_max - bbox_min)
    return max(diag, 1.0)


def axis_angle_from_z(direction):
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    unit_direction = direction / np.linalg.norm(direction)
    cross = np.cross(z_axis, unit_direction)
    cross_norm = np.linalg.norm(cross)
    dot = float(np.clip(np.dot(z_axis, unit_direction), -1.0, 1.0))

    if cross_norm < 1e-12:
        if dot > 0:
            return np.zeros(3, dtype=np.float64)
        return np.array([1.0, 0.0, 0.0], dtype=np.float64) * np.pi

    axis = cross / cross_norm
    angle = np.arccos(dot)
    return axis * angle


def sample_indices(indices, max_count, sample_rate, rng, mode="random"):
    """从候选索引中抽样，统一处理点云下采样和箭头下采样。

    Args:
        indices (np.ndarray): 候选点索引，一维整数数组。
        max_count (int): 最多保留多少个点；<=0 表示不按数量截断。
        sample_rate (float): 保留比例，范围建议为 (0, 1]；>=1 表示不按比例截断。
        rng (np.random.Generator): NumPy 随机数生成器，用于可复现随机抽样。
        mode (str): "random" 随机抽样；"stride" 按等间隔抽样。

    Returns:
        np.ndarray: 抽样后的索引数组。为了让 Open3D 显示和日志更稳定，随机结果会排序。
    """
    indices = np.asarray(indices, dtype=np.int64)
    num_candidates = len(indices)
    if num_candidates == 0:
        return indices

    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be > 0, got {sample_rate}")

    # 先由比例得到目标数量，再叠加 max_count 上限。这样用户可以只用其中一个参数，
    # 也可以二者同时使用，例如 --point-sample-rate 0.2 --max-points 100000。
    target_count = num_candidates
    if sample_rate < 1.0:
        target_count = max(1, int(np.ceil(num_candidates * sample_rate)))
    if max_count is not None and max_count > 0:
        target_count = min(target_count, int(max_count))

    if target_count >= num_candidates:
        return indices

    if mode == "stride":
        sampled_pos = np.linspace(0, num_candidates - 1, target_count, dtype=np.int64)
        return indices[sampled_pos]
    if mode != "random":
        raise ValueError(f"Unsupported sample mode: {mode}")

    sampled = rng.choice(indices, size=target_count, replace=False)
    sampled.sort()
    return sampled


def create_arrow(start, end, color, cylinder_radius, cone_radius, cone_ratio):
    vec = end - start
    length = np.linalg.norm(vec)
    if length < 1e-8:
        return None

    cone_ratio = float(np.clip(cone_ratio, 0.05, 0.45))
    cone_height = min(length * cone_ratio, length * 0.45)
    cylinder_height = max(length - cone_height, 1e-6)
    if cylinder_height <= 1e-6:
        cylinder_height = length * 0.6
        cone_height = length - cylinder_height

    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=float(cylinder_radius),
        cone_radius=float(cone_radius),
        cylinder_height=float(cylinder_height),
        cone_height=float(cone_height),
    )
    arrow.compute_vertex_normals()
    arrow.paint_uniform_color(color.tolist())

    rotvec = axis_angle_from_z(vec)
    rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(rotvec)
    arrow.rotate(rotation, center=np.zeros(3))
    arrow.translate(start)
    return arrow


def create_arrow_lines(coords, offsets, sampled_idx, kp_idx, color, arrowhead_ratio):
    """用一个 LineSet 批量绘制同一关键点的所有 offset 向量。

    Open3D 中 TriangleMesh 箭头的创建、法线计算和逐个 add_geometry 都比较慢。
    对检查 offset 方向而言，线段通常已经足够直观；把 M 根箭头合并为一个
    LineSet 后，只需要创建 1 个 geometry，渲染开销会小很多。
    """
    if len(sampled_idx) == 0:
        return None

    starts = coords[sampled_idx]
    ends = starts + offsets[sampled_idx, kp_idx]
    lengths = np.linalg.norm(ends - starts, axis=1)
    keep = lengths > 1e-8
    if not np.any(keep):
        return None

    starts = starts[keep]
    ends = ends[keep]
    lengths = lengths[keep]
    num_lines = len(starts)

    if arrowhead_ratio <= 0:
        # LineSet 的点数组按 [start0, end0, start1, end1, ...] 排列，
        # lines 只需连接相邻的 2 个点。只转换抽样后的点，避免大数组 float64 拷贝。
        line_points = np.empty((num_lines * 2, 3), dtype=np.float64)
        line_points[0::2] = starts
        line_points[1::2] = ends
        lines = np.arange(num_lines * 2, dtype=np.int32).reshape(num_lines, 2)
        colors = np.repeat(color[np.newaxis, :], num_lines, axis=0)
    else:
        # 用 3 条线表示一根箭头：主体 + 两条箭头头部线。这样仍然是一个 LineSet，
        # 但比纯线段更容易看出 offset 的方向。
        unit_dirs = (ends - starts) / lengths[:, np.newaxis]
        refs = np.repeat(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), num_lines, axis=0)
        near_z = np.abs(unit_dirs[:, 2]) > 0.9
        refs[near_z] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        side_dirs = np.cross(unit_dirs, refs)
        side_dirs /= np.linalg.norm(side_dirs, axis=1, keepdims=True)

        head_ratio = float(np.clip(arrowhead_ratio, 0.02, 0.35))
        head_lengths = lengths[:, np.newaxis] * head_ratio
        head_base = ends - unit_dirs * head_lengths
        head_side = side_dirs * head_lengths * 0.45
        head_1 = head_base + head_side
        head_2 = head_base - head_side

        # 每根箭头占 4 个点：[start, end, head_1, head_2]，
        # 对应 3 条线：[start-end], [end-head_1], [end-head_2]。
        line_points = np.empty((num_lines * 4, 3), dtype=np.float64)
        line_points[0::4] = starts
        line_points[1::4] = ends
        line_points[2::4] = head_1
        line_points[3::4] = head_2
        base = np.arange(num_lines, dtype=np.int32) * 4
        lines = np.empty((num_lines * 3, 2), dtype=np.int32)
        lines[0::3] = np.column_stack((base, base + 1))
        lines[1::3] = np.column_stack((base + 1, base + 2))
        lines[2::3] = np.column_stack((base + 1, base + 3))
        colors = np.repeat(color[np.newaxis, :], num_lines * 3, axis=0)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def build_visual_geometries(
    coords,
    offset_data,
    selected_kps,
    hide_background,
    point_size,
    keypoint_radius,
    arrow_cylinder_radius,
    arrow_cone_radius,
    arrow_cone_ratio,
    max_arrows_per_kp,
    arrow_style,
    arrow_sample_mode,
    point_sample_rate,
    max_points,
    active_point_sample_rate,
    max_active_points,
    random_seed,
    line_arrowhead_ratio,
):
    offsets = offset_data[..., :3]
    masks = offset_data[..., 3] > 0.5
    num_points, num_keypoints, _ = offset_data.shape
    bbox_scale = get_bbox_scale(coords)
    rng = np.random.default_rng(random_seed)

    keypoint_radius = keypoint_radius or bbox_scale * 0.015
    arrow_cylinder_radius = arrow_cylinder_radius or bbox_scale * 0.0015
    arrow_cone_radius = arrow_cone_radius or bbox_scale * 0.003

    if selected_kps is None:
        selected_kps = list(range(num_keypoints))

    invalid_kps = [idx for idx in selected_kps if idx < 0 or idx >= num_keypoints]
    if invalid_kps:
        raise ValueError(
            f"Invalid keypoint indices {invalid_kps}, available range is [0, {num_keypoints - 1}]"
        )

    geometries = []
    active_union = np.zeros(num_points, dtype=bool)
    keypoint_summaries = []
    total_drawn_arrows = 0

    for kp_idx in selected_kps:
        valid_idx = np.where(masks[:, kp_idx])[0]
        color = DEFAULT_COLORS[kp_idx % len(DEFAULT_COLORS)]

        if len(valid_idx) == 0:
            keypoint_summaries.append((kp_idx, 0, None, None))
            continue

        active_union[valid_idx] = True

        # 关键点坐标要用全部 mask=1 点推断，不能用可视化抽样后的点推断；
        # 否则随机抽样会让日志中的 inferred_coord 产生额外抖动。
        candidate_targets = coords[valid_idx] + offsets[valid_idx, kp_idx]
        keypoint_coord = candidate_targets.mean(axis=0)
        consistency = np.linalg.norm(candidate_targets - keypoint_coord, axis=1).max()
        keypoint_summaries.append((kp_idx, len(valid_idx), keypoint_coord, consistency))

        if max_arrows_per_kp > 0:
            sampled_idx = sample_indices(
                indices=valid_idx,
                max_count=max_arrows_per_kp,
                sample_rate=1.0,
                rng=rng,
                mode=arrow_sample_mode,
            )
            total_drawn_arrows += len(sampled_idx)

            if arrow_style == "line":
                line_set = create_arrow_lines(
                    coords=coords,
                    offsets=offsets,
                    sampled_idx=sampled_idx,
                    kp_idx=kp_idx,
                    color=color,
                    arrowhead_ratio=line_arrowhead_ratio,
                )
                if line_set is not None:
                    geometries.append(line_set)
            else:
                # mesh 模式保留原始三维箭头效果，但每根箭头都是一个 TriangleMesh。
                # 只建议在 --max-arrows-per-kp 很小的时候使用。
                for point_idx in sampled_idx:
                    start = coords[point_idx]
                    end = start + offsets[point_idx, kp_idx]
                    arrow = create_arrow(
                        start=start,
                        end=end,
                        color=color,
                        cylinder_radius=arrow_cylinder_radius,
                        cone_radius=arrow_cone_radius,
                        cone_ratio=arrow_cone_ratio,
                    )
                    if arrow is not None:
                        geometries.append(arrow)

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=float(keypoint_radius))
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(color.tolist())
        sphere.translate(keypoint_coord)
        geometries.append(sphere)

    if hide_background:
        # 只看 mask=1 区域时，点数量通常少很多；仍保留独立下采样参数，
        # 防止半径很大时 active 点过多导致交互卡顿。
        visible_idx = np.where(active_union)[0]
        visible_idx = sample_indices(
            indices=visible_idx,
            max_count=max_active_points,
            sample_rate=active_point_sample_rate,
            rng=rng,
            mode="random",
        )
    else:
        # 背景点云只用于提供空间上下文，默认最多渲染 20 万点。
        # 大多数 Open3D 交互卡顿来自这里和大量 mesh 箭头。
        visible_idx = sample_indices(
            indices=np.arange(num_points, dtype=np.int64),
            max_count=max_points,
            sample_rate=point_sample_rate,
            rng=rng,
            mode="random",
        )

    visible_points = coords[visible_idx]
    # 只给最终要显示的点创建颜色数组，避免为完整点云额外分配 N x 3 的 float64。
    # 背景点默认黑色；mask=1 的点按关键点颜色覆盖，多个关键点重叠时后面的颜色覆盖前面的。
    visible_colors = np.zeros((len(visible_idx), 3), dtype=np.float64)
    for kp_idx in selected_kps:
        color = DEFAULT_COLORS[kp_idx % len(DEFAULT_COLORS)]
        visible_colors[masks[visible_idx, kp_idx]] = color

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(visible_points, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(visible_colors)
    geometries.insert(0, pcd)

    render_stats = {
        "visible_points": len(visible_idx),
        "total_points": num_points,
        "drawn_arrows": total_drawn_arrows,
        "arrow_style": arrow_style,
    }
    return geometries, keypoint_summaries, point_size, render_stats


def visualize(geometries, point_size, line_width, window_name):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1400, height=900)
    for geom in geometries:
        vis.add_geometry(geom)

    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.line_width = line_width
    render_option.background_color = np.asarray([1.0, 1.0, 1.0])

    vis.run()
    vis.destroy_window()


def main():
    args = parse_args()

    offset_file = os.path.abspath(args.offset_file)
    pointcloud_file = (
        os.path.abspath(args.pointcloud_file)
        if args.pointcloud_file is not None
        else infer_pointcloud_path(offset_file)
    )

    offset_data = load_offset(offset_file)
    coords = load_pointcloud(pointcloud_file)

    if offset_data.shape[0] != coords.shape[0]:
        raise ValueError(
            "Point count mismatch: "
            f"offset has N={offset_data.shape[0]}, point cloud has N={coords.shape[0]}"
        )

    geometries, summaries, point_size, render_stats = build_visual_geometries(
        coords=coords,
        offset_data=offset_data,
        selected_kps=args.keypoint_index,
        hide_background=args.hide_background,
        point_size=args.point_size,
        keypoint_radius=args.keypoint_radius,
        arrow_cylinder_radius=args.arrow_cylinder_radius,
        arrow_cone_radius=args.arrow_cone_radius,
        arrow_cone_ratio=args.arrow_cone_ratio,
        max_arrows_per_kp=args.max_arrows_per_kp,
        arrow_style=args.arrow_style,
        arrow_sample_mode=args.arrow_sample_mode,
        point_sample_rate=args.point_sample_rate,
        max_points=args.max_points,
        active_point_sample_rate=args.active_point_sample_rate,
        max_active_points=args.max_active_points,
        random_seed=args.random_seed,
        line_arrowhead_ratio=args.line_arrowhead_ratio,
    )

    print(f"Offset file    : {offset_file}")
    print(f"Point cloud    : {pointcloud_file}")
    print(f"Offset shape   : {offset_data.shape}")
    print(f"Point xyz shape: {coords.shape}")
    print(
        "Render stats   : "
        f"visible_points={render_stats['visible_points']}/{render_stats['total_points']}, "
        f"drawn_arrows={render_stats['drawn_arrows']}, "
        f"arrow_style={render_stats['arrow_style']}"
    )
    print("-" * 72)
    for kp_idx, valid_count, keypoint_coord, consistency in summaries:
        if valid_count == 0:
            print(f"KP {kp_idx}: no valid mask=1 points")
            continue
        print(
            f"KP {kp_idx}: valid_points={valid_count}, "
            f"inferred_coord={np.round(keypoint_coord, 4)}, "
            f"max_target_deviation={consistency:.6f}"
        )

    visualize(
        geometries=geometries,
        point_size=point_size,
        line_width=args.line_width,
        window_name=os.path.basename(offset_file),
    )

"""
python tools/visualize_keypoint_offset_npy.py \
  --offset-file  KeyPointDataset_Split/train/keypoints/20260329_105410_942_keypoint_offset.npy\
  --keypoint-index 0 \
  --max-points 8000 \
  --max-arrows-per-kp 50 \
  --random-seed 42
    
"""
if __name__ == "__main__":
    main()
