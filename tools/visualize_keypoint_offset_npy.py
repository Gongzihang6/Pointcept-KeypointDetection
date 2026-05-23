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
        default=0.2,
        help="Cone length ratio in each arrow.",
    )
    parser.add_argument(
        "--max-arrows-per-kp",
        type=int,
        default=200,
        help="Maximum number of mask=1 arrows to draw for each keypoint.",
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
    return data[:, :3].astype(np.float64)


def load_offset(offset_file):
    if not os.path.exists(offset_file):
        raise FileNotFoundError(f"Offset file not found: {offset_file}")

    data = np.load(offset_file)
    if data.ndim != 3 or data.shape[2] != 4:
        raise ValueError(
            f"Offset label should have shape (N, K, 4), but got {data.shape} from {offset_file}"
        )
    return data.astype(np.float64)


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
):
    offsets = offset_data[..., :3]
    masks = offset_data[..., 3] > 0.5
    num_points, num_keypoints, _ = offset_data.shape
    bbox_scale = get_bbox_scale(coords)

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
    point_colors = np.zeros((num_points, 3), dtype=np.float64)
    active_union = np.zeros(num_points, dtype=bool)
    keypoint_summaries = []

    for kp_idx in selected_kps:
        valid_idx = np.where(masks[:, kp_idx])[0]
        color = DEFAULT_COLORS[kp_idx % len(DEFAULT_COLORS)]

        if len(valid_idx) == 0:
            keypoint_summaries.append((kp_idx, 0, None, None))
            continue

        active_union[valid_idx] = True

        candidate_targets = coords[valid_idx] + offsets[valid_idx, kp_idx]
        keypoint_coord = candidate_targets.mean(axis=0)
        consistency = np.linalg.norm(candidate_targets - keypoint_coord, axis=1).max()
        keypoint_summaries.append((kp_idx, len(valid_idx), keypoint_coord, consistency))

        sampled_idx = valid_idx
        if len(valid_idx) > max_arrows_per_kp:
            sampled_idx = np.linspace(0, len(valid_idx) - 1, max_arrows_per_kp, dtype=int)
            sampled_idx = valid_idx[sampled_idx]

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

    visible_points = coords[active_union] if hide_background else coords
    visible_colors = point_colors[active_union] if hide_background else point_colors

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(visible_points)
    pcd.colors = o3d.utility.Vector3dVector(visible_colors)
    geometries.insert(0, pcd)

    return geometries, keypoint_summaries, point_size


def visualize(geometries, point_size, window_name):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1400, height=900)
    for geom in geometries:
        vis.add_geometry(geom)

    render_option = vis.get_render_option()
    render_option.point_size = point_size
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

    geometries, summaries, point_size = build_visual_geometries(
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
    )

    print(f"Offset file    : {offset_file}")
    print(f"Point cloud    : {pointcloud_file}")
    print(f"Offset shape   : {offset_data.shape}")
    print(f"Point xyz shape: {coords.shape}")
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
        window_name=os.path.basename(offset_file),
    )

"""
uv run tools/visualize_keypoint_offset_npy.py \
    --offset-file KeyPointDataset_Split/train/keypoints/20260329_105410_942_keypoint_offset.npy \
    --point-size 2.0 \
    --keypoint-index 0 \
    
"""
if __name__ == "__main__":
    main()
