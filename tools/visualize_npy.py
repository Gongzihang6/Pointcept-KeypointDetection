import argparse
import glob
import os

import numpy as np
import open3d as o3d

LABEL_COLUMN = 7
MIN_COLUMNS = 8
BACKGROUND_COLOR = np.asarray([0.0, 0.0, 0.0])
BACKGROUND_POINT_COLOR = np.asarray([0.0, 0.0, 1.0])
FOREGROUND_POINT_COLOR = np.asarray([1.0, 0.0, 0.0])


def create_point_cloud(coords, colors, normals=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if normals is not None and len(normals) == len(coords):
        pcd.normals = o3d.utility.Vector3dVector(normals)

    return pcd


def create_coordinate_frame(coords, axis_size=None):
    min_bound = coords.min(axis=0)
    max_bound = coords.max(axis=0)
    extent = np.linalg.norm(max_bound - min_bound)

    if axis_size is None:
        axis_size = max(extent * 0.15, 0.1)

    return o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=axis_size,
        origin=np.zeros(3),
    )


def visualize_pcd(
    coords,
    colors,
    normals=None,
    window_name="Open3D",
    point_size=2.0,
    show_axis=True,
    axis_size=None,
):
    if len(coords) == 0:
        print(f"Skip visualization for {window_name}: empty point cloud.")
        return

    pcd = create_point_cloud(coords, colors, normals)
    print(f"Visualizing {len(coords)} points.")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)

    if show_axis:
        vis.add_geometry(create_coordinate_frame(coords, axis_size))

    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = BACKGROUND_COLOR

    vis.run()
    vis.destroy_window()


def load_data(file_path):
    print(f"Loading {file_path}...")
    try:
        data = np.load(file_path)
        if data.ndim != 2 or data.shape[1] < MIN_COLUMNS:
            print(
                f"Warning: Data shape {data.shape} in {file_path} does not meet "
                f"the minimum requirement of (N, {MIN_COLUMNS})."
            )
            return None
        return data
    except Exception as exc:
        print(f"Error loading {file_path}: {exc}")
        return None


def process_data(data):
    coords = data[:, :3]
    labels = data[:, LABEL_COLUMN]

    colors = np.zeros((len(labels), 3), dtype=np.float32)
    colors[:] = BACKGROUND_POINT_COLOR
    colors[labels == 1] = FOREGROUND_POINT_COLOR

    normals = data[:, 3:6] if data.shape[1] >= 6 else None
    return coords, colors, normals


def collect_npy_files(input_path):
    files = sorted(glob.glob(os.path.join(input_path, "*.npy")))
    if not files:
        print(f"No .npy files found in {input_path}")
        return []
    return files


def load_processed_data(file_path):
    data = load_data(file_path)
    if data is None:
        return None
    return process_data(data)


def visualize_single_file(file_path, point_size, show_axis, axis_size):
    processed = load_processed_data(file_path)
    if processed is None:
        return

    coords, colors, normals = processed
    visualize_pcd(
        coords,
        colors,
        normals,
        window_name=os.path.basename(file_path),
        point_size=point_size,
        show_axis=show_axis,
        axis_size=axis_size,
    )


def visualize_combined(files, point_size, show_axis, axis_size):
    all_coords = []
    all_colors = []
    all_normals = []
    has_normals = True

    for file_path in files:
        processed = load_processed_data(file_path)
        if processed is None:
            continue

        coords, colors, normals = processed
        all_coords.append(coords)
        all_colors.append(colors)
        if normals is None:
            has_normals = False
        else:
            all_normals.append(normals)

    if not all_coords:
        print("No valid point cloud data found.")
        return

    visualize_pcd(
        np.concatenate(all_coords, axis=0),
        np.concatenate(all_colors, axis=0),
        np.concatenate(all_normals, axis=0) if has_normals and all_normals else None,
        window_name="Combined Visualization",
        point_size=point_size,
        show_axis=show_axis,
        axis_size=axis_size,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Visualize .npy point cloud files.")
    parser.add_argument(
        "--input_path",
        type=str,
        default="/home/gzh/point/Pointcept-KeypointDetection/body_npy_output/train",
        help="Path to the .npy file or directory containing .npy files.",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=1.0,
        help="Size of the points in visualization.",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine all files in the directory into one visualization.",
    )
    parser.add_argument(
        "--hide_axis",
        action="store_true",
        help="Hide the coordinate frame in visualization.",
    )
    parser.add_argument(
        "--axis_size",
        type=float,
        default=None,
        help="Coordinate frame size. Defaults to an automatic size based on point cloud extent.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if os.path.isdir(args.input_path):
        files = collect_npy_files(args.input_path)
        if not files:
            return

        print(f"Found {len(files)} files in {args.input_path}")
        if args.combine:
            visualize_combined(
                files,
                point_size=args.point_size,
                show_axis=not args.hide_axis,
                axis_size=args.axis_size,
            )
        else:
            for file_path in files:
                visualize_single_file(
                    file_path,
                    point_size=args.point_size,
                    show_axis=not args.hide_axis,
                    axis_size=args.axis_size,
                )
                print("Press 'q' or close window to see the next file, or Ctrl+C to exit.")
    elif os.path.isfile(args.input_path):
        visualize_single_file(
            args.input_path,
            point_size=args.point_size,
            show_axis=not args.hide_axis,
            axis_size=args.axis_size,
        )
    else:
        print(f"Invalid path: {args.input_path}")


"""
python tools/visualize_npy.py
"""


if __name__ == "__main__":
    main()
