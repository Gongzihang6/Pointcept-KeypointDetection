import os
import numpy as np
import open3d as o3d
import argparse
import glob

def visualize_pcd(coords, colors, normals=None, window_name="Open3D", point_size=2.0):
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)

    print(f"Visualizing {len(coords)} points.")
    
    # Create Visualizer to set point size
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.asarray([0, 0, 0]) # Black background
    
    vis.run()
    vis.destroy_window()

def load_data(file_path):
    print(f"Loading {file_path}...")
    try:
        data = np.load(file_path)
        if data.shape[1] < 8:
            print(f"Warning: Data shape {data.shape} in {file_path} is not (N, 8).")
            return None
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def process_data(data):
    # Extract coordinates (x, y, z)
    coords = data[:, :3]
    # Extract class labels (last column)
    labels = data[:, 7]

    # Initialize colors: [0, 0, 1] for Blue (background, class 0)
    colors = np.zeros((len(labels), 3))
    colors[:, 2] = 1.0  # Blue

    # Set Red for class 1 (pig)
    pig_mask = labels == 1
    colors[pig_mask] = [1.0, 0.0, 0.0]  # Red
    
    normals = data[:, 3:6] if data.shape[1] >= 6 else None
    return coords, colors, normals

def main():
    parser = argparse.ArgumentParser(description="Visualize .npy point cloud files.")
    parser.add_argument(
        "--input_path", 
        type=str, 
        default="/home/gzh/point/Pointcept-KeypointDetection/body_npy_output/train",
        help="Path to the .npy file or directory containing .npy files."
    )
    parser.add_argument(
        "--point_size", 
        type=float, 
        default=1.0, 
        help="Size of the points in visualization."
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine all files in the directory into one visualization."
    )
    args = parser.parse_args()

    if os.path.isdir(args.input_path):
        files = sorted(glob.glob(os.path.join(args.input_path, "*.npy")))
        if not files:
            print(f"No .npy files found in {args.input_path}")
            return
        
        print(f"Found {len(files)} files in {args.input_path}")
        
        if args.combine:
            all_coords = []
            all_colors = []
            all_normals = []
            for f in files:
                data = load_data(f)
                if data is not None:
                    coords, colors, normals = process_data(data)
                    all_coords.append(coords)
                    all_colors.append(colors)
                    if normals is not None:
                        all_normals.append(normals)
            
            if all_coords:
                visualize_pcd(
                    np.concatenate(all_coords), 
                    np.concatenate(all_colors), 
                    np.concatenate(all_normals) if all_normals else None,
                    window_name="Combined Visualization",
                    point_size=args.point_size
                )
        else:
            for f in files:
                data = load_data(f)
                if data is not None:
                    coords, colors, normals = process_data(data)
                    visualize_pcd(coords, colors, normals, window_name=os.path.basename(f), point_size=args.point_size)
                    print("Press 'q' or close window to see the next file, or Ctrl+C to exit.")
    elif os.path.isfile(args.input_path):
        data = load_data(args.input_path)
        if data is not None:
            coords, colors, normals = process_data(data)
            visualize_pcd(coords, colors, normals, window_name=os.path.basename(args.input_path), point_size=args.point_size)
    else:
        print(f"Invalid path: {args.input_path}")
"""
python tools/visualize_npy.py 
"""
if __name__ == "__main__":
    main()
