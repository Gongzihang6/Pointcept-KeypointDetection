import numpy as np
import argparse
import os

def inspect_npy(file_path, num_samples=5):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return

    print(f"{'='*20} Inspecting: {os.path.basename(file_path)} {'='*20}")
    
    try:
        data = np.load(file_path)
    except Exception as e:
        print(f"Error loading npy file: {e}")
        return

    # Basic Info
    print(f"Basic Information:")
    print(f"  - Shape: {data.shape}")
    print(f"  - DataType: {data.dtype}")
    print(f"  - Total Points: {len(data)}")
    print(f"  - Memory Usage: {data.nbytes / 1024 / 1024:.2f} MB")
    
    # Column Statistics (assuming N x D)
    if data.ndim == 2:
        print(f"\nColumn-wise Statistics (Min, Max, Mean):")
        column_names = ["x", "y", "z", "nx", "ny", "nz", "curvature", "class"]
        for i in range(data.shape[1]):
            col_data = data[:, i]
            name = column_names[i] if i < len(column_names) else f"col_{i}"
            print(f"  - {name:10}: Min={col_data.min():.4f}, Max={col_data.max():.4f}, Mean={col_data.mean():.4f}")

        # Class distribution (assuming last column is class)
        if data.shape[1] >= 8:
            labels = data[:, 7]
            unique, counts = np.unique(labels, return_counts=True)
            print(f"\nClass Distribution:")
            for val, count in zip(unique, counts):
                label_name = "Pig (1)" if val == 1 else ("Background (0)" if val == 0 else f"Other ({val})")
                print(f"  - {label_name:15}: {count} points ({count/len(labels)*100:.2f}%)")
    
    # Example Data
    print(f"\nExample Data (First {num_samples} rows):")
    np.set_printoptions(precision=4, suppress=True)
    print(data[:num_samples])
    
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="Inspect a .npy file for basic info and sample data.")
    parser.add_argument(
        "file_path", 
        type=str, 
        nargs='?',
        default="Processed_Pig_Dataset/train/labels/20260329_105410_942.npy",
        help="Path to the .npy file."
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=50, 
        help="Number of sample rows to show."
    )
    args = parser.parse_args()

    inspect_npy(args.file_path, args.samples)

if __name__ == "__main__":
    main()
