import numpy as np
import open3d as o3d
# 1. 强制按 float32 和 4 列的规则解析二进制流
scan = np.fromfile("SemanticFAST-LIO2-prediction/SemanticKITTI_Mini/dataset/sequences/00/velodyne/000000.bin", dtype=np.float32).reshape(-1, 4)
# 2. 提取 XYZ 并塞给 Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(scan[:, :3])
# 3. 另存为带文件头的标准格式
o3d.io.write_point_cloud("000000.pcd", pcd)
