"""
该脚本的作用是将点云数据和对应的关键点坐标转化为带有距离掩码（mask）的偏移量（offset）标签文件。

功能说明：
1. 遍历指定数据集目录（KeyPointDataset_Split）下的 train, val, test 子目录。
2. 读取对应的点云文件（由 x,y,z, nx,ny,nz, curvature 组成）和关键点坐标文件。
3. 计算每个点云点到6个关键点的偏移量向量以及二值掩码（mask）。
4. 将生成的标签文件（包含偏移量和掩码，形状为 N x 6 x 4）保存到原始关键点所在的相同目录下，按照指定的命名规则重命名。

具体实现机制（核心算法逻辑）：
- 初始化一个距离阈值 r（超参数，默认为 0.5）。
- 从形状为 (N, 7) 的点云数据中提取前三列作为 xyz 坐标，形状为 (N, 3)。
- 读取形状为 (6, 3) 的关键点数据。
- 利用 NumPy 的广播机制，将点云坐标变形为 (N, 1, 3)，将关键点坐标变形为 (1, 6, 3)，
  相减得到所有点到所有关键点的向量差 offset = keypoint_j - point_xyz（形状为 N x 6 x 3）。
- 计算每条偏移量向量的欧式距离（L2 范数），得到距离矩阵 distances（形状为 N x 6）。
- 根据距离阈值 r 生成二值掩码 mask = distances <= r（形状为 N x 6）。
- 将 mask = 0 的位置对应的偏移量设为 (0, 0, 0)。这可以通过把 offset 乘上 expanded 后的 mask (N, 6, 1) 来实现。
- 增加 mask 的维度（N x 6 x 1），并与处理后的 offset 拼接，最终生成形状为 (N, 6, 4) 的标签张量。
- 使用 tqdm 库展示数据处理的进度，同时捕获可能存在的文件缺失等异常，并打印警告以跳过缺失的文件。
"""

import os
import glob
import numpy as np
from tqdm import tqdm
import warnings

def generate_offset_labels(data_root="KeyPointDataset_Split", r=0.5):
    """
    基于关键点坐标与点云距离阈值 r 生成带有掩码的偏移量并保存为 npy。
    
    Args:
        data_root (str): 数据集的根目录。
        r (float): 判断为有效点（mask=1）的欧几里得距离阈值。
    """
    splits = ["train", "val", "test"]
    
    for split in splits:
        split_dir = os.path.join(data_root, split)
        if not os.path.exists(split_dir):
            warnings.warn(f"目录 '{split_dir}' 不存在，跳过。")
            continue
            
        pc_dir = os.path.join(split_dir, "pointclouds")
        kp_dir = os.path.join(split_dir, "keypoints")
        
        if not os.path.exists(pc_dir) or not os.path.exists(kp_dir):
            warnings.warn(f"在此路径下 '{split_dir}' 无法找全 pointclouds 和 keypoints 文件夹，跳过。")
            continue
            
        # 查找所有的点云文件
        pc_files = glob.glob(os.path.join(pc_dir, "*.npy"))
        
        if len(pc_files) == 0:
            print(f"[{split}] 目录中未找到任何 pointcloud .npy 文件。")
            continue
            
        print(f"==== 开始处理 {split} 集，总计找到 {len(pc_files)} 个点云文件 ====")
        
        for pc_file in tqdm(pc_files, desc=f"Processing {split}"):
            file_name = os.path.basename(pc_file)
            base_name = os.path.splitext(file_name)[0]
            
            # 找到对应的关键点文件（考虑到用户描述文件名可能含有 _关键点坐标 后缀）
            kp_file = os.path.join(kp_dir, f"{base_name}_关键点坐标.npy")
            
            if not os.path.exists(kp_file):
                # 尝试没有后缀名的同名匹配策略
                fallback_kp_file = os.path.join(kp_dir, f"{base_name}.npy")
                if os.path.exists(fallback_kp_file):
                    kp_file = fallback_kp_file
                else:
                    warnings.warn(f"未找到点云 '{file_name}' 对应的关键点标记文件，跳过。")
                    continue
            
            try:
                # [核心算法过程开始]
                # 1. 加载文件 
                pc_data = np.load(pc_file) # 点云: (N, 7+)
                kp_data = np.load(kp_file) # 关键点: (6, 3)
                
                # 提取前三维 xyz 坐标 (N, 3)
                pts_xyz = pc_data[:, :3]
                
                # 2. 计算距离和生成 offset (使用广播机制避免 for 循环)
                # pts_xyz.shape       => (N, 1, 3)
                # kp_data.shape       => (1, 6, 3)
                # diffuse_offset 结果 => (N, 6, 3), 计算方式: offset = keypoint_j - point_xyz 
                offsets = kp_data[np.newaxis, :, :] - pts_xyz[:, np.newaxis, :]
                
                # 计算 L2 欧几里得距离 (N, 6)
                distances = np.linalg.norm(offsets, axis=-1)
                
                # 判断掩码 mask: 距离 <= r, 类型为 float 的 0.0 或 1.0, 形状为 (N, 6)
                mask = (distances <= r).astype(np.float32)
                
                # 扩展 mask 维度进行矩阵相乘，以便将无效点的坐标强制设为 (0,0,0)
                # mask_expanded.shape => (N, 6, 1)
                mask_expanded = mask[..., np.newaxis]
                valid_offsets = offsets * mask_expanded
                
                # 3. 拼接输出张量 -> (N, 6, 4)
                # 将 (N, 6, 3) 与 (N, 6, 1) 在最后一个维度上拼接
                final_tensor = np.concatenate((valid_offsets, mask_expanded), axis=-1)
                # [核心算法过程结束]
                
                # [保存处理]
                # 获取原关键点文件名，并按要求替换后缀
                kp_filename_only = os.path.basename(kp_file)
                if "_关键点坐标" in kp_filename_only:
                    new_kp_filename = kp_filename_only.replace("_关键点坐标", "_keypoint_offset")
                else:
                    # 如果一开始没那个后缀，强行打上 _keypoint_offset 标识
                    name_root, ext = os.path.splitext(kp_filename_only)
                    new_kp_filename = f"{name_root}_keypoint_offset{ext}"
                    
                output_path = os.path.join(kp_dir, new_kp_filename)
                
                # 保存为 npy
                np.save(output_path, final_tensor)
                
            except Exception as e:
                warnings.warn(f"处理文件 '{file_name}' 的对应关键点时发生异常: {e}")

if __name__ == "__main__":
    generate_offset_labels(data_root="KeyPointDataset_Split", r=300.0)  # 这里的 r 值可以根据实际情况调整，单位通常是米或毫米，取决于点云数据的单位。
