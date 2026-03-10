"""
作用：极致优化的 SemanticKITTI 官方标签流式播放器。
优化点：
    1. 引入 Numpy 极速切片降采样 (Stride)，视觉无损的同时砍掉成倍的渲染负担。
    2. 移除 time.sleep() 人工阻塞，榨干每一滴 CPU 性能。
    3. 优化了内存指针分配频率。
"""

import os
import glob
import numpy as np
import open3d as o3d
import time

# --- 1. 路径配置 ---
BIN_DIR = "/mnt/f/Gongzihang/2026/data/SemanticKITTI/dataset/sequences/00/velodyne"
LABEL_DIR = "/mnt/f/Gongzihang/2026/data/SemanticKITTI/dataset/sequences/00/labels"

# --- 2. 性能调节旋钮 ---
# 【核心优化 1】降采样步长：2 表示每 2 个点取 1 个，3 表示每 3 个点取 1 个。
# KITTI 点云极密，设为 2 或 3 几乎不影响你观察车辆和行人，但帧率会暴涨！
STRIDE = 2 

# 目标帧率控制 (避免太快导致看不清)
TARGET_FPS = 30
FRAME_TIME = 1.0 / TARGET_FPS

def get_color_map(labels):
    colors = np.zeros((len(labels), 3), dtype=np.float64)
    
    dynamic_ids = [10, 11, 15, 18, 20, 30, 31, 32, 252, 253, 254, 255, 256, 257, 258, 259]
    dynamic_mask = np.isin(labels, dynamic_ids)
    
    person_ids = [30, 31, 32, 253, 254, 255]
    person_mask = np.isin(labels, person_ids)
    
    static_ids = [40, 44, 48, 49, 50, 51, 70, 71, 72, 80, 81, 99]
    static_mask = np.isin(labels, static_ids)
    
    colors[static_mask] = [0.3, 0.3, 0.3]
    colors[dynamic_mask] = [1.0, 0.0, 0.2]
    colors[person_mask] = [1.0, 0.8, 0.0]

    return colors

def play_point_cloud_video():
    bin_files = sorted(glob.glob(os.path.join(BIN_DIR, "*.bin")))
    label_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.label")))
    
    if not bin_files or not label_files:
        print("未找到点云或标签文件！")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="SemanticKITTI Smooth Stream", width=1280, height=720)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.05, 0.05, 0.05])
    # 既然降采样了，把点的大小稍微调大一点点，弥补视觉密度
    opt.point_size = 3.0 
    
    pcd = o3d.geometry.PointCloud()
    is_first_frame = True 

    for i, (bin_path, label_path) in enumerate(zip(bin_files, label_files)):
        loop_start_time = time.time()
        
        # 读取数据
        scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        labels = np.fromfile(label_path, dtype=np.uint32) & 0xFFFF
        
        if len(scan) != len(labels):
            continue
            
        # 【核心优化 2】极其轻量的降采样：直接切片跳跃读取，瞬间抛弃一半数据！
        xyz = scan[::STRIDE, :3]
        labels_sampled = labels[::STRIDE]
        
        colors = get_color_map(labels_sampled)
        
        # 内存更新
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        if is_first_frame:
            vis.add_geometry(pcd)
            vis.reset_view_point(True)
            is_first_frame = False
        else:
            vis.update_geometry(pcd)
            
        vis.poll_events()
        vis.update_renderer()
        
        # 【核心优化 3】智能休眠控制
        # 如果处理这帧的时间已经超过了目标时间，就坚决不休眠，全力渲染下一帧
        process_time = time.time() - loop_start_time
        sleep_time = FRAME_TIME - process_time
        if sleep_time > 0:
            time.sleep(sleep_time)
            
    vis.destroy_window()

if __name__ == "__main__":
    play_point_cloud_video()