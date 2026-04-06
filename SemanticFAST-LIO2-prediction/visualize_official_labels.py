"""
作用：极致优化的 SemanticKITTI 双端同步流式播放器（本地 GUI + 网页端）。
功能：在保留原有 Open3D 本地非阻塞渲染的基础上，额外增加了 Rerun Web 前端推流功能。
实现了什么：
    1. 保留了原有的 Numpy 极速切片降采样 (Stride)、指针更新和智能休眠等所有硬核性能优化。
    2. 实现了数据“一源多端”同步：同一份点云和标签数据，即时在 Open3D 窗口渲染，同时也通过 WebRTC 协议推送到浏览器页面。
怎么实现的：
    1. 引入 rerun 库，在 Open3D 初始化前后，利用 rr.serve_grpc() 和 rr.serve_web_viewer() 开启独立的数据通信后端与网页前端。
    2. 在主循环中保留 Open3D 的 update_renderer() 逻辑。
    3. 增加 rr.log() 推送点云，并通过 (colors * 255).astype(np.uint8) 极速计算，将 Open3D 依赖的浮点色值动态转译为 Rerun 性能最佳的整型色值。
"""

import os
import glob
import numpy as np
import open3d as o3d
import time
import rerun as rr

# --- 1. 路径配置 ---
BIN_DIR = "/mnt/f/Gongzihang/2026/data/SemanticKITTI/dataset/sequences/05/velodyne"
LABEL_DIR = "/mnt/f/Gongzihang/2026/data/SemanticKITTI/dataset/sequences/05/labels"

# --- 2. 性能调节旋钮 ---
STRIDE = 2 
TARGET_FPS = 30
FRAME_TIME = 1.0 / TARGET_FPS

def get_color_map(labels):
    # 保持原样：Open3D 需要 0.0-1.0 的 float64 颜色
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

    print(f"共找到 {len(bin_files)} 帧数据，准备启动双端渲染...")

    # ==========================================
    # 【新增】1. 启动 Rerun 网页端服务器
    # ==========================================
    rr.init("SemanticKITTI_Dual_Stream")
    server_uri = rr.serve_grpc() 
    rr.serve_web_viewer(connect_to=server_uri) 
    
    print("\n👉 网页端已启动！请在浏览器打开: http://localhost:9090")
    print("👉 本地 Open3D 窗口即将弹出...\n")
    time.sleep(1) # 稍微给 Web 服务器一点启动时间

    # ==========================================
    # 保留：初始化 Open3D 本地端
    # ==========================================
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name="SemanticKITTI Smooth Stream (Local)", width=1280, height=720)
    
    # opt = vis.get_render_option()
    # opt.background_color = np.asarray([0.05, 0.05, 0.05])
    # opt.point_size = 1.0 
    
    pcd = o3d.geometry.PointCloud()
    is_first_frame = True 

    for i, (bin_path, label_path) in enumerate(zip(bin_files, label_files)):
        loop_start_time = time.time()
        
        # 读取数据
        scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        labels = np.fromfile(label_path, dtype=np.uint32) & 0xFFFF
        
        if len(scan) != len(labels):
            continue
            
        # 极速切片降采样
        xyz = scan[::STRIDE, :3]
        labels_sampled = labels[::STRIDE]
        
        # 获取 Open3D 适用的 float64 色彩
        colors = get_color_map(labels_sampled)
        
        # ==========================================
        # 保留：刷新 Open3D 本地端
        # ==========================================
        # pcd.points = o3d.utility.Vector3dVector(xyz)
        # pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # if is_first_frame:
        #     vis.add_geometry(pcd)
        #     vis.reset_view_point(True)
        #     is_first_frame = False
        # else:
        #     vis.update_geometry(pcd)
            
        # vis.poll_events()
        # vis.update_renderer()

        # ==========================================
        # 【新增】2. 同步推送至 Rerun 网页端
        # ==========================================
        # rr.set_time_sequence("frame_idx", i)
        # Rerun 底层由 Rust 编写，对 uint8 格式的颜色渲染效率极高
        colors_uint8 = (colors * 255).astype(np.uint8)
        rr.log("lidar_stream", rr.Points3D(positions=xyz, colors=colors_uint8))
        
        # 智能休眠控制
        process_time = time.time() - loop_start_time
        sleep_time = FRAME_TIME - process_time
        if sleep_time > 0:
            time.sleep(sleep_time)
            
    vis.destroy_window()

if __name__ == "__main__":
    play_point_cloud_video()