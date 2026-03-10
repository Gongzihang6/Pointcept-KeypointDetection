"""
作用：流式可视化 SemanticKITTI 点云与预测标签，高亮显示动态物体。
功能：像视频播放器一样，按时间顺序逐帧加载 .bin 点云和 .npy 标签，并实时渲染到 3D 窗口中。
实现了什么：
    1. 将静态环境（道路、建筑物、树木）渲染为半透明的暗灰色。
    2. 将所有潜在的动态物体（汽车、卡车、行人、自行车等）渲染为极其醒目的高亮红色/黄色。
怎么实现的：
    1. 借助 Open3D 的非阻塞渲染管线 (Non-blocking visualization)，避免每次加载新帧都重新弹窗。
    2. 定义 SemanticKITTI 的标签映射字典，通过 NumPy 向量化操作，以极速的 C 底层运算为每个点附上 RGB 颜色。
    3. 动态更新 PointCloud 对象的 points 和 colors 属性，刷新渲染器。
"""

import os
import glob
import numpy as np
import open3d as o3d
import time

# --- 1. 路径配置（请修改为你实际的路径） ---
# KITTI 05 序列原始点云路径 (.bin)
BIN_DIR = "/mnt/f/Gongzihang/2026/data/SemanticKITTI/dataset/sequences/05/velodyne"
# 刚才 Pointcept 推理生成的预测标签路径 (.npy)
PRED_DIR = "/home/gzh/slam/Pointcept/exp/semantickitti/semseg-pt-v3m1-0-base/results/05"

def get_color_map(labels):
    """
    根据 Pointcept 预测的 ID (0-19) 给点云上色。
    动态物体（车辆、行人等）高亮，静态环境变暗。
    """
    colors = np.zeros((len(labels), 3), dtype=np.float64)
    
    # SemanticKITTI Pointcept 映射后的 ID 规则 (0: 忽略/未标记)
    # 动态类 ID: 1(Car), 2(Bicycle), 3(Motorcycle), 4(Truck), 5(Other-vehicle), 6(Person), 7(Bicyclist), 8(Motorcyclist)
    dynamic_mask = (labels >= 1) & (labels <= 8)
    
    # 静态类 ID: 9(Road), 10(Parking), 11(Sidewalk), 12(Other-ground), 13(Building), 14(Fence), 15(Vegetation), 16(Trunk), 17(Terrain), 18(Pole), 19(Traffic-sign)
    static_mask = labels >= 9
    
    # --- 视觉风格配置 ---
    # 静态环境：深灰色 (不喧宾夺主)
    colors[static_mask] = [0.3, 0.3, 0.3]
    
    # 动态物体：极其醒目的霓虹红色 (警示色)
    colors[dynamic_mask] = [1.0, 0.0, 0.2]
    
    # 细分（可选）：把行人单独标成亮黄色
    person_mask = (labels == 6) | (labels == 7) | (labels == 8)
    colors[person_mask] = [1.0, 0.8, 0.0]

    return colors

def play_point_cloud_video():
    bin_files = sorted(glob.glob(os.path.join(BIN_DIR, "*.bin")))
    pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "*.npy")))
    
    if not bin_files or not pred_files:
        print("未找到点云或标签文件，请检查路径！")
        return

    print(f"共找到 {len(bin_files)} 帧数据，准备开始流式播放...")

    # 初始化 Open3D 可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="SemanticKITTI Dynamic Stream", width=1280, height=720)
    
    # 设置黑色背景，让点云更具科技感
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.05, 0.05, 0.05])
    opt.point_size = 2.0
    
    # 创建一个空点云对象，并加入渲染器
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    # 循环遍历每一帧并更新画面
    for i, (bin_path, pred_path) in enumerate(zip(bin_files, pred_files)):
        # 1. 读取 Lidar 点云 (x, y, z, intensity)
        scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        xyz = scan[:, :3]
        
        # 2. 读取网络预测标签
        labels = np.load(pred_path)
        
        # 3. 规避点数不匹配的极端情况 (通常发生在数据损坏时)
        if len(xyz) != len(labels):
            print(f"帧 {i} 点数不匹配，跳过...")
            continue
            
        # 4. 获取色彩并更新点云对象
        colors = get_color_map(labels)
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # 5. 更新渲染器
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        
        # 控制播放帧率 (大约 20 FPS)
        time.sleep(0.05)
        
    vis.destroy_window()
    print("播放结束！")

if __name__ == "__main__":
    play_point_cloud_video()
