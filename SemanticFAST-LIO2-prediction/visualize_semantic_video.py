"""
作用：使用流媒体神器 Rerun (适配最新版 API) 替代 Open3D，流式可视化点云与预测标签。
功能：在本地启动独立的数据流服务器和 Web 服务器，按时间顺序加载数据并推送到浏览器。
实现了什么：
    1. 彻底跳出了 Open3D 传统渲染管线的深坑，完美绕开 WSL2 显卡驱动不兼容导致的黑屏问题。
    2. 解决了最新版 Rerun 中 rr.serve() 被弃用导致的 AttributeError 报错。
怎么实现的：
    1. 引入 Rerun SDK，首先调用 rr.serve_grpc() 启动底层数据通信引擎。
    2. 调用 rr.serve_web_viewer() 启动前端网页，并将其连接到 gRPC 数据源。
    3. 循环内部使用 rr.log() 推送点云坐标和颜色。
"""

import os
import glob
import numpy as np
import rerun as rr
import time

# --- 1. 路径配置 ---
BIN_DIR = "/home/gzh/point/Pointcept-KeypointDetection/SemanticFAST-LIO2-prediction/SemanticKITTI_Mini/dataset/sequences/05/velodyne"
PRED_DIR = "/home/gzh/point/Pointcept-KeypointDetection/exp/semantickitti/semseg-pt-v2m2-0-base/results/05"

def get_color_map(labels):
    """
    根据标签生成 RGB 颜色。
    注意：Rerun 推荐使用 0-255 的 uint8 格式以获得最佳渲染性能。
    """
    colors = np.zeros((len(labels), 3), dtype=np.uint8)
    
    dynamic_mask = (labels >= 1) & (labels <= 8)
    static_mask = labels >= 9
    person_mask = (labels == 6) | (labels == 7) | (labels == 8)
    
    # 静态环境：深灰色
    colors[static_mask] = [76, 76, 76]
    # 动态物体：极其醒目的绿色
    colors[dynamic_mask] = [0, 255, 0] 
    # 行人细分：亮黄色
    colors[person_mask] = [255, 204, 0]

    return colors

def play_point_cloud_video():
    bin_files = sorted(glob.glob(os.path.join(BIN_DIR, "*.bin")))
    pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "*.npy")))
    
    if not bin_files or not pred_files:
        print("未找到点云或标签文件，请检查路径！")
        return

    print(f"共找到 {len(bin_files)} 帧数据。")
    
    # --- 启动 Rerun Web 流媒体服务器 ---
    print("\n" + "="*55)
    print("👉 正在启动 Rerun 推流服务器...")
    print("💡 请在浏览器中打开: http://localhost:9090")
    print("="*55 + "\n")
    
    rr.init("SemanticKITTI_Stream")
    
    # --- 【关键修复：适配最新版 Rerun API】 ---
    # 1. 启动底层 gRPC 数据传输服务器，并获取自动分配的地址
    server_uri = rr.serve_grpc() 
    
    # 2. 启动轻量级的 Web 前端播放器，并让它连接到刚才的 gRPC 地址
    rr.serve_web_viewer(connect_to=server_uri) 
    
    # 给服务器一点点启动时间
    time.sleep(2)

    for i, (bin_path, pred_path) in enumerate(zip(bin_files, pred_files)):
        scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        xyz = scan[:, :3]
        labels = np.load(pred_path)
        
        if len(xyz) != len(labels):
            continue
            
        colors = get_color_map(labels)
        
        # 记录时间步，网页端会自动生成一条播放进度条
        # rr.set_time_sequence("frame_idx", i)
        
        # 将点云和颜色一次性打包推送到前端
        rr.log("lidar_stream", rr.Points3D(positions=xyz, colors=colors))
        
        # 控制推送节奏，让网络传输更平滑
        time.sleep(0.05)
        
    print("全部数据推送结束！")
    print("现在您可以直接在浏览器网页的下方，自由拖动时间轴进度条进行回放分析了。")

if __name__ == "__main__":
    play_point_cloud_video()