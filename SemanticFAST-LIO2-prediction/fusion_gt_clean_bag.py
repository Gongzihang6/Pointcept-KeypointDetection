"""
作用：利用 SemanticKITTI 官方真值标签 (Ground Truth)，构建完美剔除动态物体的 Oracle ROSBag。
功能：交织 100Hz 原生 IMU 与 10Hz Lidar，并在打包瞬间根据官方 .label 文件抹除所有车辆与行人。
实现了什么：生成一个没有任何动态干扰的绝对纯净版 05 序列，用于探究 FAST-LIO2 在该场景下的理论最高精度。
怎么实现的：
    1. 使用 np.fromfile 读取 uint32 格式的 .label 文件。
    2. 使用位运算 `& 0xFFFF` 剥离实例 ID，提取纯语义 ID。
    3. 利用 NumPy 的 isin 算子，精准匹配并剔除官方定义的动态类别（包含静止车辆、行人和正在移动的物体）。
"""

import os
import numpy as np
import rosbag
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField, Imu
from datetime import datetime

def parse_time(ts_str):
    """解析时间戳"""
    ts_str = ts_str.strip()
    main_part, frac_part = ts_str.split('.')
    frac_part = frac_part[:6].ljust(6, '0')
    dt = datetime.strptime(f"{main_part}.{frac_part}", "%Y-%m-%d %H:%M:%S.%f")
    return dt.timestamp()

def create_cloud_msg(header, points):
    """构建自带 time 和 ring 的 PointCloud2 消息"""
    dt = np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('intensity', np.float32), ('ring', np.uint16), ('time', np.float32)
    ])
    structured_points = np.zeros(points.shape[0], dtype=dt)
    structured_points['x'] = points[:, 0]
    structured_points['y'] = points[:, 1]
    structured_points['z'] = points[:, 2]
    structured_points['intensity'] = points[:, 3]
    structured_points['ring'] = 0     
    structured_points['time'] = 0.0   
    
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = points.shape[0]
    msg.is_dense = False
    msg.is_bigendian = False
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1), PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1), PointField('intensity', 12, PointField.FLOAT32, 1),
        PointField('ring', 16, PointField.UINT16, 1), PointField('time', 18, PointField.FLOAT32, 1)
    ]
    msg.point_step = structured_points.itemsize
    msg.row_step = msg.point_step * msg.width
    msg.data = structured_points.tobytes()
    return msg

# ================= 路径配置 (已为你适配真实路径) =================
# 1. 原始驱动数据 (获取 100Hz 组合导航 IMU)
RAW_ROOT = '/mnt/f/Gongzihang/2026/data/2011_09_30/2011_09_30_drive_0018_extract'
# 2. 官方 Odometry 05 点云 (.bin)
BIN_DIR = '/mnt/f/Gongzihang/2026/data/SemanticKITTI/dataset/sequences/05/velodyne'
# 3. 官方真值标签 (.label)
LABEL_DIR = '/mnt/f/Gongzihang/2026/data/SemanticKITTI/dataset/sequences/05/labels'

# 终极纯净包的输出名称
OUTPUT_BAG = 'kitti_05_Oracle_clean.bag'

def build_bag():
    print("开始融合打包 [官方真值 Oracle 纯净版]...")
    events = []

    # --- 1. 读取 Lidar 任务 ---
    lidar_ts_file = os.path.join(RAW_ROOT, 'velodyne_points', 'timestamps.txt')
    with open(lidar_ts_file, 'r') as f:
        for i, line in enumerate(f):
            ts = parse_time(line)
            bin_path = os.path.join(BIN_DIR, f"{i:06d}.bin")
            label_path = os.path.join(LABEL_DIR, f"{i:06d}.label")
            if os.path.exists(bin_path) and os.path.exists(label_path):
                events.append(('lidar', ts, bin_path, label_path))

    # --- 2. 读取 IMU 任务 ---
    imu_ts_file = os.path.join(RAW_ROOT, 'oxts', 'timestamps.txt')
    imu_data_dir = os.path.join(RAW_ROOT, 'oxts', 'data')
    with open(imu_ts_file, 'r') as f:
        for i, line in enumerate(f):
            ts = parse_time(line)
            txt_path = os.path.join(imu_data_dir, f"{i:010d}.txt")
            if os.path.exists(txt_path):
                events.append(('imu', ts, txt_path, None))

    events.sort(key=lambda x: x[1])
    
    # 官方定义的动态干扰物 ID 集合
    # 包含：静止的车辆/行人 (10-32) + 明确处于移动状态的物体 (252-259)
    # SLAM 中通常为了保证环境特征绝对静态，会将它们全部挖去
    oracle_dynamic_ids = [10, 11, 15, 18, 20, 30, 31, 32, 
                          252, 253, 254, 255, 256, 257, 258, 259]

    with rosbag.Bag(OUTPUT_BAG, 'w') as bag:
        for idx, event in enumerate(events):
            topic_type, ts, file_path1, file_path2 = event
            ros_time = rospy.Time.from_sec(ts)
            header = Header(stamp=ros_time, frame_id="camera_init")

            if topic_type == 'lidar':
                # 读原点云
                scan = np.fromfile(file_path1, dtype=np.float32).reshape(-1, 4)
                
                # 读取官方二进制标签，并位运算提取低 16 位的 Semantic ID
                labels = np.fromfile(file_path2, dtype=np.uint32) & 0xFFFF
                
                # 寻找属于动态 ID 集合的索引
                dynamic_mask = np.isin(labels, oracle_dynamic_ids)
                
                # 极速内存切片取反，只保留绝对静态点 (建筑物、道路、树木等)
                clean_scan = scan[~dynamic_mask] 
                
                msg = create_cloud_msg(header, clean_scan)
                bag.write('/kitti/velodyne', msg, ros_time)

            elif topic_type == 'imu':
                with open(file_path1, 'r') as f:
                    values = [float(x) for x in f.read().split()]
                imu_msg = Imu()
                imu_msg.header = header
                # 加速度
                imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z = values[14], values[15], values[16]
                # 角速度
                imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z = values[17], values[18], values[19]
                bag.write('/kitti/oxts/imu', imu_msg, ros_time)
                
            if idx % 2000 == 0 and idx > 0:
                print(f"已处理 {idx}/{len(events)} 条数据...")

    print(f"天花板包打包完毕！已生成极其纯净的 {OUTPUT_BAG}")

if __name__ == '__main__':
    build_bag()
