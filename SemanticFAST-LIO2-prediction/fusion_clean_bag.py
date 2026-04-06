import os
import numpy as np
import rosbag
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField, Imu
from datetime import datetime

def parse_time(ts_str):
    ts_str = ts_str.strip()
    main_part, frac_part = ts_str.split('.')
    frac_part = frac_part[:6].ljust(6, '0')
    dt = datetime.strptime(f"{main_part}.{frac_part}", "%Y-%m-%d %H:%M:%S.%f")
    return dt.timestamp()

def create_cloud_msg(header, points):
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

# ================= 路径配置 (请仔细核对) =================
# 1. 原始驱动数据 (为了拿高频 IMU 和 绝对时间戳)
RAW_ROOT = '/mnt/f/Gongzihang/2026/data/2011_09_30/2011_09_30_drive_0018_extract'
# 2. 官方 Odometry 05 点云 (.bin)
BIN_DIR = '/mnt/f/Gongzihang/2026/data/SemanticKITTI/dataset/sequences/05/velodyne'
# 3. 你刚刚推理生成的预测结果 (.npy)
PRED_DIR = '/home/gzh/point/Pointcept-KeypointDetection/exp/semantic_kitti/swin3d_mini_run/results/05'
# 输出的干净包
OUTPUT_BAG = 'kitti_05_clean.bag'

def build_bag():
    print("开始融合打包 [点云 + IMU + 语义掩码]...")
    events = []

    # --- 读取 Lidar 任务 ---
    lidar_ts_file = os.path.join(RAW_ROOT, 'velodyne_points', 'timestamps.txt')
    with open(lidar_ts_file, 'r') as f:
        for i, line in enumerate(f):
            ts = parse_time(line)
            bin_path = os.path.join(BIN_DIR, f"{i:06d}.bin")
            npy_path = os.path.join(PRED_DIR, f"{i:06d}.npy")
            if os.path.exists(bin_path) and os.path.exists(npy_path):
                events.append(('lidar', ts, bin_path, npy_path))

    # --- 读取 IMU 任务 ---
    imu_ts_file = os.path.join(RAW_ROOT, 'oxts', 'timestamps.txt')
    imu_data_dir = os.path.join(RAW_ROOT, 'oxts', 'data')
    with open(imu_ts_file, 'r') as f:
        for i, line in enumerate(f):
            ts = parse_time(line)
            txt_path = os.path.join(imu_data_dir, f"{i:010d}.txt")
            if os.path.exists(txt_path):
                events.append(('imu', ts, txt_path, None))

    events.sort(key=lambda x: x[1])
    
    with rosbag.Bag(OUTPUT_BAG, 'w') as bag:
        for idx, event in enumerate(events):
            topic_type, ts, file_path1, file_path2 = event
            ros_time = rospy.Time.from_sec(ts)
            header = Header(stamp=ros_time, frame_id="camera_init")

            if topic_type == 'lidar':
                # 1. 读原点云
                scan = np.fromfile(file_path1, dtype=np.float32).reshape(-1, 4)
                # 2. 读你的模型预测标签
                labels = np.load(file_path2)
                
                # 3. 施展魔法：剔除动态物体 (0~7为动态)
                dynamic_mask = (labels >= 0) & (labels <= 7)
                clean_scan = scan[~dynamic_mask]  # 取反，只保留非动态点
                
                msg = create_cloud_msg(header, clean_scan)
                bag.write('/kitti/velodyne', msg, ros_time)

            elif topic_type == 'imu':
                with open(file_path1, 'r') as f:
                    values = [float(x) for x in f.read().split()]
                imu_msg = Imu()
                imu_msg.header = header
                imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z = values[14], values[15], values[16]
                imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z = values[17], values[18], values[19]
                bag.write('/kitti/oxts/imu', imu_msg, ros_time)
                
            if idx % 2000 == 0 and idx > 0:
                print(f"已处理 {idx}/{len(events)} 条数据...")

    print(f"动态物体剔除完毕！已生成极其纯净的 {OUTPUT_BAG}")

if __name__ == '__main__':
    build_bag()