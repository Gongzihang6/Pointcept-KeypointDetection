"""
作用：一键构建 Mini-SemanticKITTI 微型数据集。
功能：从原始的 SemanticKITTI 数据集中，抽取极少量的点云和标签样本，并伪造出完整的序列目录结构。
实现了什么：生成一个体积不到 200MB 的数据集，完美兼容 Pointcept 的 DataLoader，防止显存溢出并成百倍缩短训练时间。
怎么实现的：
    1. 遍历 00 到 10 的序列名称。
    2. 创建对应的 velodyne 和 labels 文件夹。
    3. 利用 shutil.copy 仅拷贝每个序列的前 30 帧 .bin 和 .label 文件。
"""

import os
import shutil

def create_mini_dataset(src_root, dst_root, frames_per_seq=30):
    print(f"开始构建 Mini 数据集，目标路径: {dst_root}")
    
    # SemanticKITTI 训练和验证需要的序列 00-10
    sequences = [f"{i:02d}" for i in range(11)]
    
    for seq in sequences:
        src_velo_dir = os.path.join(src_root, "dataset", "sequences", seq, "velodyne")
        src_label_dir = os.path.join(src_root, "dataset", "sequences", seq, "labels")
        
        dst_velo_dir = os.path.join(dst_root, "dataset", "sequences", seq, "velodyne")
        dst_label_dir = os.path.join(dst_root, "dataset", "sequences", seq, "labels")
        
        # 创建目标文件夹
        os.makedirs(dst_velo_dir, exist_ok=True)
        os.makedirs(dst_label_dir, exist_ok=True)
        
        # 如果原始路径不存在，跳过（防止报错）
        if not os.path.exists(src_velo_dir) or not os.path.exists(src_label_dir):
            print(f"警告: 原始序列 {seq} 不存在，已创建空文件夹。")
            continue
            
        # 获取所有 .bin 文件并排序
        bin_files = sorted(os.listdir(src_velo_dir))
        
        # 核心修改：均匀间隔采样
        total_frames = len(bin_files)
        if total_frames > frames_per_seq:
            # 计算步长，均匀抽取
            step = total_frames / frames_per_seq
            sampled_files = [bin_files[int(i * step)] for i in range(frames_per_seq)]
        else:
            sampled_files = bin_files

        # 遍历均匀采样出的文件列表
        for file_name in sampled_files:
            # 拷贝点云 .bin
            src_bin = os.path.join(src_velo_dir, file_name)
            dst_bin = os.path.join(dst_velo_dir, file_name)
            shutil.copy(src_bin, dst_bin)
            
            # 拷贝对应的标签 .label (同名文件，后缀不同)
            label_name = file_name.replace('.bin', '.label')
            src_label = os.path.join(src_label_dir, label_name)
            dst_label = os.path.join(dst_label_dir, label_name)
            
            # 确保标签文件存在且非空（如果官方缺标签，防止报错）
            if os.path.exists(src_label) and os.path.getsize(src_label) > 0:
                shutil.copy(src_label, dst_label)
            else:
                # 只有测试集(11-21)才会缺标签，训练集如果缺了需要警告
                print(f"警告：找不到对应的标签文件 {src_label}")

                
        print(f"序列 {seq} 抽取完成: {frames_per_seq} 帧。")

    print("Mini 数据集构建完美收官！")

if __name__ == "__main__":
    # 【注意】这里请填入你存放 data_odometry_velodyne 的真实解压合并路径
    # 假设你原始的完整数据放在了下面这个目录
    SOURCE_ROOT = "/mnt/f/Gongzihang/2026/data/SemanticKITTI" 
    
    # 这是我们要生成的小数据集的路径
    DEST_ROOT = "SemanticFAST-LIO2-prediction/SemanticKITTI_Mini"
    
    create_mini_dataset(SOURCE_ROOT, DEST_ROOT, frames_per_seq=90)
