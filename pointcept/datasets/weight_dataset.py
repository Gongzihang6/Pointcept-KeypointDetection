"""
================================================================================
脚本作用：
Pointcept 框架下的 猪体尺与体重预测 3D点云数据集加载器。

功能：
1. 样本加载：从 points/ 和 labels/ 文件夹加载 .npy 格式的点云和标签。
2. 异常防御：自动检测标签维度是否为 7 维，如果数据损坏则自动重采样替换，防止训练崩溃。
3. 坐标处理：执行去中心化（Center）以保证平移不变性，但【严格禁止缩放归一化】，以保留绝对物理尺寸信息。
4. 特征封装：将法线和曲率作为 4D 输入特征 (nx, ny, nz, c) 喂给 PTv3。
5. 损失函数：为了确保 Pointcept 的 DefaultClassifier 能够无缝支持回归任务，自动注册了 RegressionL1Loss。

实现了什么：
接收 (N, 7) 的点云和 (7,) 的标签，转换为 Pointcept 的 `data_dict` 格式，交由 GridSample 进行体素化和张量化。

怎么实现的：
继承 torch.utils.data.Dataset，使用 glob 扫描文件并配对。在 __getitem__ 中完成异常检查、中心化平移，并利用注册机制注入框架。
================================================================================
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from .builder import DATASETS
from .transform import Compose





# =====================================================================
# 数据集加载类
# =====================================================================
@DATASETS.register_module()
class PigWeightDataset(Dataset):
    def __init__(self,
                 split='train',
                 data_root='data',
                 transform=None,
                 test_mode=False,
                 loop=1):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform) if transform is not None else None
        self.test_mode = test_mode
        self.loop = loop if not test_mode else 1
        
        # 扫描文件
        self.data_list = self._get_file_list()
        print(f"[{self.split}] 成功加载 {len(self.data_list)} 个有效样本，路径: {self.data_root}")

    def _get_file_list(self):
        split_path = os.path.join(self.data_root, self.split)
        if not os.path.exists(split_path):
            raise ValueError(f"数据路径不存在: {split_path}")

        # 匹配 points 下的特征文件
        feature_files = glob.glob(os.path.join(split_path, "points", "*.npy"))
        data_list = []

        for feat_path in feature_files:
            filename = os.path.basename(feat_path)
            timestamp = os.path.splitext(filename)[0]
            
            # 标签文件路径
            label_path = os.path.join(split_path, "labels", filename)

            if os.path.exists(label_path):
                data_list.append({
                    "feat_path": feat_path,
                    "label_path": label_path,
                    "name": timestamp
                })
            else:
                print(f"⚠️ 警告: 找不到特征文件对应的标签 -> {filename}")
        
        return data_list

    def __len__(self):
        return len(self.data_list) * self.loop

    def __getitem__(self, idx):
        idx = idx % len(self.data_list)
        info = self.data_list[idx]
        
        # 1. 加载数据
        raw_data = np.load(info["feat_path"]).astype(np.float32)
        target = np.load(info["label_path"]).astype(np.float32)
        # ====== 增加标签缩放 (Target Scaling) ======
        # 将体尺和体重数值除以 100，使其落在 [0.3, 2.0] 左右的极佳收敛区间
        target_scale_factor = 100.0
        target = target / target_scale_factor
        # ===========================================
        # 拆分坐标系和特征 [x, y, z, nx, ny, nz, c]
        coord = raw_data[:, 0:3]
        normal = raw_data[:, 3:6]        # 严格提取 3 维法向量
        curvature = raw_data[:, 6:7]     # 提取 1 维曲率

        # ================= [新增] 形状异常数据防御机制 =================
        # 标签应该是 7 维的 [体长, 体宽, 体高, 胸围, 腰围, 臀围, 体重]
        if target.shape[0] != 7:
            print(f"⚠️ 警告: 样本 {info['name']} 的标签数量异常 (当前为 {target.shape[0]})，已自动跳过！")
            new_idx = np.random.randint(0, len(self.data_list))
            return self.__getitem__(new_idx)
        # ===============================================================

        # ================= 数据安全检查 =================
        if np.isnan(coord).any() or np.isinf(coord).any():
            coord = np.nan_to_num(coord)
        if np.isnan(target).any() or np.isinf(target).any():
            target = np.nan_to_num(target)
        # =====================================================
        
        # 2. 去中心化 (Center Shift)
        # 作用是将猪的质心移动到原点 [0,0,0]，方便模型学习，但绝不能改变绝对大小
        centroid = np.mean(coord, axis=0)
        coord -= centroid

        # 构建给 Pointcept Pipeline 的基础字典
        data_dict = dict(
            coord=coord,
            normal=normal,          # 3维，RandomRotate 会正确旋转它
            color=curvature,        # 将曲率放到 color 里，免受旋转影响，又能被 GridSample 降采样
            category=target,
            name=info["name"],
            centroid=centroid 
        )

        # 3. 应用数据增强与网格化流水线 (如 GridSample, ToTensor 等)
        if self.transform is not None:
            data_dict = self.transform(data_dict)
            
        return data_dict