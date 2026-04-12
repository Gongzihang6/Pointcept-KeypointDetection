import os
import glob
from typing import Any
from typing import Any
from typing import Any
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset
from .builder import DATASETS
from .transform import Compose, TRANSFORMS

@DATASETS.register_module()
class KeypointDataset(Dataset):
    def __init__(self,
                 split='train',
                 data_root='data',
                 transform=None,
                 test_mode=False,
                 loop=1):
        super().__init__()
        self.data_root = data_root
        self.split = split
        # 加载转换流水线 (GridSample, ToTensor 等)
        self.transform = Compose(transform)
        self.test_mode = test_mode
        self.loop = loop if not test_mode else 1
        
        # 扫描文件
        self.data_list = self._get_file_list()
        print(f"[{self.split}] Loaded {len(self.data_list)} samples from {self.data_root}")

    def _get_file_list(self):
        split_path = os.path.join(self.data_root, self.split)
        if not os.path.exists(split_path):
            raise ValueError(f"数据路径不存在: {split_path}")

        # 1. 匹配特征文件: 直接匹配 pointclouds 下的所有 .npy 文件
        # 使用 os.path.join 保证路径拼接的跨平台兼容性
        feature_files = glob.glob(os.path.join(split_path, "pointclouds", "*.npy"))
        data_list = []

        for feat_path in feature_files:
            filename = os.path.basename(feat_path)
            
            # 2. 解析文件名: 例如 20260329_105410_942.npy
            # 因为我们之前保存的名字就是纯时间戳，直接去掉后缀即可，不需要复杂的 split
            timestamp = os.path.splitext(filename)[0]
            
            # 3. 拼接标签路径: 指向 keypoints 文件夹
            # 对应的标签文件命名为: 时间戳_关键点坐标.npy
            label_filename = f"{timestamp}_关键点坐标.npy"
            label_path = os.path.join(split_path, "keypoints", label_filename)

            # 4. 验证特征文件和标签文件是否成对存在
            if os.path.exists(label_path):
                data_list.append({
                    "feat_path": feat_path,
                    "label_path": label_path,
                    "name": timestamp  # 直接用时间戳作为该样本的名称标识
                })
            else:
                # 方便排查是否有数据丢失
                print(f"⚠️ 警告: 找不到特征文件对应的标签 -> {label_filename}")
        
        return data_list

    def __len__(self):
        return len(self.data_list) * self.loop

    def __getitem__(self, idx):
        """
        代码作用：获取单个数据样本，并在遇到损坏/残缺数据时自动重采样。
        
        修改点说明：
        在加载完 target (标签) 之后，立即检查其维度 `target.shape[0]`。
        如果不是 6，则打印警告，并递归调用自身获取一个随机的新样本，防止脏数据流入后续的 Batch 拼接环节导致崩溃。
        """
        idx = idx % len(self.data_list)
        info = self.data_list[idx]
        
        # 1. 加载数据
        raw_data = np.load(info["feat_path"]).astype(np.float32)
        coord = raw_data[:, 0:3]
        feat = raw_data[:, 3:]
        target = np.load(info["label_path"]).astype(np.float32)

        # ================= [新增] 形状异常数据防御机制 =================
        # 检查关键点数量是否严格等于 6
        if target.shape[0] != 6:
            print(f"⚠️ 警告: 样本 {info['name']} 的关键点数量异常 (当前为 {target.shape[0]})，已自动跳过！")
            # 随机生成一个新的索引来顶替当前这个坏样本
            # 注意：因为上面已经 import numpy as np，这里直接用 np.random.randint 即可
            new_idx = np.random.randint(0, len(self.data_list))
            return self.__getitem__(new_idx)
        # ===============================================================

        # 提取给 Swin3D 做位置编码辅助的特征 (coord_feat)
        # Swin3D 需要这个键。既然没有 RGB，就用 "法向量" (第3,4,5列) 代替
        # 维度: (N, 3)
        coord_feat = raw_data[:, 3:6]

        # ================= 数据安全检查 =================
        # 检查是否有 NaN 或 Inf
        if np.isnan(coord).any() or np.isinf(coord).any():
            print(f"⚠️ Warning: Found NaN/Inf in {info['name']}, replacing with 0.")
            coord = np.nan_to_num(coord) # 将 NaN 替换为 0
            
        if np.isnan(target).any() or np.isinf(target).any():
            print(f"⚠️ Warning: Found NaN/Inf in target of {info['name']}, replacing with 0.")
            target = np.nan_to_num(target)
        # =====================================================
        
        # 2. 去中心化
        centroid = np.mean(coord, axis=0)
        coord -= centroid
        target -= centroid

        # 3. 归一化
        # 增加 eps 防止除以 0 的隐患
        dist = np.sqrt(np.sum(coord ** 2, axis=1))
        m = np.max(dist) if dist.shape[0] > 0 else 0
        
        if m < 1e-6:
            m = 1.0
        
        m = float(m) 
        
        # 防止 m 为 0
        scale = np.array(m, dtype=np.float32) 
        
        coord = coord / scale
        target = target / scale

        # 构造数据字典
        data_dict = dict(
            coord=coord,
            feat=feat,
            target=target, 
            coord_feat=coord_feat,  
            name=info["name"],
            centroid=centroid, 
            scale=scale  
        )

        # 4. 应用变换 (GridSample 等)
        if self.transform is not None:
            data_dict = self.transform(data_dict)
            
        return data_dict

