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
            raise ValueError(f"Data path does not exist: {split_path}")

        # 匹配特征文件: *d_pc_clipped.npy
        feature_files = glob.glob(os.path.join(split_path, "*_d_pc_clipped.npy"))
        data_list = []

        for feat_path in feature_files:
            filename = os.path.basename(feat_path)
            # 解析文件名: dev_2_005J_20251102_110034_430_d_pc_clipped.npy
            # parts: ['dev', '2', '005J', '20251102', '110034', '430', ...]
            parts = filename.split('_')
            
            # 提取时间戳 (根据你的示例是第3,4,5个部分)
            # 请根据实际文件名调整这里的索引
            try:
                timestamp = f"{parts[3]}_{parts[4]}_{parts[5]}"
            except IndexError:
                print(f"Skipping invalid filename: {filename}")
                continue
            
            label_filename = f"关键点坐标_{timestamp}.npy"
            label_path = os.path.join(split_path, label_filename)

            if os.path.exists(label_path):
                data_list.append({
                    "feat_path": feat_path,
                    "label_path": label_path,
                    "name": filename
                })
        
        return data_list

    def __len__(self):
        return len(self.data_list) * self.loop

    def __getitem__(self, idx):
        idx = idx % len(self.data_list)
        info = self.data_list[idx]
        
        # 1. 加载数据
        raw_data = np.load(info["feat_path"]).astype(np.float32)
        coord = raw_data[:, 0:3]
        feat = raw_data[:, 3:]
        target = np.load(info["label_path"]).astype(np.float32)

        # 提取给 Swin3D 做位置编码辅助的特征 (coord_feat)
        # Swin3D 需要这个键。既然没有 RGB，就用 "法向量" (第3,4,5列) 代替
        # 维度: (N, 3)
        coord_feat = raw_data[:, 3:6]

        # ================= [新增] 数据安全检查 =================
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
        # coord = coord / m  <-- 原代码
        # target = target / m <-- 原代码
        
        # [修改建议] 防止 m 为 0 (虽然你前面处理了，但双重保险更好)
        scale = np.array(m, dtype=np.float32) 
        
        coord = coord / scale
        target = target / scale

        # 构造数据字典
        data_dict = dict(
            coord=coord,
            feat=feat,
            target=target, 
            coord_feat=coord_feat,  # [新增] Swin3D 位置编码辅助特征
            name=info["name"],
            centroid=centroid, 
            scale=scale  # [重点] 这里传入 numpy 数组，方便 DataLoader 自动堆叠
        )

        # 4. 应用变换 (GridSample 等)
        if self.transform is not None:
            data_dict = self.transform(data_dict)
            
        return data_dict

