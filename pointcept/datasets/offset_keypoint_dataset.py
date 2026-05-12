import os
import glob
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset
from pointcept.datasets.builder import DATASETS
from pointcept.datasets.transform import Compose

@DATASETS.register_module()
class OffsetKeypointDataset(Dataset):
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
        feature_files = glob.glob(os.path.join(split_path, "pointclouds", "*.npy"))
        data_list = []

        for feat_path in feature_files:
            filename = os.path.basename(feat_path)
            
            # 去掉后缀拿到时间戳（或者说base name）
            timestamp = os.path.splitext(filename)[0]
            
            # 拼接标签路径: 指向 keypoints 文件夹，使用新标签后缀 _keypoint_offset.npy
            label_filename = f"{timestamp}_keypoint_offset.npy"
            label_path = os.path.join(split_path, "keypoints", label_filename)

            # 验证特征文件和标签文件是否成对存在
            if os.path.exists(label_path):
                data_list.append({
                    "feat_path": feat_path,
                    "label_path": label_path,
                    "name": timestamp
                })
            else:
                print(f"⚠️ 警告: 找不到特征文件对应的标签 -> {label_filename}")
        
        return data_list

    def __len__(self):
        return len(self.data_list) * self.loop

    def __getitem__(self, idx):
        """
        代码作用：获取单个数据样本。
        标签 target 现在是一个形状为 (N, 6, 4) 的张量，其中前三维是 xyz 偏移量，第四维是 mask
        """
        idx = idx % len(self.data_list)
        info = self.data_list[idx]
        
        # 1. 加载数据
        raw_data = np.load(info["feat_path"]).astype(np.float32)
        coord = raw_data[:, 0:3]
        feat = raw_data[:, 3:]
        target = np.load(info["label_path"]).astype(np.float32)  # (N, 6, 4)

        # 校验点云本身是否有 Nan/Inf
        if np.isnan(coord).any() or np.isinf(coord).any():
            coord = np.nan_to_num(coord)
            
        if np.isnan(target).any() or np.isinf(target).any():
            target = np.nan_to_num(target)

        # 检查 target 形状是否合法 (应该为 N, 6, 4)
        if len(target.shape) != 3 or target.shape[1] != 6 or target.shape[2] != 4:
            print(f"⚠️ 警告: 样本 {info['name']} 的偏移量标签形状异常 (当前为 {target.shape})，已自动跳过！")
            new_idx = np.random.randint(0, len(self.data_list))
            return self.__getitem__(new_idx)
            
        # 确保 target 和 coord 点云数量一致
        if target.shape[0] != coord.shape[0]:
            print(f"⚠️ 警告: 样本 {info['name']} 的标签点数与点云数不匹配，已自动跳过！")
            new_idx = np.random.randint(0, len(self.data_list))
            return self.__getitem__(new_idx)

        # 提取给模型提供位置信息的特征
        coord_feat = raw_data[:, 3:6]

        # 2. 去中心化 
        # offset 是相对位移（向量），不会因为平移坐标系而改变，所以 target 这边不需要去中心化，只要中心化 coord 即可
        centroid = np.mean(coord, axis=0)
        coord -= centroid

        # 3. 归一化
        # 但是对于缩放 scale 操作，由于点云缩放了，因此距离偏移量（offset）也要跟着等比例缩放！掩码层(mask)保持不变。
        dist = np.sqrt(np.sum(coord ** 2, axis=1))
        m = np.max(dist) if dist.shape[0] > 0 else 0
        if m < 1e-6:
            m = 1.0
        scale = np.array(m, dtype=np.float32) 
        
        coord = coord / scale
        
        # 将位移按照 scale 进行缩放， mask维不变
        target[..., :3] = target[..., :3] / scale

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

        # 4. 应用变换
        if self.transform is not None:
            data_dict = self.transform(data_dict)
            
        return data_dict
