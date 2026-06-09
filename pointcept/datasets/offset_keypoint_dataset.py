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
                 loop=1,
                 offset_radius=None,
                 online_offset=None,
                 num_keypoints=6):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.offset_radius = None if offset_radius is None else float(offset_radius)
        self.online_offset = self.offset_radius is not None if online_offset is None else online_offset
        self.num_keypoints = num_keypoints
        if self.online_offset and self.offset_radius is None:
            raise ValueError("online_offset=True 时必须设置 offset_radius")
        # 加载转换流水线 (GridSample, ToTensor 等)
        self.transform = Compose(transform)
        self.test_mode = test_mode
        self.loop = loop if not test_mode else 1
        
        # 扫描文件
        self.data_list = self._get_file_list()
        mode = f"online offset labels, R={self.offset_radius}" if self.online_offset else "precomputed offset labels"
        print(f"[{self.split}] Loaded {len(self.data_list)} samples from {self.data_root} ({mode})")

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

            if self.online_offset:
                keypoint_path = self._find_keypoint_path(split_path, timestamp)
                if keypoint_path is None:
                    print(f"⚠️ 警告: 找不到特征文件对应的关键点坐标 -> {timestamp}_关键点坐标.npy / {timestamp}.npy")
                    continue
                data_list.append({
                    "feat_path": feat_path,
                    "keypoint_path": keypoint_path,
                    "name": timestamp
                })
            else:
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

    def _find_keypoint_path(self, split_path, timestamp):
        kp_dir = os.path.join(split_path, "keypoints")
        candidates = [
            os.path.join(kp_dir, f"{timestamp}_关键点坐标.npy"),
            os.path.join(kp_dir, f"{timestamp}.npy"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _generate_offset_target(self, coord, keypoint, sample_name):
        keypoint = np.asarray(keypoint, dtype=np.float32)
        if keypoint.ndim == 1 and keypoint.size % 3 == 0:
            keypoint = keypoint.reshape(-1, 3)

        if keypoint.shape != (self.num_keypoints, 3):
            raise ValueError(
                f"样本 {sample_name} 的关键点坐标形状异常: {keypoint.shape}, "
                f"期望为 ({self.num_keypoints}, 3)"
            )

        offsets = keypoint[np.newaxis, :, :] - coord[:, np.newaxis, :]
        distances = np.linalg.norm(offsets, axis=-1)
        mask = (distances <= self.offset_radius).astype(np.float32)
        mask_expanded = mask[..., np.newaxis]

        target = np.empty((coord.shape[0], self.num_keypoints, 4), dtype=np.float32)
        target[..., :3] = offsets * mask_expanded
        target[..., 3] = mask
        return target

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

        # 校验点云本身是否有 Nan/Inf；在线生成 target 时也要使用清理后的坐标。
        if np.isnan(coord).any() or np.isinf(coord).any():
            coord = np.nan_to_num(coord)

        if self.online_offset:
            keypoint = np.load(info["keypoint_path"]).astype(np.float32)
            if np.isnan(keypoint).any() or np.isinf(keypoint).any():
                keypoint = np.nan_to_num(keypoint)
            try:
                target = self._generate_offset_target(coord, keypoint, info["name"])
            except ValueError as e:
                print(f"⚠️ 警告: {e}，已自动跳过！")
                new_idx = np.random.randint(0, len(self.data_list))
                return self.__getitem__(new_idx)
        else:
            target = np.load(info["label_path"]).astype(np.float32)  # (N, 6, 4)

        if np.isnan(target).any() or np.isinf(target).any():
            target = np.nan_to_num(target)

        # 检查 target 形状是否合法 (应该为 N, num_keypoints, 4)
        if len(target.shape) != 3 or target.shape[1] != self.num_keypoints or target.shape[2] != 4:
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
