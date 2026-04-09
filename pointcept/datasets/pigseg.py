import os
import numpy as np
from .builder import DATASETS
from .defaults import DefaultDataset

@DATASETS.register_module()
class PigDataset(DefaultDataset):
    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        
        data = np.load(data_path, allow_pickle=True)
        
        coord = data[:, 0:3].astype(np.float32)
        normal = data[:, 3:6].astype(np.float32)
        curvature = data[:, 6:7].astype(np.float32)
        segment = data[:, 7].astype(np.int32)
        
        if coord.shape[0] > 0:
            # 1. 致命 NaN 清洗 (剔除 Open3D 边缘点计算失败产生的 NaN，防止底层浮点异常)
            valid_nan = ~(np.isnan(normal).any(axis=1) | np.isnan(curvature).any(axis=1) | np.isnan(coord).any(axis=1))
            coord = coord[valid_nan]
            normal = normal[valid_nan]
            curvature = curvature[valid_nan]
            segment = segment[valid_nan]

        if coord.shape[0] > 0:
            # 2. 强制中心化与飞点过滤 (毫米级别)
            median_coord = np.median(coord, axis=0)
            coord = coord - median_coord
            dist = np.linalg.norm(coord, axis=1)
            valid_mask = dist < 5000.0  # 安全截断 5000 毫米 (5米) 外的离群噪点，绝不误伤主体
            
            coord = coord[valid_mask]
            normal = normal[valid_mask]
            curvature = curvature[valid_mask]
            segment = segment[valid_mask]

        data_dict = dict(
            coord=coord,
            normal=normal,
            curvature=curvature,
            segment=segment,
            name=name,
            index_valid_keys=["coord", "normal", "curvature", "segment"]
        )
        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split('.')[0]