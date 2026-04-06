"""
作用：使用预训练的 Pointcept 模型对 SemanticKITTI 单帧/序列点云进行语义分割推理。
功能：加载指定配置、模型架构和权重，逐帧读取 .bin 文件，通过网络前向传播生成预测，并保存为 .npy 标签。
实现了什么：
    1. 解析并加载 Pointcept 的 Python/YAML 配置文件，构建如 PTv3 等复杂的 3D 深度学习网络架构。
    2. 自动挂载预训练权重 (.pth) 并将模型设置为评估 (Eval) 模式。
    3. 调用配置文件中定义的 test_pipeline 进行数据预处理（如体素化、网格采样、坐标归一化）。
    4. 将前向传播输出的 Logits (概率分布) 转换为最终的类别 ID，并保存至指定目录。
怎么实现的：
    1. 引入 pointcept.utils.config 读取配置，并使用 build_model 动态实例化网络。
    2. 使用 torch.load 获取网络权重，并通过 load_state_dict 注入到模型实例中。
    3. 模拟标准 DataLoader 的行为，利用 pointcept.datasets.transform.Compose 按序处理维度为 $N \times 4$ 的原始点云矩阵。
    4. 针对 Batch Size = 1 的情况，手动构造 offset 张量（Point Transformer V3 等稀疏网络强依赖此张量区分批次边界），执行 torch.no_grad() 加速推理并使用 $\arg\max$ 提取类别。
"""
import sys
import os
import glob
import torch
import numpy as np
from tqdm import tqdm

# 添加项目根目录到 python path，确保能导入 pointcept
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# --- 引入 Pointcept 核心组件 ---
from pointcept.utils.config import Config, DictAction
from pointcept.models import build_model
from pointcept.datasets import build_dataset, point_collate_fn
from pointcept.utils.misc import intersection_and_union, make_dirs
from pointcept.engines.defaults import default_argument_parser
from pointcept.datasets.transform import Compose


# --- 1. 路径与参数配置 ---
# 根据您的文件结构设定的路径
BIN_DIR = "/mnt/f/Gongzihang/2026/data/SemanticKITTI/dataset/sequences/05/velodyne"
PRED_DIR = "/home/gzh/point/Pointcept-KeypointDetection/exp/semantickitti/semseg-pt-v2m2-0-base/results/full_05"

# Pointcept 对应的配置与权重路径（请根据实际相对/绝对路径微调）
CONFIG_PATH = "/home/gzh/point/Pointcept-KeypointDetection/configs/semantic_kitti/semseg-pt-v2m2-0-base.py"
WEIGHT_PATH = "/home/gzh/point/Pointcept-KeypointDetection/exp/SemanticKITTI_PTV2_Mini/model/model_best.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    os.makedirs(PRED_DIR, exist_ok=True)

    # --- 2. 加载配置与模型 ---
    print(f"正在加载配置文件: {CONFIG_PATH}")
    cfg = Config.fromfile(CONFIG_PATH)
    
    # 【修复 1：读取适合单帧推理的 val transform，避开路径错误和复杂的 TTA】
    print("正在构建纯净版推理数据流水线...")
    custom_transform_list = [
        dict(
            type="GridSample",
            grid_size=0.05,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
            return_inverse=True, # 强制要求底层 C++ 算子返回映射索引
        ),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "segment", "inverse"), # 强制挂载到 data_dict
            feat_keys=("coord", "strength"),
        ),
    ]
    test_transform = Compose(custom_transform_list)
    
    print("正在构建模型架构...")
    model = build_model(cfg.model).to(DEVICE)
    
    print(f"正在加载模型权重: {WEIGHT_PATH}")
    checkpoint = torch.load(WEIGHT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()

    # --- 3. 获取点云文件列表 ---
    bin_files = sorted(glob.glob(os.path.join(BIN_DIR, "*.bin")))
    if not bin_files:
        print("未找到 .bin 文件，请检查 BIN_DIR 路径！")
        return
    print(f"共发现 {len(bin_files)} 帧点云，开始推理...")

    # --- 4. 核心推理循环 ---
    with torch.no_grad():
        for bin_path in tqdm(bin_files, desc="Inference Progress"):
            file_name = os.path.splitext(os.path.basename(bin_path))[0]
            pred_save_path = os.path.join(PRED_DIR, f"{file_name}.npy")
            
            scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
            coord = scan[:, :3]
            strength = scan[:, 3].reshape(-1, 1)

            data_dict = {
                "coord": coord,
                "strength": strength,
                "segment": np.zeros(len(coord), dtype=np.int32), 
            }
            
            # 经过 Transform Pipeline (包含体素化，此时点数会减少，并生成 inverse 索引)
            data_dict = test_transform(data_dict)
            
            # 【注意】这里的 num_points 是体素化后的点数，不是原始点数
            num_points = data_dict["coord"].shape[0]
            data_dict["offset"] = torch.tensor([num_points], dtype=torch.int32)
            data_dict["batch"] = torch.zeros(num_points, dtype=torch.long)
            
            for key in data_dict.keys():
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].to(DEVICE)
                elif isinstance(data_dict[key], np.ndarray):
                    data_dict[key] = torch.from_numpy(data_dict[key]).to(DEVICE)
            
            # --- 网络前向传播 ---
            output = model(data_dict)
            
            if isinstance(output, dict):
                seg_logits = output["seg_logits"]
            else:
                seg_logits = output
                
            # 获取体素化级别的预测 ID
            pred_labels = seg_logits.argmax(dim=1).cpu().numpy().astype(np.int32)
            
            # print(f"当前 data_dict 包含的键: {data_dict.keys()}")   #  dict_keys(['coord', 'grid_coord', 'segment', 'inverse', 'offset', 'feat', 'batch'])
            # 【修复 2：将体素预测逆向映射回原始点云，保证 len(xyz) == len(labels)】
            if "inverse" in data_dict:
                inverse_map = data_dict["inverse"].cpu().numpy()
                pred_labels = pred_labels[inverse_map]
            
            # 保存为 .npy
            np.save(pred_save_path, pred_labels)

    print(f"\n推理完成！所有预测标签已存入: {PRED_DIR}")

if __name__ == "__main__":
    main()