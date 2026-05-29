"""
作用：提供基于 FastAPI 的点云推理 Web 接口，供 Windows 端的 Qt 程序调用。
功能：
1. 在应用启动时，初始化并加载基于 Pointcept 框架的 OctFormer 模型，驻留在 GPU 内存中。
2. 提供一个 /predict 路由，接收前端发送的点云坐标 (x, y, z)。
3. 将接收到的数据转换为模型所需的格式 (通过自定义算子构建八叉树等)，执行推理。
4. 将推理得到的逐点预测结果 (Labels) 返回给前端。
怎么实现的：利用 FastAPI 搭建轻量级 HTTP 服务器，结合 PyTorch 完成内存中的端到端推理。
"""

from fastapi import FastAPI, UploadFile, File
import torch
import numpy as np
import uvicorn
from io import BytesIO

from pointcept.utils.config import Config
from pointcept.models import build_model

app = FastAPI()
model = None

@app.on_event("startup")
def load_model():
    global model
    print("正在构建并加载模型...")
    
    # 1. 加载你训练该模型时所对应的配置文件
    # 训练脚本通常会在 exp/keypoint_swin3d/ 目录下保存一份 config.py 或 config.yaml
    config_path = "configs/pigseg/semseg-swin3d-v1m1-0-base.py" 
    cfg = Config.fromfile(config_path)
    
    # 2. 根据配置文件自动实例化正确的模型 (这样无论你训练的是 Swin3D 还是 PTv3，都能自动对应)
    model = build_model(cfg.model)
    
    # 3. 加载权重文件
    # 增加 weights_only=False 消除 FutureWarning，或者在确认安全的情况下设为 True
    checkpoint_path = "exp/Swin3D_PigSeg_0512/model/model_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 4. 注入权重
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    
    model.cuda()
    model.eval()
    print("模型加载完成，等待前端请求...")

@app.post("/predict")
async def predict_pointcloud(file: UploadFile = File(...)):
    content = await file.read()
    
    # 假设前端发送过来的是 (N, 6) 的数组，前3列是坐标，后3列是特征(法向量)
    # 请根据你实际 Qt 端发出的数据维度进行调整
    raw_points = np.frombuffer(content, dtype=np.float32).reshape(-1, 7)
    
    raw_coord = raw_points[:, 0:3]
    feat = raw_points[:, 3:]
    coord_feat = raw_points[:, 3:6] # 取决于你前面特征是怎么定义的
    
    # ==========================================
    # 1. 完全对齐 Dataset 的预处理逻辑
    # ==========================================
    coord = raw_coord.copy()
    
    # 去中心化
    centroid = np.mean(coord, axis=0)
    coord -= centroid
    
    # 归一化
    dist = np.sqrt(np.sum(coord ** 2, axis=1))
    m = np.max(dist) if dist.shape[0] > 0 else 0
    if m < 1e-6:
        m = 1.0
    scale = np.array(m, dtype=np.float32)
    
    coord = coord / scale
    
    # ==========================================
    # 2. 构造张量并转移到 GPU
    # ==========================================
    coord_tensor = torch.tensor(coord, dtype=torch.float32).cuda()
    feat_tensor = torch.tensor(feat, dtype=torch.float32).cuda()
    coord_feat_tensor = torch.tensor(coord_feat, dtype=torch.float32).cuda()
    offset_tensor = torch.tensor([coord_tensor.shape[0]], dtype=torch.int32).cuda()
    
    # ----------------------------------------------------
    # [新增核心修复]：计算离散化的网格坐标 grid_coord
    # ----------------------------------------------------
    # 注意：这里的 grid_size 必须与你训练模型时 config 文件中的设定保持一致！
    # 通常归一化后的点云 grid_size 为 0.01、0.02 或 0.05
    grid_size = 0.02 
    
    # 1. 将浮点坐标除以网格大小并四舍五入，转换为整数坐标
    grid_coord_tensor = torch.round(coord_tensor / grid_size).to(torch.int32)
    # 2. 稀疏算子（如 Spconv / Pointops）通常要求网格索引是非负的，所以减去最小值进行平移
    grid_coord_tensor = grid_coord_tensor - grid_coord_tensor.min(dim=0)[0]
    # ----------------------------------------------------

    data_dict = {
        "coord": coord_tensor,
        "feat": feat_tensor,
        "coord_feat": coord_feat_tensor,
        "offset": offset_tensor,
        "grid_coord": grid_coord_tensor  # <--- 将计算好的 grid_coord 喂给模型
    }

    # ==========================================
    # 3. 模型推理
    # ==========================================
    with torch.no_grad():
        output = model(data_dict) 
        print("模型输出的 Keys 是: ", output.keys())

        # --------------------------------------------------
        # [修改点]：因为 output 是个字典，需要提取内部的 Tensor
        # --------------------------------------------------
        # 打印一下看看字典里有哪些键，方便核对（可选，调试用）
        # print("Output keys:", output.keys())
        
        # 假设 Pointcept 模型将预测结果存放在 "pred" 或 "seg_logits" 键下
        # 请根据你的模型实际结构修改下面的键名（通常回归任务是 "pred"）
        if isinstance(output, dict):
            if "pred" in output:
                pred_tensor = output["pred"]
            elif "seg_logits" in output:
                # 如果是分割头的输出，这里可能需要 argmax 或者取特定维度
                pred_tensor = output["seg_logits"]
            else:
                # 如果不知道键名是什么，可以直接取字典里的第一个值，或者查看刚才 print 的 keys
                pred_tensor = list(output.values())[0] 
        else:
            pred_tensor = output

        # 现在 pred_tensor 肯定是个张量了，可以放心转 numpy
        pred_coords_normalized = pred_tensor.cpu().numpy() 

    # ==========================================
    # 4. 反向恢复真实世界坐标 (核心步骤！)
    # ==========================================
    # 预测出的坐标 * scale + centroid，恢复到与输入点云相同的尺度和位置
    pred_coords_original = (pred_coords_normalized * scale) + centroid

    # 5. 返回结果给 Qt
    result_bytes = pred_coords_original.astype(np.float32).tobytes()
    from fastapi.responses import Response
    return Response(content=result_bytes, media_type="application/octet-stream")

if __name__ == "__main__":
    # 在 WSL2 中启动，监听 0.0.0.0，允许 Windows 宿主机访问
    uvicorn.run(app, host="0.0.0.0", port=8000)