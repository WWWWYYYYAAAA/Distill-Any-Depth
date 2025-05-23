import h5py
import numpy as np
from PIL import Image
import os

# 定义路径
file_path = "data/nyu_depth_v2_labeled.mat"
# rgb_output_dir = "RGB"
# depth_output_dir = "Depth"
# os.makedirs(rgb_output_dir, exist_ok=True)
# os.makedirs(depth_output_dir, exist_ok=True)

# 打开 HDF5 文件
with h5py.File(file_path, 'r') as file:
    # 检查数据集是否存在
    if 'images' not in file or 'depths' not in file:
        print("未找到所需数据集 ('images' 或 'depths')！")
        exit()

    # 加载数据集
    images = file['images']
    depths = file['depths']
    num_samples = images.shape[0]
    print(f"发现 {num_samples} 个样本。图像形状：{images.shape}，深度图形状：{depths.shape}")
