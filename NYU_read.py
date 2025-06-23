import h5py
import numpy as np
from PIL import Image
import cv2
# import os

# 定义路径
# file_path = "data/nyu_depth_v2_labeled.mat"
file_path = "data/ID_1000.mat"
# rgb_output_dir = "RGB"
# depth_output_dir = "Depth"
# os.makedirs(rgb_output_dir, exist_ok=True)
# os.makedirs(depth_output_dir, exist_ok=True)

# 打开 HDF5 文件
with h5py.File(file_path, 'r') as file:
    # 检查数据集是否存在
    if 'images' not in file or 'depths' not in file:
        print("data not found ('images' 或 'depths')!")
        exit()

    # 加载数据集
    images = file['images']
    depths = file['depths']
    # labels = file['labels']
    num_samples = images.shape[0]
    print(f"发现 {num_samples} 个样本。图像形状：{images.shape}，深度图形状：{depths.shape}")
    # img  = labels[222]
    img = images[222]
    # for i in range(640):
    #     for j in range(480):
    #         print(img[i, j], end=" ")
    #     print("")
    # for i in range(1449):
    #     if labels[i].min() > 0:
    #         print(">0")
    # img_normalized = (img - img.min()) / (img.max() - img.min()) * 255
    # print(img.max() , img.min())
    # img_normalized = img_normalized.astype(np.uint8)
    # img_colored = cv2.applyColorMap(img_normalized, cv2.COLORMAP_INFERNO)
    img = np.array(img)
    img = img.transpose(1,2,0)
    
    cv2.imshow('label', img)
    cv2.waitKey(0)
    

    #发现 1449 个样本。图像形状：(1449, 3, 640, 480)，深度图形状：(1449, 640, 480)
# out = np.hstack((np.array(images), np.array(depths)))
    # print(np.max(depths), np.min(depths))
    # print(np.max(labels), np.min(labels))
    
# print(out.shape)

# for i, img in enumerate(images):