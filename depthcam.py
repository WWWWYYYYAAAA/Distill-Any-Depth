
import time

import torch
from PIL import Image
import numpy as np
from distillanydepth.modeling.archs.dam.dam import DepthAnything
from distillanydepth.depth_anything_v2.dpt import DepthAnythingV2
from distillanydepth.utils.image_util import chw2hwc, colorize_depth_maps
from distillanydepth.midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import cv2
# from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# 初始化摄像头
cap = cv2.VideoCapture(1)

# 检查摄像头是否成功打开
if not cap.isOpened():
    raise IOError("无法打开摄像头")

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "./model/model_small.safetensors"
model_size = "small"

model_kwargs = {
        "large": dict(
            encoder="vitl", 
            features=256, 
            out_channels=[256, 512, 1024, 1024], 
            use_bn=False, 
            use_clstoken=False, 
            max_depth=150.0, 
            mode='disparity',
            pretrain_type='dinov2',
            del_mask_token=False
        ),
        "base": dict(
            encoder='vitb',
            features=128,
            out_channels=[96, 192, 384, 768],
        ),
        "small": dict(
            encoder='vits',
            features=64,
            out_channels=[48, 96, 192, 384],
        )
    }

if model_size == "large":
    model = DepthAnything(**model_kwargs[model_size]).to(device)
else:
    model = DepthAnythingV2(**model_kwargs[model_size]).to(device)

model_weights = load_file(checkpoint_path)
model.load_state_dict(model_weights)
model = model.to(device)


t1, t2, t3, t4 = 0, 0, 0, 0
size_h = 200
size_w = 200
ones_subarray = np.zeros((size_h, size_w, 3))
start_h = 160
start_w = 160
# 处理函数
def process_frame(image):
    # # 转换颜色空间并调整尺寸
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # original_height, original_width = frame_rgb.shape[:2]
    
    # # # 转换为模型输入格式
    # input_tensor = torch.from_numpy(frame_rgb).permute(2,0,1).float().to(device)
    # input_tensor = input_tensor.unsqueeze(0) / 255.0  # 添加batch维度并归一化

    # # 执行推理
    # with torch.no_grad():
    #     depth = model.infer(input_tensor)

    # # 后处理
    # depth_np = depth.squeeze().cpu().numpy()
    # depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min()) * 255
    # depth_normalized = depth_normalized.astype(np.uint8)
    
    # # 应用颜色映射并恢复原始尺寸
    # # depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
    # # depth_colored = cv2.resize(depth_colored, (original_width, original_height))
    
    
    # from zoedepth.utils.misc import colorize

    # depth_colored = colorize(depth_normalized)
    # depth_colored = cv2.resize(depth_colored, (original_width, original_height))
    # depth_colored = depth_colored[:, :, :3]
    # return depth_colored
    if model is None:
        return None

    # Preprocess the image
    image_np = np.array(image)[..., ::-1] / 255
    image_np[start_h:start_h+size_h, start_w:start_w+size_w, :] = ones_subarray
    # image_np[start_h+20:start_h+size_h+20, start_w+20:start_w+size_w+20, :] = ones_subarray
    # print(np.shape(image_np))
    t2 = time.time()
    transform = Compose([
        Resize(700, 700, resize_target=False, keep_aspect_ratio=False, ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet()
    ])

    image_tensor = transform({'image': image_np})['image']
    image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).to(device)
    # print(image_tensor.shape)
    with torch.no_grad():  # Disable autograd since we don't need gradients on CPU
        pred_disp, _ = model(image_tensor)
    torch.cuda.empty_cache()
    # Ensure the depth map is in the correct shape before colorization
    pred_disp_np = pred_disp.cpu().detach().numpy()[0, 0, :, :]  # Remove extra singleton dimensions
    
    # Normalize depth map
    pred_disp = (pred_disp_np - pred_disp_np.min()) / (pred_disp_np.max() - pred_disp_np.min())
    # pred_disp = pred_disp_np / 10
    print("\r",pred_disp_np.max(),pred_disp_np.min(),end=" ")
    # Colorize depth map
    cmap = "Spectral_r"
    depth_colored = colorize_depth_maps(pred_disp[None, ..., None], 0, 1, cmap=cmap).squeeze()  # Ensure correct dimension
    
    # Convert to uint8 for image display
    depth_colored = (depth_colored * 255).astype(np.uint8)
    
    # Convert to HWC format (height, width, channels)
    depth_colored_hwc = chw2hwc(depth_colored)
    
    # Resize to match the original image dimensions (height, width)
    h, w = image_np.shape[:2]
    depth_colored_hwc = cv2.resize(depth_colored_hwc, (w, h), cv2.INTER_LINEAR)
    # print(depth_colored_hwc.shape, image_np.shape, h, w)
    # Convert to a PIL image
    # depth_image = Image.fromarray(depth_colored_hwc)
    return depth_colored_hwc

fps = 0

# 主循环
try:
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
            
        # 处理帧并显示结果
        depth_frame = process_frame(frame)
        # depth_frame = np.hstack((orin_frame, depth_frame))
        end_time = time.time()
        processing_time = end_time - start_time
        fps = 0.1*fps + 0.9*(1/processing_time)  # 平滑处理
        cv2.putText(depth_frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('RGB-Depth', depth_frame)
        # print(np.shape(frame), np.shape(depth_frame))
        # 按ESC退出
        if cv2.waitKey(1) == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()