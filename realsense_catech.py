import pyrealsense2 as rs
import numpy as np
import cv2
import os

# 创建输出文件夹
color_dir = "./data/rs/images"
depth_dir = "./data/rs/depths"
os.makedirs(color_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

# 初始化相机管道
pipeline = rs.pipeline()
config = rs.config()

# 配置640x480分辨率（深度90FPS + RGB 30FPS）[2,3](@ref)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 90)  # 深度流：640x480@90Hz
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB流：640x480@30Hz

# 启动相机
profile = pipeline.start(config)

# 创建对齐对象（深度图对齐到RGB图）
align_to = rs.stream.color
align = rs.align(align_to)

# 获取深度比例系数（毫米转米）
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"深度比例系数: {depth_scale} (单位: 米/数值)")

frame_counter = 0  # 统一序号计数器

try:
    while True:
        # 等待帧数据
        frames = pipeline.wait_for_frames()
        
        # 深度图对齐到RGB坐标系[4,8](@ref)
        aligned_frames = align.process(frames)
        
        # 获取对齐后的帧
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
        
        # 转换为OpenCV格式
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # 深度图可视化（仅预览）
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )
        gray_image = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)
        # 显示实时画面
        cv2.imshow('RGB 640x480', color_image)
        cv2.imshow('Depth 640x480', depth_colormap)
        
        # 键盘控制
        key = cv2.waitKey(1)
        
        # 按't'保存图像对
        if key == ord('t'):
            frame_id = f"{frame_counter}"
            
            # 保存RGB图像（JPEG）
            cv2.imwrite(f"{color_dir}/{frame_id}.jpg", color_image)
            
            # 保存原始深度数据（16位PNG）
            cv2.imwrite(f"{depth_dir}/{frame_id}.png", depth_image)
            
            print(f"已保存: #{frame_id} | RGB: {color_image.shape} | Depth: {depth_image.shape}")
            frame_counter += 1
        
        # 按'q'退出
        elif key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"程序结束 | 共保存 {frame_counter} 组图像对")