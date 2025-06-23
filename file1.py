import numpy as np
import cv2
import os

def create_image_data_file( output_file="path/to/your/output/image_depth_data.py",  # 输出的Python文件路径
            image_dir="path/to/your/image/folder", 
            depth_dir="path/to/your/depth/folder"  ):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(image_files) != len(depth_files):
        raise ValueError("图片和深度图数量不匹配")
    
    images_data = []
    depths_data = []
    

    for img_file, depth_file in zip(image_files, depth_files):
        img_path = os.path.join(image_dir, img_file)
        depth_path = os.path.join(depth_dir, depth_file)
        
        img = cv2.imread(img_path)
        depth = cv2.imread(depth_path, 0) 
        
        if img is None:
            raise ValueError(f"无法读取图片: {img_path}")
        if depth is None:
            raise ValueError(f"无法读取深度图: {depth_path}")
        
        img = cv2.resize(img, (640, 480))
        depth = cv2.resize(depth, (640, 480))
        
        images_data.append(img.tolist())
        depths_data.append(depth.tolist())
    
    # 生成Python文件内容
    file_content = f"""
# 自动生成的图片和深度图数据
import numpy as np

# 图片数据 (BGR格式, 640×480)
images = {images_data}

# 深度图数据 (单通道, 640×480)
depths = {depths_data}
"""
    
    # 写入文件
    with open(output_file, 'w') as f:
        f.write(file_content)
    
    print(f"数据已成功保存到 {output_file}")
