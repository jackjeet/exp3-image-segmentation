import torch
import numpy as np
from PIL import Image
from data_loader import test_dataset  # 复用已加载的测试集

# --------------------------
# 随机提取一张测试集图像（可修改类别或索引）
# --------------------------
# 选择要提取的类别（0-9对应10类，这里选索引3=cat，可修改）
target_class = 3  # 0=飞机,1=汽车,2=鸟,3=猫,4=鹿,5=狗,6=青蛙,7=马,8=船,9=卡车
image_index = None

# 从测试集中找目标类别的第一张图像
for idx, (img, label) in enumerate(test_dataset):
    if label == target_class:
        image_index = idx
        break

# 提取并反标准化图像
image, label = test_dataset[image_index]
image_np = image.numpy().transpose(1, 2, 0)  # 转换为(H,W,C)
mean = np.array([0.4914, 0.4822, 0.4465])
std = np.array([0.2023, 0.1994, 0.2010])
image_np = std * image_np + mean  # 反标准化
image_np = np.clip(image_np, 0, 1)  # 修正像素范围
image_pil = Image.fromarray((image_np * 255).astype(np.uint8))  # 转为PIL图像

# 保存图像
image_name = f'./test_{test_dataset.classes[label]}.jpg'
image_pil.save(image_name)

# 输出信息
print(f"✅ 测试图像生成成功！")
print(f"图像路径：{image_name}")
print(f"真实类别：{test_dataset.classes[label]}")
print(f"图像尺寸：{image_pil.size}（自动适配32×32）")