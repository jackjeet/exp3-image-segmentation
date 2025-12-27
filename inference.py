import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from models import SimpleCNN
from data_loader import val_test_transform

# --------------------------
# 解决中文显示和绘图后端问题
# --------------------------
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 支持中文
plt.rcParams["axes.unicode_minus"] = False    # 解决负号显示
plt.switch_backend('agg')  # 切换到非交互式后端，避免PyCharm报错

# 类别名称
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


# --------------------------
# 1. 加载模型
# --------------------------
def load_model(model_path='./best_model_5epochs.pth'):
    model = SimpleCNN(num_classes=10)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("模型加载成功，可用于推理")
    return model


# --------------------------
# 2. 预处理单张图像
# --------------------------
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = val_test_transform
    image_tensor = transform(image).unsqueeze(0)
    return image, image_tensor


# --------------------------
# 3. 模型推理并保存结果（不显示，直接保存图片）
# --------------------------
def predict_image(model, image_path):
    image, image_tensor = preprocess_image(image_path)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = classes[predicted.item()]
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100

    # 保存结果图像（不显示，避免交互错误）
    plt.figure(figsize=(6, 4))
    plt.imshow(image)
    plt.title(f'预测类别: {predicted_class}\n置信度: {probabilities[predicted.item()]:.2f}%')
    plt.axis('off')
    plt.savefig('./inference_result.jpg', dpi=150)  # 保存到当前文件夹
    print(f"推理结果已保存为 './inference_result.jpg'")

    return predicted_class


# --------------------------
# 执行推理
# --------------------------
if __name__ == '__main__':
    model = load_model()
    test_image_path = './test_cat.jpg'  # 确保这个文件存在
    predict_image(model, test_image_path)
    print("推理完成，可查看保存的结果图片")