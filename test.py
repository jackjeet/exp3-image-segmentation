import torch
import numpy as np
from models import SimpleCNN
from data_loader import test_loader  # 仅导入测试集加载器

# 独立定义CIFAR-10类别列表（无需依赖data_loader.py）
classes = [
    'airplane',    # 0
    'automobile',  # 1
    'bird',        # 2
    'cat',         # 3
    'deer',        # 4
    'dog',         # 5
    'frog',        # 6
    'horse',       # 7
    'ship',        # 8
    'truck'        # 9
]

# 验证测试集是否能正常加载
def test_data_loader():
    try:
        images, labels = next(iter(test_loader))
        print(f"测试集加载正常：批次大小={images.shape[0]}, 图像尺寸={images.shape[1:]}")
        return True
    except Exception as e:
        print(f"测试集加载失败：{e}")
        return False

# 模型评估主函数
def evaluate_model():
    # 先检查数据加载
    if not test_data_loader():
        return

    # 加载模型
    try:
        model = SimpleCNN(num_classes=10)
        model.load_state_dict(torch.load('./best_model_5epochs.pth'))
        model.eval()  # 切换到评估模式
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败：{e}")
        return

    # 计算准确率
    total_correct = 0
    total_samples = 0
    class_correct = [0] * 10  # 每个类别的正确数
    class_total = [0] * 10    # 每个类别的总数

    with torch.no_grad():  # 关闭梯度计算
        for batch_idx, (images, labels) in enumerate(test_loader):
            # 前向传播
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # 预测结果

            # 累计总准确率
            batch_size = labels.size(0)
            total_samples += batch_size
            total_correct += (predicted == labels).sum().item()

            # 累计每个类别的准确率
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

            # 打印进度
            if (batch_idx + 1) % 20 == 0:
                print(f"已处理 {batch_idx + 1}/{len(test_loader)} 批次")

    # 输出评估结果
    overall_acc = 100 * total_correct / total_samples
    print(f"\n测试集整体准确率：{overall_acc:.2f}%")
    print("-" * 60)
    print("每个类别的准确率：")
    for i in range(10):
        acc = 100 * class_correct[i] / class_total[i]
        print(f"{classes[i]:<10} {acc:.2f}%")

# 执行评估
if __name__ == '__main__':
    evaluate_model()