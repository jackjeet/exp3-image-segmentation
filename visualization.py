import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from models import SimpleCNN
from data_loader import test_loader
import os

# --------------------------
# 简化字体配置（避免冗余警告）
# --------------------------
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 只保留Windows系统默认中文字体（黑体）
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.switch_backend('agg')  # 非交互式后端，避免绘图错误

# 类别名称
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# 创建保存文件夹
os.makedirs('./visualizations', exist_ok=True)

# --------------------------
# 训练数据（你的5轮结果）
# --------------------------
train_losses = np.array([1.5254, 1.1418, 0.9984, 0.9025, 0.8324])
val_losses = np.array([1.2141, 1.0336, 0.9328, 0.9033, 0.8357])
train_accs = np.array([44.83, 59.33, 64.51, 68.19, 70.62])
val_accs = np.array([57.44, 62.52, 67.20, 67.84, 70.60])


# --------------------------
# 1. 损失与准确率曲线
# --------------------------
def plot_loss_acc():
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='训练损失')
    plt.plot(epochs, val_losses, 'ro-', label='验证损失')
    plt.title('训练与验证损失')
    plt.xlabel('轮次（Epoch）')
    plt.ylabel('损失值')
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'bo-', label='训练准确率')
    plt.plot(epochs, val_accs, 'ro-', label='验证准确率')
    plt.title('训练与验证准确率')
    plt.xlabel('轮次（Epoch）')
    plt.ylabel('准确率（%）')
    plt.legend()

    plt.tight_layout()
    plt.savefig('./visualizations/loss_acc_curve.png', dpi=150)  # 提高分辨率
    print("✅ 损失与准确率曲线已保存")


# --------------------------
# 2. 混淆矩阵
# --------------------------
def plot_confusion_matrix():
    model = SimpleCNN(num_classes=10)
    model.load_state_dict(torch.load('./best_model_5epochs.pth'))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                annot_kws={'fontsize': 8})  # 调整标注字体大小
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title('测试集混淆矩阵')
    plt.tight_layout()
    plt.savefig('./visualizations/confusion_matrix.png', dpi=150)
    print("✅ 混淆矩阵已保存")


# --------------------------
# 3. 错误分类示例
# --------------------------
def plot_misclassified_examples(num_examples=5):
    model = SimpleCNN(num_classes=10)
    model.load_state_dict(torch.load('./best_model_5epochs.pth'))
    model.eval()

    misclassified_images = []
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            misclassified_idx = (preds != labels).nonzero()
            if misclassified_idx.numel() == 0:
                continue
            # 处理批量维度（兼容单样本和多样本）
            misclassified_idx = misclassified_idx.squeeze()
            if isinstance(misclassified_idx, int):
                misclassified_idx = [misclassified_idx]
            for idx in misclassified_idx:
                misclassified_images.append(images[idx].numpy())
                true_labels.append(labels[idx].item())
                pred_labels.append(preds[idx].item())
                if len(misclassified_images) >= num_examples:
                    break
            if len(misclassified_images) >= num_examples:
                break

    plt.figure(figsize=(15, 3))
    for i in range(num_examples):
        img = misclassified_images[i].transpose(1, 2, 0)
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        plt.subplot(1, num_examples, i + 1)
        plt.imshow(img)
        plt.title(f'真实: {classes[true_labels[i]]}\n预测: {classes[pred_labels[i]]}', fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('./visualizations/misclassified_examples.png', dpi=150)
    print("✅ 错误分类示例已保存")


# --------------------------
# 执行所有可视化
# --------------------------
if __name__ == '__main__':
    plot_loss_acc()
    plot_confusion_matrix()
    plot_misclassified_examples()
    print("\n🎉 所有可视化结果已保存至 './visualizations' 文件夹！")