import torch
import torch.nn as nn
import torch.optim as optim
import os
from models import SimpleCNN
from data_loader import train_loader, val_loader

# --------------------------
# 超参数设置（固定5轮训练）
# --------------------------
num_epochs = 5  # 固定训练5轮
batch_size = 64  # 批处理大小（与data_loader一致）
learning_rate = 0.001  # 学习率（根据你的收敛速度，保持当前值）
momentum = 0.9  # SGD动量
weight_decay = 1e-4  # L2正则化
save_path = './best_model_5epochs.pth'  # 最佳模型保存路径

# --------------------------
# 初始化组件
# --------------------------
model = SimpleCNN(num_classes=10)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum=momentum,
    weight_decay=weight_decay
)

best_val_acc = 0.0  # 记录5轮中最好的验证准确率


# --------------------------
# 5轮训练循环
# --------------------------
def train():
    global best_val_acc
    print(f"开始训练，共{num_epochs}轮...\n")

    for epoch in range(num_epochs):
        # --------------------------
        # 训练阶段
        # --------------------------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()  # 清零梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            # 累计训练损失和准确率
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # 计算训练集平均指标
        train_avg_loss = train_loss / len(train_loader.dataset)
        train_acc = 100 * train_correct / train_total

        # --------------------------
        # 验证阶段
        # --------------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():  # 关闭梯度计算
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # 计算验证集平均指标
        val_avg_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * val_correct / val_total

        # --------------------------
        # 输出日志并保存最佳模型
        # --------------------------
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {train_avg_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss:   {val_avg_loss:.4f} | Val Acc:   {val_acc:.2f}%')
        print('-' * 50)

        # 保存5轮中验证准确率最高的模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f'已保存当前最佳模型（准确率：{best_val_acc:.2f}%）\n')

    print(f'5轮训练完成！最佳验证准确率：{best_val_acc:.2f}%')


# --------------------------
# 执行训练
# --------------------------
if __name__ == '__main__':
    # 创建保存目录（如果需要）
    if not os.path.exists(os.path.dirname(save_path)) and os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path))
    train()