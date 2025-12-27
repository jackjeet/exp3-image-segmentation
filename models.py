import torch
import torch.nn as nn
import torch.nn.functional as F

# 自定义简易卷积神经网络模型
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 第1个卷积块：卷积层 + 批归一化 + ReLU + 池化
        self.conv1 = nn.Conv2d(
            in_channels=3,      # 输入通道数：RGB图像为3
            out_channels=32,    # 输出通道数（卷积核数量）
            kernel_size=3,      # 3x3卷积核
            stride=1,           # 步长1
            padding=1           # 填充1（保持尺寸不变）
        )
        self.bn1 = nn.BatchNorm2d(32)  # 批归一化（加速训练收敛）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2最大池化（尺寸减半）

        # 第2个卷积块
        self.conv2 = nn.Conv2d(
            in_channels=32,     # 输入通道数=上一层输出32
            out_channels=64,    # 输出通道数64
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)

        # 第3个卷积块
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,   # 输出通道数128
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)

        # 全连接层（分类器）
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # 输入特征数：128通道 * 4x4尺寸（32经3次池化后）
        self.fc2 = nn.Linear(512, num_classes)   # 输出10类（CIFAR-10）
        self.dropout = nn.Dropout(0.5)  # 随机丢弃50%神经元，防止过拟合

    def forward(self, x):
        # 前向传播过程
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x32 → 16x16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16x16 → 8x8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8x8 → 4x4

        # 展平特征图：(batch_size, 128, 4, 4) → (batch_size, 128*4*4)
        x = x.view(-1, 128 * 4 * 4)

        # 全连接层计算
        x = F.relu(self.fc1(x))       # 第一层全连接+ReLU激活
        x = self.dropout(x)           # 应用dropout
        x = self.fc2(x)               # 输出层（未用softmax，损失函数会处理）
        return x


# 测试模型是否能正常运行（仅在直接运行该文件时执行）
if __name__ == '__main__':
    # 1. 初始化模型
    model = SimpleCNN(num_classes=10)  # CIFAR-10是10分类任务

    # 2. 生成模拟输入数据（批量大小64，3通道，32x32图像）
    test_input = torch.randn(64, 3, 32, 32)  # 模拟64张CIFAR-10格式的图片

    # 3. 执行前向传播
    output = model(test_input)

    # 4. 验证输出形状是否正确（应为 [64, 10]）
    print(f"模型输出形状：{output.shape}")
    print(f"预期形状：torch.Size([64, 10])")

    # 5. 验证模型参数是否可训练
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型可训练参数总数：{total_params}")  # 约330万参数，适合入门训练