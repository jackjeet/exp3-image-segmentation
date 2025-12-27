import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# --------------------------
# 数据预处理与增强
# --------------------------
# 训练集增强
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])

# 验证集和测试集预处理
val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])

# --------------------------
# 加载数据集并划分
# --------------------------
# 完整训练集（50000样本）
full_train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=train_transform
)

# 测试集（10000样本）
test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=val_test_transform
)

# 划分训练集（45000）和验证集（5000）
train_size = 45000
val_size = 5000
train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
val_dataset.dataset.transform = val_test_transform  # 验证集用非增强预处理

# --------------------------
# 数据加载器
# --------------------------
batch_size = 64
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)