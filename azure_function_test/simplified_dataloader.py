import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SimpleDataset(Dataset):
    def __init__(self, size=100):
        # 构造一个简单的数据集，包含100个样本
        self.size = size

    def __len__(self):
        # 数据集中样本的数量
        return self.size

    def __getitem__(self, idx):
        # 每个样本是一个随机生成的数据和标签
        data = torch.randn(3, 224, 224)  # 假设每个样本是一个3x224x224的图像
        label = torch.randint(0, 10, (1,))  # 假设标签是0到9之间的整数
        return data, label

# 创建数据集实例
simple_dataset = SimpleDataset(size=100)

# 创建一个DataLoader，使用两个工作进程来加载数据
simple_dataloader = DataLoader(simple_dataset, batch_size=10, shuffle=True, num_workers=2)

# 遍历DataLoader中的数据
try:
    for i, (data, label) in enumerate(simple_dataloader):
        print(f"Batch {i}, Data shape: {data.shape}, Label shape: {label.shape}")
        if i == 2:  # 为了测试，只遍历前三个批次
            break
except Exception as e:
    print(f"An error occurred: {str(e)}")
