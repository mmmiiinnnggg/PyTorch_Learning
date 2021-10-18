# CTRL+P 查看有哪些参数可用

import torchvision
from torch.utils.tensorboard import SummaryWriter

data_transform = torchvision.transforms.ToTensor()

# root - 下载存放位置 train - 训练集还是测试集 transform - 是否有变换 download - 是否需要下载
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=data_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=data_transform, download=True)

# img, label = test_set[0]

writer = SummaryWriter("p10")
for i in range(10):
    img, label = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()
