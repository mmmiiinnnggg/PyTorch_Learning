import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset", train = False, transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset, batch_size = 64)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3 ,stride=1,padding=0)
        # 卷积核一开始的值是随机初始化的
        # conv1 = Conv2d() 类似于取出了Conv2d这个工具并设置了一些属性，之后conv1作为单独的函数使用

    def forward(self, x):
        self.conv1(x)
        return x

net = Net()

for data in dataloader:
    imgs, targets = data
    output = net(imgs)


