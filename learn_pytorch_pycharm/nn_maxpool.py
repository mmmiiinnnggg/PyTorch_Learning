import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset", train = False, transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset, batch_size = 64)

# input = torch.tensor([
#     [1,2,0,3,1],
#     [0,1,2,3,1],
#     [1,2,1,0,0],
#     [5,2,3,1,1],
#     [2,1,0,1,1]
# ],dtype=torch.float32)

input = torch.reshape(input, (-1,1,5,5))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode= True)
        # 卷积核一开始的值是随机初始化的

    def forward(self, x):
        return self.maxpool1(x)

# net = Net()
# print(net(input))

# ../logs - 上级目录