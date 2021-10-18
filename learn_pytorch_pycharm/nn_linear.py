import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset", train = False, transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset, batch_size = 64,drop_last= True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = Linear(196608,10)

    def forward(self, x):
        return self.linear(x)

net = Net()


for data in dataloader:

    imgs, targets = data
    print(imgs.shape)

    output = torch.flatten(imgs)
    print(output.shape)

    output = net(output)
    print(output.shape)