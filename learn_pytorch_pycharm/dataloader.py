# dataloader - 取数据到神经网络的方式
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
# drop_last - 除不尽batch_size的时候剩下的怎么办
# 采取随机采样 而不是从数据集开头开始采集
writer = SummaryWriter("data_loader")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step = step + 1

writer.close()
