import torch
import torchvision.datasets

# train_data = torchvision.datasets.ImageNet("./data_imagenet", split = 'train', download= True, transform=torchvision.transforms.ToTensor())
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)


# # save model and load model - way 1
# torch.save(vgg16_false, "vgg16_false.pth")
#
# model = torch.load("vgg16_false.pth")
# 如果是自己的模型，那么需要重写或者加载一编网络的结构

#
# # save model and load model - way 2
# torch.save(vgg16_false.state_dict(), "vgg16_false_2.pth") #将参数保存成字典的形式 更为推荐
# vgg16_false.load_state_dict(torch.load("vgg16_false_2.pth"))


vgg16_true = torchvision.models.vgg16(pretrained=True) # 528M

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10("./dataset", train=True, transform= torchvision.transforms.ToTensor(),download=True)

# transfer learning

vgg16_true.classifier.add_module('add_linear', nn.Linear(1000,10))
print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096,10)
print(vgg16_false)

