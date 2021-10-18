import torch
from torch import nn
from torch.nn import ReLU

input = torch.tensor([
    [1, -0.5],
    [-1, -3]
])

input = torch.reshape(input, (-1, 1, 2, 2))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu1 = ReLU()

    def forward(self, x):
        return self.relu1(x)

net = Net()
output = net(input)
print(output)
