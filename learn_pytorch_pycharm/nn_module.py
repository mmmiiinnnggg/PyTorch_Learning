import torch
from torch import nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input +1

net = Network()
x = torch.tensor(1.0)

output = net(x)

print(output)

# if __name__ == '__main__':

# 这个的作用是 当这个py文件作为模块在其他py中被调用的时候，不会运行这个程序

# 每个python模块（python文件，也就是此处的 test.py 和 import_test.py）都包含内置的变量 __name__，
# 当该模块被直接执行的时候，__name__ 等于文件名（包含后缀 .py ）；如果该模块 import 到其他模块中，则该模块的 __name__ 等于模块名称（不包含后缀.py）。
# 而 “__main__” 始终指当前执行模块的名称（包含后缀.py）。进而当模块被直接执行时，__name__ == 'main' 结果为真。

# https://blog.csdn.net/heqiang525/article/details/89879056


# net.train() net.eval() - 如果有特殊的层 比如batchnorm, dropout 那么需要这个mode

# 测试的时候 需要 with torch.no_grad():

