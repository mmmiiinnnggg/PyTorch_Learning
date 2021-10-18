# if torch.cuda.is_available():

# 网络模型 net = net.cuda()
# 数据 imgs = imgs.cuda()  targets = targets.cuda()
# loss = loss.cuda()

# 第二种方式： .to(device)
#  device = torch.device("cpu")
#  torch.device("cuda")
#  torch.device("cuda:1")
#  使用的时候: net.to(device)

import time

start_time = time.time()

end_time = time.time()

print(end_time-start_time)

# nvidia -smi

# 用gpu训练的模型 在cpu设备上加载的时候需要
# modle = torch.load("model_gpu.pth", map_location = torch.device('cpu')

# 运行代码的时候
# python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
# -- 表示参数名称 后面跟的值是赋值