# Tensorboard可以可视化展示训练过程

from torch.utils.tensorboard import SummaryWriter
import numpy as np

from PIL import Image

writer = SummaryWriter("logs") # log - 事件
image_path = "dataset/train/ants_image/0013035.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

writer.add_image("test", img_array, 4, dataformats = 'HWC')

# for i in range(100):
#    writer.add_scalar("y=2x", 2*i, i)

writer.close()

# add_scalar:
# Args:
# tag(string): Data identifier
# scalar_value(float or string / blobname): Value to save 迭代值
# global_step(int): Global step value to record 迭代的次数 step几

# tensorboard --logdir=logs

# 如果不想在一个board上显示多个过程，那么就需要删掉logs文件，重启浏览器

