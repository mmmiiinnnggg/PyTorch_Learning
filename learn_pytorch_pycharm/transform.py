from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# 图片转化工具包

img_path = "dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs1")

tensor_trans = transforms.ToTensor() # 初始化一个ToTensor对象
# ToTensor 可以将PIL和narray转化为tensor
tensor_img = tensor_trans(img)  # 自动调用 __call__方法

writer.add_image("Tensor_img",tensor_img)

print(tensor_img)

#
# PIL - Image.open()
# tensor - ToTensor()
# narrays - cv.imread()