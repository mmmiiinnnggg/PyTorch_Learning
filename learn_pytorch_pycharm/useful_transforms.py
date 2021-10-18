from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")

img = Image.open("dataset/train/ants_image/0013035.jpg")

# ToTensor 的使用
trans = transforms.ToTensor()
img_tensor = trans(img)
writer.add_image("ToTensor",img_tensor)


# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]) # channels数和mean/std
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm)

# Resize
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
img_resize = trans(img_resize)

writer.add_image("Resize", img_resize, 0)
# 或者可以用compose
print(img_resize)

#  Compose - 结合多个transforms
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans])
img_resize_2 = trans_compose(img)
writer.add_image("Compose", img_resize_2, 1)


# RandomCrop
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random,trans])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop",img_crop,i)

writer.close()

