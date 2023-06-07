import os
import glob  # 用于读取文件夹下的文件列表
import random
from torch.utils.data import Dataset

from PIL import Image  # 图像处理相关操作

import torchvision.transforms as trForms  # 数据增强


## 如果输入的数据集是灰度图像，将图片转化为rgb图像(本次采用的facades不需要这个)
def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root="", transform=None,model="train"):
        self.transform = trForms.Compose(transform) #合并transformer

        # root数据存放的根目录，model一个是train一个是test
        self.pathA = os.path.join(root, model+"A/*")
        self.pathB = os.path.join(root, model+"B/*")

        self.list_A = glob.glob(self.pathA)
        self.list_B = glob.glob(self.pathB)
    
    def __getitem__(self,index): #读取对应数据
        img_pathA=self.list_A[index % len(self.list_A)]
        img_pathB=random.choice(self.list_B) #随机选个数据

        #读取数据
        img_A=Image.open(img_pathA)
        img_B=Image.open(img_pathB)

        # 如果是灰度图，把灰度图转换为RGB图
        if img_A.mode != "RGB":
            img_A = to_rgb(img_A)
        if img_B.mode != "RGB":
            img_B = to_rgb(img_B)

        #数据预处理
        item_A=self.transform(img_A)
        item_B=self.transform(img_B)

        #返回预处理之后的数据
        return {"A":item_A,"B":item_B}
    
    def __len__(self): #取A和B列表中的最大值，作为数据集的长度
        return max(len(self.list_A),len(self.list_B))
    
# if __name__=='__main__':
#     from torch.utils.data import DataLoader

#     root="./dataset/apple2orange"

#     transform_=[trForms.Resize(256,Image.BILINEAR),trForms.ToTensor()] #BILINEAR作为差值
#     dataloader=DataLoader(ImageDataset(root,transform_,"train"),
#                           batch_size=1,
#                           shuffle=True,
#                               num_workers=1)
    
#     for i,batch in enumerate(dataloader):
#         print(i)
#         print(batch)
#         break;