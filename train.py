import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
from models import Discriminator, Generator
from utils import ReplayBuffer, LambdaLR, weights_init_normal
from datasets import ImageDataset
from torchvision.utils import save_image, make_grid
import itertools
import time
from torch.autograd import Variable
import sys
import tensorboardX
import datetime
import os
import torch
print(torch.cuda.is_available())#是否有可用的gpu
print(torch.cuda.device_count())#有几个可用的gpu
print(torch.cuda.current_device())#可用gpu编号
print( torch.cuda.get_device_capability(device=None),  torch.cuda.get_device_name(device=None))#可用gpu内存大小，可用gpu的名字
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#声明gpu
dev=torch.device('cuda:0')#调用哪个gpu
a=torch.rand(100,100).to(dev)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1
size = 256
lr = 0.0002
n_epoch = 50
epoch = 0
decay_epoch = 25
sample_interval=100

img_height=256
img_width=256
input_shape = (3, img_height, img_width)
betas=(0.5,0.9999)


#loss
criterion_GAN=torch.nn.MSELoss()
criterion_cycle =torch.nn.L1Loss()
criterion_identity =torch.nn.L1Loss() #相似性loss 

# networks
netG_A2B = Generator(input_shape).to(device)
netG_B2A = Generator(input_shape).to(device)
netD_A = Discriminator(input_shape).to(device)
netD_B = Discriminator(input_shape).to(device)



#optimizer & LR
opt_G=torch.optim.Adam(itertools.chain(netG_A2B.parameters(),netG_B2A.parameters()),lr=lr,betas=betas) #连接两个网络的参数
opt_DA=torch.optim.Adam(netD_A.parameters() ,lr=lr,betas=betas) 
opt_DB=torch.optim.Adam(netD_B.parameters() ,lr=lr,betas=betas) 

# 学习率更新进程
lr_scheduler_G=torch.optim.lr_scheduler.LambdaLR(opt_G,lr_lambda=LambdaLR(n_epoch,epoch,decay_epoch).step)
lr_scheduler_DA=torch.optim.lr_scheduler.LambdaLR(opt_DA,lr_lambda=LambdaLR(n_epoch,epoch,decay_epoch).step)
lr_scheduler_DB=torch.optim.lr_scheduler.LambdaLR(opt_DB,lr_lambda=LambdaLR(n_epoch,epoch,decay_epoch).step)

#train
data_root="dataset/apple2orange"
input_A=torch.ones([batch_size,3,size,size],dtype=torch.float).to(device)
input_B=torch.ones([batch_size,3,size,size],dtype=torch.float).to(device)

## 定义buffer
fake_A_buffer=ReplayBuffer()
fake_B_buffer=ReplayBuffer()

# log
log_path="logs"
writer_log=tensorboardX.SummaryWriter(log_path)

transforms_=[
    transforms.Resize(int(img_height*1.12)),
    transforms.RandomCrop(img_height,img_width), #crop
    transforms.RandomHorizontalFlip(), #图像翻转
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) #正则
]

#dataloader
dataloader=DataLoader(ImageDataset(data_root,transform=transforms_),
                      batch_size=batch_size,shuffle=True,num_workers=0)

## Test data loader
val_dataloader = DataLoader(
    ImageDataset(data_root, transform=transforms_,  model="test"), 
    batch_size=5,
    shuffle=True,
    num_workers=0,
)

## 每间隔100次打印图片
def sample_images(batches_done):      ## （100/200/300/400...）
    """保存测试集中生成的样本"""
    imgs = next(iter(val_dataloader))      ## 取一张图像 
    netG_A2B.eval()
    netG_B2A.eval()
    real_A = Variable(imgs["A"]).cuda()    ## 取一张真A
    fake_B = netG_A2B(real_A)                  ## 用真A生成假B
    real_B = Variable(imgs["B"]).cuda()    ## 去一张真B
    fake_A = netG_B2A(real_B)                  ## 用真B生成假A
    # Arange images along x-axis
    ## make_grid():用于把几个图像按照网格排列的方式绘制出来
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arange images along y-axis
    ## 把以上图像都拼接起来，保存为一张大图片
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "images/%s/%s.png" % ("apple2orange", batches_done), normalize=False)

step=0
prev_time = time.time()                             ## 开始时间
for epo in range(epoch,n_epoch):

    for i,batch in enumerate(dataloader):
        #真实图像
        real_A=torch.tensor(input_A.copy_(batch['A']),dtype=torch.float).to(device)
        real_B=torch.tensor(input_B.copy_(batch['B']),dtype=torch.float).to(device)
        # real_A = Variable(batch["A"]).cuda()  ## 真图像A
        # real_B = Variable(batch["B"]).cuda()  ## 真图像B


        ## label
        # label_real=torch.ones([1],requires_grad=False,dtype=torch.float).to(device) #真实的样本定义为1
        # label_fake=torch.zeros([1],requires_grad=False,dtype=torch.float).to(device) #假的样本定义为0
        label_real = Variable(torch.ones((real_A.size(0), *netD_A.output_shape)), requires_grad=False).cuda()     ## 定义真实的图片label为1 ones((1, 1, 16, 16))
        label_fake = Variable(torch.zeros((real_A.size(0), *netD_A.output_shape)), requires_grad=False).cuda()     ## 定义假的图片的label为0 zeros((1, 1, 16, 16))

        netG_A2B.train()
        netG_B2A.train()


        same_B=netG_A2B(real_B) #利用A2B生成器生成B，查看与真实B的偏差，loss越小越好
        loss_identity_B=criterion_identity(same_B,real_B)*5.0

        same_A=netG_B2A(real_A)
        loss_identity_A=criterion_identity(same_A,real_A)*5.0
        loss_identity = (loss_identity_A + loss_identity_B) / 2    

        #真A生成假的B,用判别器预测结果 GAN_LOSS
        fake_B=netG_A2B(real_A)
        pred_fake=netD_B(fake_B)
        loss_GAN_A2B=criterion_GAN(pred_fake,label_real)

         #真b生成假的A,用判别器预测结果
        fake_A=netG_B2A(real_B)
        pred_fake=netD_A(fake_A)
        loss_GAN_B2A=criterion_GAN(pred_fake,label_real)

        loss_GAN = (loss_GAN_A2B + loss_GAN_B2A) / 2


        #cycle loss：需要保证cycle的一致性：利用生成的假的A和假的B，利用生成器恢复真的A\B，保证复原的结果与原始结果一致
        recovered_A=netG_B2A(fake_B)
        loss_cycle_ABA=criterion_cycle(recovered_A,real_A)
        recovered_B=netG_A2B(fake_A)
        loss_cycle_BAB=criterion_cycle(recovered_B,real_B)
        loss_cycle = (loss_cycle_ABA + loss_cycle_BAB) / 2

        #计算生成器整体loss:所有loss相加
        # loss_G=loss_identity_A+loss_identity_B+loss_GAN_A2B+loss_GAN_B2A+loss_cycle_ABA+loss_cycle_BAB
        loss_G = loss_GAN + 10 * loss_cycle + 5.0 * loss_identity
        opt_G.zero_grad() #梯度先置零

        #定义好生成器后，对生成器进行反向传播
        loss_G.backward()
        opt_G.step()


        ######################################
        ############ 判别器  A   ######
        ######################################

        #定义判别器A的loss - 真的图像判别为真
        pred_real=netD_A(real_A)
        loss_D_real=criterion_GAN(pred_real,label_real)

        fake_A=fake_A_buffer.push_and_pop(fake_A)
        pred_fake=netD_A(fake_A.detach()) #对于生成数据，判别器预测结果 防止判别器更新时对生成器的梯度进行计算，进行detach
        loss_D_fake=criterion_GAN(pred_fake,label_fake) #判别器对生成数据判别为负样本

        #total loss
        loss_D_A=(loss_D_real+loss_D_fake)*0.5
        opt_DA.zero_grad()
        loss_D_A.backward()
        opt_DA.step()


        ######################################
        ############ 判别器  B   ######
        ######################################

        #定义判别器B的loss
        pred_real=netD_B(real_B)
        loss_D_real=criterion_GAN(pred_real,label_real)

        fake_B=fake_B_buffer.push_and_pop(fake_B)
        pred_fake=netD_B(fake_B.detach()) #对于生成数据，判别器预测结果
        loss_D_fake=criterion_GAN(pred_fake,label_fake) #判别器对生成数据判别为负样本

        #total loss
        loss_D_B=(loss_D_real+loss_D_fake)*0.5
        opt_DB.zero_grad()
        loss_D_B.backward()
        opt_DB.step()

        loss_D = (loss_D_A + loss_D_B) / 2


         ## 确定剩下的大约时间  假设当前 epoch = 5， i = 100
        batches_done = epoch * len(dataloader) + i                                        ## 已经训练了多长时间 5 * 400 + 100 次
        batches_left = n_epoch * len(dataloader) - batches_done                      ## 还剩下 50 * 400 - 2100 次
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))  ## 还需要的时间 time_left = 剩下的次数 * 每次的时间
        prev_time = time.time()

         # Print log
        sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                % (
                    epo,
                    n_epoch,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_identity.item(),
                    time_left,
                )
            )
        # 每训练100张就保存一组测试集中的图片
        if batches_done % sample_interval == 0:
                sample_images(batches_done)


      
        # print("loss_G:{},loss_G_identity:{},loss_G_GAN:{},"
        #       "loss_G_cycle:{},loss_D_A:{},loss_D_B:{},epoch:{}".format(loss_G,loss_identity,loss_GAN,
        #                                                        loss_cycle,
        #                                                        loss_D,epoch) )
        # writer_log.add_scalar("loss_G",loss_G,global_step=step+1)
        # writer_log.add_scalar("loss_G_identity",loss_identity,global_step=step+1)
        # writer_log.add_scalar("loss_G_GAN",loss_GAN_A2B+loss_GAN_B2A,global_step=step+1)
        # writer_log.add_scalar("loss_G_cycle",loss_GAN,global_step=step+1)
        # writer_log.add_scalar("loss_D",loss_cycle,global_step=step+1)

        # step+=1
    #训练完一次epoch，更新学习率
    lr_scheduler_G.step()
    lr_scheduler_DA.step()
    lr_scheduler_DB.step()
    #保存模型
    torch.save(netG_A2B.state_dict(),"models/netG_A2B.pth")
    torch.save(netG_B2A.state_dict(),"models/netG_B2A.pth")
    torch.save(netD_A.state_dict(),"models/netD_A.pth")
    torch.save(netD_B.state_dict(),"models/netD_B.pth")