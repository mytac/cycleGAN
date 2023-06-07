import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
from models import Discriminator, Generator
from utils import ReplayBuffer, LambdaLR, weights_init_normal
from datasets import ImageDataset
import itertools
import tensorboardX

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1
size = 256
lr = 0.0002
n_epoch = 200
epoch = 0
decay_epoch = 100

# networks
netG_A2B = Generator().to(device)
netG_B2A = Generator().to(device)
netD_A = Discriminator().to(device)
netD_B = Discriminator().to(device)

#loss
loss_GAN=torch.nn.MSELoss()
loss_cycle=torch.nn.L1Loss()
loss_identity=torch.nn.L1Loss() #相似性loss 

#optimizer & LR
opt_G=torch.optim.Adam(itertools.chain(netG_A2B.parameters(),netG_B2A.parameters()),lr=lr,betas=(0.5,0.9999)) #连接两个网络的参数

opt_DA=torch.optim.Adam(netD_A.parameters() ,lr=lr,betas=(0.5,0.9999)) 
opt_DB=torch.optim.Adam(netD_B.parameters() ,lr=lr,betas=(0.5,0.9999)) 

lr_scheduler_G=torch.optim.lr_scheduler.LambdaLR(opt_G,lr_lambda=LambdaLR(n_epoch,epoch,decay_epoch).step)
lr_scheduler_DA=torch.optim.lr_scheduler.LambdaLR(opt_DA,lr_lambda=LambdaLR(n_epoch,epoch,decay_epoch).step)
lr_scheduler_DB=torch.optim.lr_scheduler.LambdaLR(opt_DB,lr_lambda=LambdaLR(n_epoch,epoch,decay_epoch).step)

#train
data_root="dataset/apple2orange"
input_A=torch.ones([1,3,size,size],dtype=torch.float).to(device)
input_B=torch.ones([1,3,size,size],dtype=torch.float).to(device)
## label
label_real=torch.ones([1],requires_grad=False,dtype=torch.float).to(device) #真实的样本定义为1
label_fake=torch.zeros([1],requires_grad=False,dtype=torch.float).to(device) #假的样本定义为0
## 定义buffer‘
fake_A_buffer=ReplayBuffer()
fake_B_buffer=ReplayBuffer()

# log
log_path="logs"
writer_log=tensorboardX.SummaryWriter(log_path)

transforms_=[
    transforms.Resize(int(256*1.12),Image.BICUBIC),
    transforms.RandomCrop(256), #crop
    transforms.RandomHorizontalFlip(), #图像翻转
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
]

#dataloader
dataloader=DataLoader(ImageDataset(data_root,transforms_),batch_size=batch_size,shuffle=True,num_workers=0)
step=0
for epoch in range(n_epoch):
    for i,batch in enumerate(dataloader):
        real_A=torch.tensor(input_A.copy_(batch['A']),dtype=torch.float).to(device)
        real_B=torch.tensor(input_B.copy_(batch['B']),dtype=torch.float).to(device)

        opt_G.zero_grad() #梯度先置零

        same_B=netG_A2B(real_B) #利用A2B生成器生成B，查看与真实B的偏差，loss越小越好
        loss_identity_B=loss_identity(same_B,real_B)*5.0

        same_A=netG_B2A(real_A)
        loss_identity_A=loss_identity(same_A,real_A)*5.0

        #真A生成假的B,用判别器预测结果 GAN_LOSS
        fake_B=netG_A2B(real_A)
        pred_fake=netD_B(fake_B)
        loss_GAN_A2B=loss_GAN(pred_fake,label_real)

         #真b生成假的A,用判别器预测结果
        fake_A=netG_B2A(real_B)
        pred_fake=netD_A(fake_A)
        loss_GAN_B2A=loss_GAN(pred_fake,label_real)

        #cycle loss：需要保证cycle的一致性：利用生成的假的A和假的B，利用生成器恢复真的A\B，保证复原的结果与原始结果一致
        recovered_A=netG_B2A(fake_B)
        loss_cycle_ABA=loss_cycle(recovered_A,real_A)*10.0
        recovered_B=netG_A2B(fake_A)
        loss_cycle_BAB=loss_cycle(recovered_B,real_B)*10.0

        #计算生成器整体loss:所有loss相加
        loss_G=loss_identity_A+loss_identity_B+loss_GAN_A2B+loss_GAN_B2A+loss_cycle_ABA+loss_cycle_BAB

        #定义好生成器后，对生成器进行反向传播
        loss_G.backward()
        opt_G.step()


        ######################################
        opt_DA.zero_grad()

        #定义判别器A的loss
        pred_real=netD_A(real_A)
        loss_D_real=loss_GAN(pred_real,label_real)

        fake_A=fake_A_buffer.push_and_pop(fake_A)
        pred_fake=netD_A(fake_A.detach()) #对于生成数据，判别器预测结果 防止判别器更新时对生成器的梯度进行计算，进行detach
        # TODO 下面的 pred_real e应该是 pred_fake吧？
        loss_D_fake=loss_GAN(pred_fake,label_fake) #判别器对生成数据判别为负样本

        #total loss
        loss_D_A=(loss_D_real+loss_D_fake)*0.5
        loss_D_A.backward()
        opt_DA.step()



        #定义判别器B的loss
        opt_DB.zero_grad()
        pred_real=netD_B(real_B)
        loss_D_real=loss_GAN(pred_real,label_real)

        fake_B=fake_B_buffer.push_and_pop(fake_B)
        pred_fake=netD_B(fake_B.detach()) #对于生成数据，判别器预测结果
        # TODO 下面的 pred_real e应该是 pred_fake吧？
        loss_D_fake=loss_GAN(pred_real,label_fake) #判别器对生成数据判别为负样本

        #total loss
        loss_D_B=(loss_D_real+loss_D_fake)*0.5
        loss_D_B.backward()
        opt_DB.step()

        print("loss_G:{},loss_G_identity:{},loss_G_GAN:{},"
              "loss_G_cycle:{},loss_D_A:{},loss_D_B:{},epoch:{}".format(loss_G,loss_identity_A+loss_identity_B,loss_GAN_A2B+loss_GAN_B2A,
                                                               loss_cycle_ABA+loss_cycle_BAB,
                                                               loss_D_A,loss_D_B,epoch) )
        writer_log.add_scalar("loss_G",loss_G,global_step=step+1)
        writer_log.add_scalar("loss_G_identity",loss_identity_A+loss_identity_B,global_step=step+1)
        writer_log.add_scalar("loss_G_GAN",loss_GAN_A2B+loss_GAN_B2A,global_step=step+1)
        writer_log.add_scalar("loss_G_cycle",loss_cycle_ABA+loss_cycle_BAB,global_step=step+1)
        writer_log.add_scalar("loss_D_A",loss_D_A,global_step=step+1)
        writer_log.add_scalar("loss_D_B",loss_D_B,global_step=step+1)

        step+=1
    #训练完一次epoch，更新学习率
    lr_scheduler_G.step()
    lr_scheduler_DA.step()
    lr_scheduler_DB.step()
    #保存模型
    torch.save(netG_A2B.state_dict(),"models/netG_A2B.pth")
    torch.save(netG_B2A.state_dict(),"models/netG_B2A.pth")
    torch.save(netD_A.state_dict(),"models/netD_A.pth")
    torch.save(netD_B.state_dict(),"models/netD_B.pth")