import torch.nn as nn
import torch.nn.functional as F


class resBlock(nn.Module):
    def __init__(self,in_channel):
        super(resBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),  # 避免卷积后生成图像的损失,第一个卷积3*3
            nn.Conv2d(in_channel,in_channel,3),
            nn.InstanceNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),  # 避免卷积后生成图像的损失,第一个卷积3*3
            nn.Conv2d(in_channel,in_channel,3),
            nn.InstanceNorm2d(in_channel),
        ]

        self.conv_block=nn.Sequential(*conv_block) #串联

    def forward(self,x):
        return x+self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9):
        super(Generator,self).__init__()
        net=[
             nn.ReflectionPad2d(3), # 7*7卷积核
             nn.Conv2d(input_nc,64,7), #原始图片3,输出64，卷积核7
             nn.InstanceNorm2d(64), #输出为64
             nn.ReLU(inplace=True)
        ]

        # 下采样：每次下采样后channel数量加倍
        in_channel=64
        out_channel=in_channel*2

        for _ in range(2):
            net +=[
              nn.Conv2d(in_channel,out_channel,3,stride=2,padding=1), #原始图片3,输出64，卷积核7
              nn.InstanceNorm2d(out_channel), #输出为64
              nn.ReLU(inplace=True)
            ]
            in_channel=out_channel
            out_channel=in_channel*2
        
        for _ in range(9):
            net+=[resBlock(in_channel)]

        # 上采样：反卷积
        out_channel=in_channel//2
        for _ in range(2):
            net +=[nn.ConvTranspose2d(in_channel,out_channel,3,
                                      stride=2,
                                      padding=1,
                                      output_padding=1),
                    nn.InstanceNorm2d(out_channel),
                    nn.ReLU(inplace=True)]
            in_channel=out_channel
            out_channel=in_channel//2


        # 输出层
        net +=[
              nn.ReflectionPad2d(3),
              nn.Conv2d(64,output_nc,7),
              nn.Tanh()
          ]
        
        self.model=nn.Sequential(*net) #串联 
      
    def forward(self,x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super(Discriminator,self).__init__()

        #下采样
        model=[nn.Conv2d(input_nc,64,4,stride=2,padding=1),
               nn.LeakyReLU(0.2,inplace=True)]
        model+=[nn.Conv2d(64,128,4,stride=2,padding=1),
              nn.InstanceNorm2d(128), 
               nn.LeakyReLU(0.2,inplace=True)]
        model+=[nn.Conv2d(128,256,4,stride=2,padding=1),
                 nn.InstanceNorm2d(256), 
               nn.LeakyReLU(0.2,inplace=True)]
        model+=[nn.Conv2d(256,512,4,stride=2,padding=1),
                 nn.InstanceNorm2d(512), 
               nn.LeakyReLU(0.2,inplace=True)]
        model+=[nn.Conv2d(512,1,4,padding=1)]
        self.model=nn.Sequential(*model)

    def forward(self,x):
        x=self.model(x)
        return F.avg_pool2d(x,x.size()[2:]).view(x.size()[0],-1)
        

# if __name__=="__main__":
#     G=Generator()
#     D=Discriminator()

#     import torch
#     input_tensor=torch.ones((1,3,256,256),dtype=torch.float)
#     out=G(input_tensor)
#     print(out.size())

#     out=D(input_tensor)
#     print(out.size())