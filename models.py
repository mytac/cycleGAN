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
    def __init__(self, input_shape, n_residual_blocks=9):
        super(Generator,self).__init__()

        channels = input_shape[0]
        out_channel=64

        net=[
             nn.ReflectionPad2d(channels), # 7*7卷积核
             nn.Conv2d(channels,out_channel,7), #原始图片3,输出64，卷积核7
             nn.InstanceNorm2d(out_channel), #输出为64
             nn.ReLU(inplace=True)
        ]

        in_channel=out_channel

        in_channel=64
        # 下采样：每次下采样后channel数量加倍
        for _ in range(2):
            out_channel*=2
            net +=[
              nn.Conv2d(in_channel,out_channel,3,stride=2,padding=1), #原始图片3,输出64，卷积核7
              nn.InstanceNorm2d(out_channel), #输出为64
              nn.ReLU(inplace=True)
            ]
            in_channel=out_channel
        
        for _ in range(n_residual_blocks):
            net+=[resBlock(in_channel)]

        # 上采样：反卷积
        for _ in range(2):
            out_channel//=2
            net +=[
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channel,out_channel,3,
                                      stride=1,
                                      padding=1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True)]
            in_channel=out_channel


        # 输出层
        net +=[
              nn.ReflectionPad2d(channels),
              nn.Conv2d(out_channel,channels,7),
              nn.Tanh()
          ]
        
        self.model=nn.Sequential(*net) #串联 
      
    def forward(self,x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator,self).__init__()

        channels, height, width = input_shape

        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)    

        def discriminator_block(in_filters, out_filters, normalize=True):           ## 鉴别器块儿
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]   ## layer += [conv + norm + relu]    
            if normalize:                                                           ## 每次卷积尺寸会缩小一半，共卷积了4次
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        #下采样
        self.model = nn.Sequential(                                                 
            *discriminator_block(channels, 64, normalize=False),        ## layer += [conv(3, 64) + relu]
            *discriminator_block(64, 128),                              ## layer += [conv(64, 128) + norm + relu]
            *discriminator_block(128, 256),                             ## layer += [conv(128, 256) + norm + relu]
            *discriminator_block(256, 512),                             ## layer += [conv(256, 512) + norm + relu]
            nn.ZeroPad2d((1, 0, 1, 0)),                                 ## layer += [pad]
            nn.Conv2d(512, 1, 4, padding=1)                             ## layer += [conv(512, 1)]
        )


    def forward(self,x):
        return self.model(x)
        

# if __name__=="__main__":
#     G=Generator()
#     D=Discriminator()

#     import torch
#     input_tensor=torch.ones((1,3,256,256),dtype=torch.float)
#     out=G(input_tensor)
#     print(out.size())

#     out=D(input_tensor)
#     print(out.size())