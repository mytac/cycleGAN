# GAN和CycleGAN拼接
import random
import torch
import numpy as np
from torch.autograd import Variable


def tensor2image(tensor): # 测试时用
    image=127.5*(tensor[0].cpu().float().numpy()+1.0)
    if image.shape[0]==1:
        image=np.tile(image,(3,1,1))
    return image.astype(np.uint8)

# 训练模型时,cycle需要拿到生成的数据，利用这些数据，作为判别器的输入
# 为了保证训练稳定，在抽取生成数据时，并没有从当前的输入数据直接使用，而是从已经生成好的数据放到队列中，以队列的形式作为判别器的输入
class ReplayBuffer(): 
    def __init__(self,max_size=50):
        assert(max_size>0),'empty buffer or trying to create a black hole'
        self.max_size=max_size
        self.data=[]

    def push_and_pop(self,data):
        to_return=[]
        for element in data.data:
            element=torch.unsqueeze(element,0)
            if len(self.data)<self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1)>0.5: #大于maxsize的数据，以随机的0.5的概率，作为随机参数放入到to_return中
                  i=random.randint(0,self.max_size-1)
                  to_return.append(self.data[i].clone())
                  self.data[i]=element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return)) #拼接成最终的输入


# 学习率衰减:根据输入的epoch和epoch总数对学习率衰减
class LambdaLR():
    def __init__(self,n_epochs,offset,decay_start_epoch):
        assert((n_epochs-decay_start_epoch)>0),"decay must start before the training session ends"
        self.n_epochs=n_epochs
        self.offset=offset
        self.decay_start_epoch=decay_start_epoch

    def step(self,epoch):
        return 1.0-max(0,epoch+self.offset-self.decay_start_epoch)/(self.n_epochs-self.decay_start_epoch)
                


## 定义参数初始化函数
def weights_init_normal(m):                                    
    classname = m.__class__.__name__                        ## m作为一个形参，原则上可以传递很多的内容, 为了实现多实参传递，每一个moudle要给出自己的name. 所以这句话就是返回m的名字. 
    if classname.find("Conv") != -1:                        ## find():实现查找classname中是否含有Conv字符，没有返回-1；有返回0.
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)     ## m.weight.data表示需要初始化的权重。nn.init.normal_():表示随机初始化采用正态分布，均值为0，标准差为0.02.
        if hasattr(m, "bias") and m.bias is not None:       ## hasattr():用于判断m是否包含对应的属性bias, 以及bias属性是否不为空.
            torch.nn.init.constant_(m.bias.data, 0.0)       ## nn.init.constant_():表示将偏差定义为常量0.
    elif classname.find("BatchNorm2d") != -1:               ## find():实现查找classname中是否含有BatchNorm2d字符，没有返回-1；有返回0.
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)     ## m.weight.data表示需要初始化的权重. nn.init.normal_():表示随机初始化采用正态分布，均值为0，标准差为0.02.
        torch.nn.init.constant_(m.bias.data, 0.0)           ## nn.init.constant_():表示将偏差定义为常量0.
