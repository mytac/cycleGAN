# CycleGAN

## 任务目标

让模型学会将苹果的图像转换成橙子的图像，或者将橙子的图像转换成苹果的图像，进行风格迁移。

## 数据集

[apple2orange 数据集](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/)是用于训练 CycleGAN 模型的数据集之一，它包含了苹果和橙子两种水果的图像，目标是让模型学会将苹果的图像转换成橙子的图像，或者将橙子的图像转换成苹果的图像。

## 网络结构

### 1. 生成器

CycleGAN 生成器模型的网络结构是一个由多个卷积层、反卷积层和残差块组成的深度神经网络。下面是一个简化版的 CycleGAN 生成器模型的网络结构图：

```
Generator (G_A2B)                  Generator (G_B2A)
---------------------------------------------
|      | Conv2D | Instance |                |
| Input|--------|  Norm    |----------------|
|      |  C64   |  ReLU    |    C64'        |
|      |        |          |                |
|      | Conv2D | Instance |                |
|      |--------|  Norm    |----------------|
|      |  C128  |  ReLU    |    C128'       |
|      |        |          |                |
|      | Conv2D | Instance |                |
|      |--------|   Norm   |----------------|
|      |  C256  | ReLU     |    C256'       |
|      |        |          |                |
|           Residual Blocks (9)             |
|      | Conv2D | Instance |                |
|      |--------|  Norm    |----------------|
|      |  C256  |  ReLU    |                |
|      | Conv2DTranspose   | Instance       |
|      |--------|  Norm    |----------------|
|      |  C128  | ReLU     |    C128''      |
|      | Conv2DTranspose   | Instance       |
|      |--------|  Norm    |----------------|
|      |  C64   | ReLU     |    C64''       |
|      | Conv2DTranspose  | Instance        |
|      |--------|  Norm   |-----------------|
|      |     Output (C3)  |                 |
---------------------------------------------
```

其中，G_A2B 用于将 A 类图像转换成 B 类图像，G_B2A 用于将 B 类图像转换成 A 类图像。整个模型由三部分组成：编码器、残差块和解码器。

具体来说，编码器部分由三个卷积层组成，每个卷积层后面跟着一个 Instance Normalization 和 ReLU 激活函数。随后，通过堆叠多个残差块来提高模型的性能。最后，解码器部分由三个反卷积层组成，每个反卷积层后面跟着一个 Instance Normalization 和 ReLU 激活函数。最后一层输出通道数为 3，表示生成的图像是 RGB 图像。

### 2. 判别器

CycleGAN 判别器模型的网络结构也是一个由多个卷积层和全连接层组成的深度神经网络。下面是一个简化版的 CycleGAN 判别器模型的网络结构图：

```
Discriminator (D)
-----------------------------
|      | Conv2D | LeakyReLU |
| Input|--------|--------- -|
|      |  C64   |           |
|      |        |           |
|      | Conv2D | Instance  |
|      |--------|  Norm     |
|      |  C128  | LeakyReLU |
|      |        |           |
|      | Conv2D | Instance  |
|      |--------|  Norm     |
|      |  C256  | LeakyReLU |
|      |        |           |
|      | Conv2D | Instance  |
|      |--------|  Norm     |
|      |  C512  | LeakyReLU |
|      |        |           |
|      |  FC1   | LeakyReLU |
|      |--------|-----------|
|      |  FC2   | Sigmoid   |
|      |--------|---------- |
|      | Output |           |
-----------------------------
```

判别器模型的输入是一张图像，输出是一个二元值（0 或 1），表示输入图像是否为真实图像。判别器模型的核心思想是通过对抗训练的方式来学习区分真实图像和生成图像的能力。具体来说，它通过最小化真实图像和生成图像之间的差异，并同时最大化生成器生成的图像被判别为真实图像的概率，从而不断提高自己的鉴别能力。

### 损失函数

#### 1. 生成器损失函数

共有两个生成器，分别为由 A 生成 B、和由 B 生成 A 的两个生成器，均使用 MSELOSS。
最终，总的生成器 LOSS 为：`loss_GAN = (loss_GAN_A2B + loss_GAN_B2A) / 2`

#### 2. 判别器损失函数

有两个判别器，判别器 B 的作用是利用 A2B 生成器生成 B，查看与真实 B 的偏差，判别器 A 同理，使用 L1Loss。
最终，总的判别器 LOSS 为：`loss_D = (loss_D_A + loss_D_B) / 2`

#### 3. cycleLoss

利用生成的假的 A 和假的 B，利用生成器恢复真的 A 和 B，保证复原的结果与原始结果一致，此为 cycle 的 loss，使用 L1Loss。

```
        loss_cycle_ABA=criterion_cycle(recovered_A,real_A)
        recovered_B=netG_A2B(fake_A)
        loss_cycle_BAB=criterion_cycle(recovered_B,real_B)
        loss_cycle = (loss_cycle_ABA + loss_cycle_BAB) / 2
```

## 目录结构说明

```
cycleGAN
├─ .gitignore
├─ dataset                  // 数据集  (该目录体积过大，git已忽略，请自行下载)
├─ images                   // 训练时生成的实时结果，每训练100个样本输出一次，后续会覆盖
├─ logs                     // 日志
├─ models                   // 模型文件
│  └─ netD_A.pth            // 判别器A
│  └─ netD_B.pth            // 判别器B
│  └─ netG_A2B.pth          // 生成器：使用A生成B
│  └─ netG_B2A.pth          // 生成器：使用B生成A
├─ outputs-epoch100         // 训练100个epoch后的测试结果  (该目录体积过大，git已忽略，请自行生成)
│  └─ A
│  └─ B
├─ LICENSE
├─ README.md
├─ result                   // 自己手动取的对比图
│  └─ 220.jpg
├─ test.py                  // 测试脚本
├─ train.py                 // 训练脚本
├─ datasets.py              // 数据预处理
└─ utils.py                 // 工具函数集合

```

## 实验结果

<!-- ### 1. 20 个 epoch

```
[Epoch 20/20] [Batch 1018/1019] [D loss: 0.168274] [G loss: 1.308673, adv: 0.394398, cycle: 0.032541, identity: 0.117772] ETA: 2:11:55.82506760343
```

### 2. 50 个 epoch

```

```

### 3.  -->

```
[Epoch 50/50] [Batch 1019/1019] [D loss: 0.168199] [G loss: 1.151066, adv: 0.407010, cycle: 0.026193, identity: 0.096426] ETA: 5:47:13.2538603708706
```

最终训练 50 个 epoch，训练样本数 1019，最终判别器 loss= 0.168199，cycle 生成器 loss= 0.026193

### 最终效果

左边是生成的图，右边是原图

苹果 to 橙子

![demo](./result/220.jpg)

橙子 to 苹果

![demo](./result/133.jpg)

<!-- ```
[Epoch 100/100] [Batch 1018/1019] [D loss: 0.340311] [G loss: 0.552427, adv: 0.116484, cycle: 0.017132, identity: 0.052925] ETA: 11：49：41.962243098504
``` -->
