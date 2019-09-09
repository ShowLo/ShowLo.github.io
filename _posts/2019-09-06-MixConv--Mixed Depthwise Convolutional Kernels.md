---
layout:     post
title:      MixConv--混合的深度卷积核
subtitle:   MixConv--Mixed Depthwise Convolutional Kernels
date:       2019-09-06
author:     CJR
header-img: img/2019-09-06-MixConv--Mixed Depthwise Convolutional Kernels/post-bg.jpg
catalog: true
mathjax: true
tags:
    - Lightweight Network
    - MixNet
    - CNN
---

## MixNet

&emsp;这篇文章来自谷歌大脑，是最近（19年7月）刚挂出来的，然后应该是发表在了BMVC 2019上。原文可见[MixConv: Mixed Depthwise Convolutional Kernels](https://arxiv.org/abs/1907.09595)。

---

## 摘要

&emsp;深度卷积在现代高效卷积神经网络中越来越受欢迎，但其核大小却常常被忽略。在本文中，我们系统地研究了不同核大小的影响，并观察到将不同尺寸核的优点结合起来可以获得更好的精度和效率。在此基础上，我们提出了一种新的混合深度卷积（MixConv），它很自然地将多个核大小混合在一个卷积中。我们的MixConv作为对普通的深度卷积的一种简单的代替，提高了现有MobileNets在ImageNet分类和COCO目标检测方面的准确性和效率。为了证明MixConv的有效性，我们将其整合到AutoML搜索空间中，开发了一个新的模型家族，命名为MixNets，其性能优于之前的移动模型，包括MobileNetV2（ImageNet top-1准确率+4.2%）、ShuffleNetV2（+3.5%）、MnasNet（+1.3%)、ProxylessNAS（+2.2%）和FBNet（+2.0%）。特别值得一提的是，我们的MixNet-L在典型的移动设置（<600M FLOPS）下实现了最新的78.9%的ImageNet top-1精度。代码位于<https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet>。

## 1. 引言

&emsp;卷积神经网络（ConvNets）在图像分类、检测、分割等方面有着广泛的应用。卷积神经网络设计的一个最新趋势是提高精度和效率。随着这一趋势，深度卷积在现代卷积中越来越受欢迎，如MobileNets、ShuffleNets、NASNet、AmoebaNet、MnasNet和EfficientNet。与普通卷积不同，深度卷积核分别应用于每个单独的通道，从而将计算成本降低了一个因子C，其中C是通道数。在设计具有深度卷积核的卷积神经网络时，一个重要但常常被忽略的因素是核大小。虽然常用的做法是简单地使用3x3卷积核，但是最近的研究结果表明，更大的核尺寸，如$5\times 5$卷积核和$7\times 7$卷积核，可能会提高模型的精度和效率。

&emsp;在本文中，我们重新讨论了一个基本问题：更大的核是否总是能够获得更高的精度？从在AlexNet中首次观察到之后，我们就知道每个卷积核都负责捕获一个局部图像模式，这可能是早期阶段的边缘，也可能是后期阶段的对象。大的核往往以牺牲更多的参数和计算为代价来获取具有更多细节的高分辨率模式，但是它们总是提高精度吗？为了回答这个问题，我们基于MobileNets系统地研究了核大小的影响。图1显示了结果。正如预期的那样，较大的内核大小会显著地使用更多参数从而增加模型大小；然而，从$3\times 3$到$7\times 7$模型精度首先提高，但当核大小大于$9\times 9$时，模型精度迅速下降，这表明非常大的核大小可能会损害精度和效率。事实上，这个观察符合卷积神经网络的最初直觉：在核大小等于输入分辨率的极端情况下，ConvNet简单地成为一个完全连接的网络，这是众所周知的次品。本研究指出单一核大小的局限性：我们既需要大卷积核来捕获高分辨率模式，也需要小卷积核来捕获低分辨率模式，以获得更好的模型精度和效率。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-09-06-MixConv--Mixed Depthwise Convolutional Kernels/figure1.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图1. 精度vs核大小--每个点代表MobileNet V1和V2的模型变体，其中模型大小由点大小表示。较大的内核会产生更多的参数，但当核大小大于$9\times 9$时，精度实际上会下降。</div>
</center>

&emsp;在此基础上，我们提出了一种混合深度卷积（MixConv），它将不同的核大小混合在一个卷积运算中，这样就可以很容易地用不同的分辨率捕获不同的模式。图2显示了MixConv的结构，它将通道划分为多个组，并对每组通道应用不同的核大小。我们的MixConv是一个简单的替换普通深度卷积的方法，但是它在ImageNet分类和COCO目标检测上都能显著提高MobileNets的准确性和效率。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-09-06-MixConv--Mixed Depthwise Convolutional Kernels/figure2.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图2. 混合深度卷积(MixConv)——与对所有通道应用单个核的普通深度卷积不同，MixConv将通道划分为多个组，并对每个组应用不同的核大小。</div>
</center>

&emsp;为了进一步证明MixConv的有效性，我们利用神经结构搜索开发了一个名为MixNets的新模型家族。实验结果表明，我们的MixNet模型明显优于之前所有的移动卷积神经网络，如ShuffleNets、MnasNet、FBNet和ProxylessNAS。特别是，我们的中等尺寸的MixNet-M实现了与ResNet-152相同的77.0% ImageNet top-1精度，同时使用了比ResNet-152少12倍的参数和31倍的FLOPs。

>注：原文用的是FLOPS，虽然只有大小写上的区别，但实际上FLOPS和FLOPs是不同的概念，FLOPS是“每秒所执行的浮点运算次数”（floating-point operations per second）的缩写，它常被用来估算电脑的执行效能，而FLOPs全称是floating point operations，即表示浮点运算次数，所以这里应该是FLOPs才对。

## 2. 相关工作

**高效卷积神经网络：** 近年来，人们在提高卷积神经网络效率方面做出了巨大的努力，从更高效的卷积操作、瓶颈层，到更高效的架构。特别是，深度卷积在所有移动尺寸的卷积神经网络中越来越流行，如MobileNets、ShuffleNets、MnasNet以及其他。最近，EfficientNet通过广泛使用深度和逐点卷积，甚至实现了最先进的ImageNet精度和十倍更高的效率。与常规卷积不同，深度卷积对每个通道分别执行卷积核，从而减小了参数大小和计算成本。我们提出的MixConv推广了深度卷积的概念，可以认为是普通深度卷积的直接替代。

**多尺度网络和特性：** 我们的想法与之前的多分支卷积神经网络有很多相似之处，比如Inceptions、Inception-ResNet、ResNeXt和NASNet。通过在每一层中使用多个分支，这些卷积神经网络能够在一层中使用不同的操作（例如卷积和池化）。同样，之前也有很多工作是将不同层次的多尺度特征图结合起来，如DenseNet和特征金字塔网络。然而，与以前的工作不同，这些工作主要集中在改变神经网络的宏观架构以便利用不同的卷积运算，我们的工作旨在设计单个深度卷积的直接替换，目的是轻松利用不同的卷积核大小而不改变网络结构。

**神经架构搜索：** 最近，神经架构搜索通过自动化设计过程和学习更好的设计选择，实现了比手工模型更好的性能。因为我们的MixConv是一个灵活的操作，有许多可能的设计选择，所以我们使用了与[ProxylessNAS, MnasNet, FBNet]类似的现有架构搜索方法，通过将MixConv添加到搜索空间来开发一个新的MixNets家族。

## 3. MixConv

&emsp;

### 3.1 架构

&emsp;

&emsp;

&emsp;

&emsp;

&emsp;

### 3.2 模块

&emsp;


### 3.3 架构

&emsp;

#### 3.3.1 细节

&emsp;

&emsp;

## 4. 评估

&emsp;

&emsp;

&emsp;

&emsp;

&emsp;

## 5

&emsp;

&emsp;

### 5.1

&emsp;

### 5.2

&emsp;

&emsp;

### 5.3

&emsp;

&emsp;

&emsp;

## 6. 结论

&emsp;

&emsp;

&emsp;

&emsp;

---

## 个人看法

&emsp;
