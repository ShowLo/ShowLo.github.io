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

&emsp;MixConv的主要思想是在一个深度卷积操作中混合多个不同大小的卷积核，这样就可以很容易地从输入图像中捕获不同类型的模式。在本节中，我们将讨论MixConv的特征图和设计选择。

### 3.1 MixConv Feature Map

&emsp;我们从普通的深度卷积开始。设$X^{(h,w,c)}$为形状为$(h,w,c)$的输入张量，其中$h$为空间高度，$W^{(k,k,c,m)}$为空间宽度，$c$为通道大小。设$W$为深度卷积核，其中$k\times k$为核大小，$c$为输入通道大小，$m$为通道乘数。为了简单起见，这里我们假设核宽度和高度是相同的$k$，但是很容易将其推广到核宽度和高度不同的情况。输出张量$Y^{(h,w,c\cdot m)}$将具有相同的空间形状$(h,w)$和成倍的输出通道大小$m\cdot c$，每个输出特征图的值计算如下：

$$
Y_{x,y,z}=\sum_{-\frac{k}{2}\le i\le\frac{k}{2},-\frac{k}{2}\le j\le\frac{k}{2}}X_{x+i,y+j,z/m}\cdot W_{i,j,z},\qquad\forall z=1,\ldots,m\cdot c
$$

&emsp;与普通的深度卷积不同，MixConv将通道划分为组，并为每个组提供不同的核大小，如图2所示。更具体地说，输入张量划分为g组的虚拟张量$<\hat{X}^{(h,w,c_{1})},\ldots,\hat{X}^{(h,w,c_{g})}>$，所有虚拟张量$\hat{X}$具有相同的空间高度$h$和宽度$w$，并且他们的总通道大小与原始输入张量的相等：$c_{1}+c_{2}+\ldots+c_{g}=c$。同样的，我们也将卷积核分成g组虚拟核：$<\hat{W}^{(k_{1},k_{1},c_{1},m)},\ldots,\hat{W}^{(k_{g},k_{g},c_{g},m)}>$。对于第t组虚拟输入张量和核，相应的虚拟输出计算如下：

$$
\hat{Y}^{t}_{x,y,z}=\sum_{-\frac{k_{t}}{2}\le i\le\frac{k_{t}}{2},-\frac{k_{t}}{2}\le j\le\frac{k_{t}}{2}}\hat{X}^{t}_{x+i,y+j,z/m}\cdot \hat{W}^{t}_{i,j,z},\qquad\forall z=1,\ldots,m\cdot c_{t}
$$

&emsp;最终输出张量是所有虚拟输出张量$<\hat{Y}^{1}_{x,y,z_{1}},\ldots,\hat{Y}^{g}_{x,y,z_{g}}>$的串联（concatenation）：

$$
Y_{x,y,z_{o}}=Concat(\hat{Y}^{1}_{x,y,z_{1}},\ldots,\hat{Y}^{g}_{x,y,z_{g}})
$$

&emsp;其中$z_{o}=z_{1}+\ldots+z_{g}=m\cdot c$是最终的输出通道大小。

&emsp;图3是一个python实现的基于Tensorflow的MDConv简单样例。在某些平台上，MixConv可以作为一个单独的操作实现，并使用分组卷积进行优化。尽管如此，如图所示，MixConv可以被认为是一种简单的取代普通深度卷积的方法。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-09-06-MixConv--Mixed Depthwise Convolutional Kernels/figure3.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图3. Tensorflow 实现的MDConv样例</div>
</center>

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
