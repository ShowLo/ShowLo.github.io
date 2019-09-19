---
layout:     post
title:      EffNet--卷积神经网络的一种高效结构
subtitle:   EffNet--An Efficient Structure For Convolutional Neural Networks
date:       2019-09-19
author:     CJR
header-img: img/2019-09-19-EffNet/post-bg.jpg
catalog: true
mathjax: true
tags:
    - Lightweight Network
    - EffNet
    - CNN
---

## MixNet

&emsp;这篇文章主要是在MobileNet之上做修改，增加了一个空间可分离卷积，发表在2018 25th IEEE International Conference on Image Processing (ICIP)上。原文可见[EffNet: An Efficient Structure for Convolutional Neural Networks](https://arxiv.org/abs/1801.06434)。

---

## 摘要

&emsp;随着卷积神经网络在客户产品中的应用日益广泛，对模型在嵌入式、移动硬件上高效运行的需求也日益显现。因此，从二值网络到修正卷积层，用各种各样的方法使得模型变得更加精细，成为研究的热点。我们对后者做出了贡献，并提出了一种新的卷积块，它在大大降低计算负担的同时，超过了目前的SOTA。我们的模型，被称为EffNet，是针对一开始就比较瘦的模型进行优化的，旨在解决现有模型（如MobileNet和ShuffleNet）中的问题。

## 1. 引言

&emsp;随着最近工业上对人工神经网络对产品性能的好处的认识，对高效算法的需求出现了，这种算法需要在成本低廉的硬件上实时运行。这在某种程度上与几乎平行的大学研究相矛盾。后者在执行周期和硬件方面享有相对的自由，而前者受市场压力和产品需求的影响。

&emsp;多年来，多篇论文提出了在小型硬件上进行实时推理的不同方法。一个例子是对训练过的网络进行剪枝。另一个是32位网络到二进制模型的定点转换。最近的研究集中在神经元的相互连接和普通卷积层的本质。普通卷积层的核心是一个四维张量，它以如下格式\[行、列、输入通道、输出通道\]扫过输入信号，产生四倍分量乘法，从而将计算成本放大四倍。

&emsp;由于$3\times 3$卷积现在是一种标准，它们自然成为优化的候选项。论文如\[MobileNet\]和\[ShuffleNet\]通过将计算按不同的维度分开来解决这个问题。然而，在他们的方法有两个问题没有解决。首先，两篇论文都报道了将大型网络变得更小、更高效。当将他们的模型应用于更小的网络时，结果会有所不同。其次，两种模型都为网络中的数据流造成了一个严重的瓶颈。这种瓶颈在高冗余模型中可能被证明是无关紧要的，但正如我们的实验所示，它对较小的模型具有破坏性的影响。

&emsp;因此，我们提出了一个替代系列，它保留了计算量的大部分比例的下降，而对精度几乎没有影响。我们通过优化数据流并忽略在这个独特领域中被证明有害的实践来实现这一改进。我们的新型卷积模块允许我们将更大的网络部署到低容量硬件上，或者提高现有模型的效率。

## 2. 相关工作

&emsp;

## 3. MixConv

&emsp;

### 3.1 MixConv Feature Map

&emsp;

&emsp;

&emsp;

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-09-19-EffNet/figure1.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图1. </div>
</center>

&emsp;

&emsp;

&emsp;

&emsp;

### 3.2 MixConv的设计选择

&emsp;

### 3.3 MixConv在MobileNets上的性能

&emsp;

## 4. MixNet

&emsp;

### 4.1 架构搜索

&emsp;

&emsp;

&emsp;

### 4.2 MixNet在ImageNet上的表现

&emsp;

&emsp;

&emsp;

### 4.3 MixNet架构

&emsp;

### 4.4 迁移学习性能

&emsp;

&emsp;

## 5. 结论

&emsp;

---

## 个人看法

&emsp;
