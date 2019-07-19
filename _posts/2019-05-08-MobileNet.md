---
layout:     post
title:      MobileNet
subtitle:   A lightweighted network
date:       2019-05-08
author:     CJR
header-img: img/2019-05-08-MobileNet/post-bg-net.jpg
catalog: true
mathjax: true
tags:
    - MobileNet
    - Neural Network
    - AI
    - Deep Learning
    - Machine Learning
    - Lightweight Network
---

# MobileNet

&emsp;MobileNet是谷歌于2017年发布的一个可以用于移动端以及嵌入式设备的轻量级网络，原论文见[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)。这篇论文最重要的部分是深度可分离卷积，接下来将会详细介绍一下深度可分离卷积的原理。

---

## 标准卷积

&emsp;首先我们先看一下标准卷积的过程，如图1所示：

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-08-MobileNet/conv-std.jpg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图1. 标准卷积</div>
</center>

&emsp;上面是一个标准的卷积过程，输入通道数为$M$，输出通道数为$N$，输入图片一个通道的大小为$H\times W$，一共有$N$个滤波器，每个滤波器由$M$个$K\times K$大小的卷积核组成，所以标准卷积的计算量为$HWK^{2}MN$。

## 深度可分离卷积

&emsp;接下来就是MobileNet的重点--深度可分离卷积了。因为标准卷积实际上在做卷积操作的时候是在空间和通道两个维度同时进行的，然后深度可分离卷积的做法就是把空间和通道两个维度的卷积操作给分开成了空间维度的深度卷积和通道维度的逐点卷积两个部分。

### 深度卷积

&emsp;首先看一下深度卷积，如图2所示：

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-08-MobileNet/depthwise-conv.jpg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图2. 深度卷积</div>
</center>

&emsp;深度卷积的最大特点是滤波器的数量和输入通道数一致，且每个滤波器就只有一个$K\times K$大小的卷积核，做深度卷积的时候一个卷积核只和对应的通道做卷积，输出一个通道的feature map。所以对于具有$M$个通道的每个通道大小为$H\times W$的输入，只需要总共$M$个$K\times K$大小的卷积核，输出$M$张大小为$H\times W$的feature map。其计算量为$HWK^{2}M$。

### 逐点卷积

&emsp;逐点卷积的示意图如图3所示：

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-08-MobileNet/pointwise-conv.jpg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图3. 逐点卷积</div>
</center>

&emsp;实际上，这就是$K=1$时的标准卷积，其计算量为$HWMN$。


&emsp;因此，我们可以看到深度可分离卷积总共的计算量为$HWK^{2}M+HWMN$，跟标准卷积相比仅为原来的$\frac{HWK^{2}M+HWMN}{HWK^{2}MN}=\frac{1}{N}+\frac{1}{K^{2}}$，一般情况下我们会取$K=3$，$N$通常也是一个很大的数，所以跟标准卷积相比，深度可分离卷积的计算量仅为原来的$\frac{1}{8}\sim\frac{1}{9}$。