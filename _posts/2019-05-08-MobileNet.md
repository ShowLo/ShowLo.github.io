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

&emsp;上面是一个标准的卷积过程，输入通道数为M，输出通道数为N，输入图片一个通道的大小为H×W，一共有M×N个卷积核且每个卷积核的大小为K×K，所以标准卷积的计算量为$HWK^{2}MN$。

## 深度可分离卷积

&emsp;接下来就是MobileNet的重点--深度可分离卷积了。因为标准卷积实际上在做卷积操作的时候是在空间和通道两个维度同时进行的，然后深度可分离卷积的做法就是把空间和通道两个维度的卷积操作给分开成了深度卷积和逐点卷积两个部分。

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
