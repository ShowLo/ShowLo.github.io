---
layout:     post
title:      批归一化--通过减少internal covariate shift来加速深度网络训练
subtitle:   Batch Normalization--Accelerating Deep Network Training by Reducing Internal Covariate Shift
date:       2019-09-27
author:     CJR
header-img: img/2019-09-27-BN-Inception(InceptionV2)/post-bg.jpg
catalog: true
mathjax: true
tags:
    - Batch Normalization
    - Inception
    - CNN
---

## BN-Inception/Inception V2

&emsp;这篇文章是Inception系列的第二篇，虽然在之后的第三篇中提出了官方定义的Inception V2，不过大家貌似更喜欢把这篇文章提出的BN-Inception当作Inception V2，文章发表在了ICML 2015。原文可见[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)。

---

## 摘要

&emsp;由于训练过程中各层输入的分布随前一层参数的变化而变化，使得训练深度神经网络变得复杂。这降低了训练的速度，因为需要更低的学习速度和更仔细的参数初始化，并且使得用饱和非线性来训练模型变得非常困难。我们将这种现象称为internal covariate shift，并通过归一化层输入来解决这个问题。我们的方法将归一化作为模型体系结构的一部分，并对每个训练小批次执行归一化，从而获得了它的优势。批归一化使我们可以使用更高的学习率，而对初始化则不必那么小心。它还作为一个正则化器，在某些情况下消除了dropout的需要。批归一化应用于最先进的图像分类模型，以少了14倍的训练步骤即可达到相同的精度，并且在很大程度上击败了原始模型。使用一组批归一化网络，我们对ImageNet分类的最佳发表结果进行了改进：达到4.9%的top-5验证误差（和4.8%的测试误差），超过了人类评分者的准确度。

## 1. 引言

&emsp;深度学习极大地提高了视觉、语言和许多其他领域的技术水平。随机梯度下降（SGD）已被证明是训练深度网络的一种有效方法，SGD变体如momentum和Adagrad已被用来实现最先进的性能。SGD优化网络参数$\Theta$，以减少损失

$$
\Theta=\mathop{\arg\min}_{\Theta}\frac{1}{N}\sum_{i=1}^{N}l\left(x_{i},\Theta\right)
$$

&emsp;

&emsp;

## 2. 相关工作

&emsp;

&emsp;

&emsp;

&emsp;

&emsp;

## 3. 动

&emsp;

&emsp;

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-09-27-BN-Inception(InceptionV2)/figure1.png">
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

&emsp;

## 4. 架构细节

&emsp;

&emsp;

&emsp;

&emsp;

&emsp;

&emsp;

&emsp;

## 5. Net

&emsp;

&emsp;

&emsp;

&emsp;

&emsp;

&emsp;

## 6. 训练

&emsp;

&emsp;

## 7. 结果

&emsp;

&emsp;

&emsp;

&emsp;

## 8. 和结果

&emsp;

&emsp;

&emsp;

## 9. 结论

&emsp;

&emsp;

&emsp;

---

## 个人看法

&emsp;
