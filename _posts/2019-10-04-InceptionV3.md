---
layout:     post
title:      重新思考计算机视觉的Inception架构
subtitle:   Rethinking the Inception Architecture for Computer Vision
date:       2019-10-04
author:     CJR
header-img: img/2019-10-04-InceptionV3/post-bg.jpg
catalog: true
mathjax: true
tags:
    - Inception
    - CNN
---

## Inception V3

&emsp;这篇文章是Inception系列的第三篇，在文章中提出了Inception V3（arxiv上的论文还有V2的，不过在CVPR上的文章基本都改成了V3，可能是考虑到大家都喜欢把之前的BN-Inception称为Inception V2吧。。。），文章发表在了CVPR 2016。原文可见[Rethinking the Inception Architecture for Computer Vision](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)。

---

## 摘要

&emsp;卷积网络是大部分最先进的计算机视觉解决方案的核心。2014年以来，深度卷积网络开始成为主流，在各种基准测试中获得了巨大的收益。虽然增加的模型大小和计算成本往往转化为大多数任务的即时质量收益（只要提供足够的标记数据用于训练），计算效率和低参数量仍然是各种用例（如移动视觉和大数据场景）的有利因素。在这里，我们正在探索扩展网络的方法，目的是通过适当的分解卷积和积极的正则化，尽可能有效地利用增加的计算量。我们在ILSVRC 2012分类挑战验证集上对我们的方法进行了基准测试，结果表明，与SOTA相比，我们的方法取得了显著的进步：使用一个计算成本为50亿乘加法/推断和使用少于2500万个参数的网络进行单帧评估时，top-1错误率为21.2%，top-5错误率为5.6%。通过4个模型的集成和multi-crop评估，我们在验证集上报告了3.5%的top-5错误率和17.3%的top-1错误率，在官方测试集上报告了3.6%的top-5错误率。

## 1. 引言

&emsp;自Krizhevsky等人2012年ImageNet竞赛获奖以来，他们的网络“AlexNet”已成功应用于各种各样的计算机视觉任务，例如目标检测、分割、人体姿势估计、视频分类、目标跟踪和超分辨率。

&emsp;这些成功催生了一个新的研究方向，即寻找性能更高的卷积神经网络。从2014年开始，通过利用更深更广的网络，网络架构的质量得到了显着改善。VGGNet和GoogLeNet在2014年ILSVRC分类挑战中取得了类似的高性能。一个有趣的观察结果是，在广泛的应用领域中，分类性能的提高往往会转化为显著的质量提高。这意味着，深度卷积架构中的架构改进可以用于提高大多数其他计算机视觉任务的性能，这些任务越来越依赖于高质量、学习到的视觉特征。此外，在AlexNet特征无法与手工设计的解决方案（如检测中的proposal生成）竞争的情况下，网络质量的提高导致卷积网络有了新的应用领域。

&emsp;尽管VGGNet具有引人注目的体系结构简单性特性，但这样做的代价很高：评估网络需要大量计算。另一方面，GoogLeNet的Inception架构也被设计为即使在内存和计算预算受到严格约束的情况下也能很好地执行。例如，GoogleNet使用了大约700万个参数，这比它的前身AlexNet减少了9倍，后者使用了6000万个参数。此外，VGGNet使用的参数大约是AlexNet的三倍。

&emsp;Inception的计算成本也比VGGNet或其性能更好的后继低得多。这使得在需要以合理成本处理大量数据的大数据场景或内存或计算能力固有限制的场景（例如在移动视觉环境中）中利用Inception网络变得可行。通过对目标内存使用应用专门的解决方案，或者通过计算技巧优化某些操作的执行，当然可以缓解部分问题。然而，这些方法增加了额外的复杂性。此外，这些方法也可以用于优化Inception架构，再次扩大效率差距。

&emsp;

&emsp;

&emsp;

## 2. Towards Reducing Internal Covariate Shift

&emsp;

&emsp;

&emsp;

&emsp;

## 3. 通

&emsp;

&emsp;

&emsp;

&emsp;

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-09-27-InceptionV3/algorithm1.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">1. </div>
</center>

&emsp;

&emsp;

### 3.1 批

&emsp;

### 3.2 批

&emsp;

&emsp;

&emsp;

### 3.3 批

&emsp;

&emsp;

&emsp;

### 3.4 批

&emsp;

## 4. 实验

### 4.1 5

&emsp;

&emsp;

### 4.2 ImageNet分类

&emsp;

&emsp;

#### 4.2.1 加

&emsp;

&emsp;

&emsp;

&emsp;

&emsp;

&emsp;

&emsp;

#### 4.2.2 网络分类

&emsp;

&emsp;

&emsp;

&emsp;

&emsp;

&emsp;

&emsp;

&emsp;

&emsp;

#### 4.2.3 分类

&emsp;

&emsp;

&emsp;

## 5. 结论

&emsp;

&emsp;

&emsp;

---

## 个人看法

&emsp;
