---
layout:     post
title:      MobileNetV3
subtitle:   Third Version of MobileNet
date:       2019-05-22
author:     CJR
header-img: img/post-bg-brain.jpg
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

## MobileNetV3
&emsp;谷歌在2019年5月在arxiv上公开了MobileNetV3的论文，原论文见 [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244v2)。

&emsp;下面先对论文做一个简单的翻译：

## 摘要
&emsp;我们展示了基于互补搜索技术和新颖架构设计相结合的下一代MobileNet。MobileNetV3通过结合硬件感知的网络架构搜索(NAS)和NetAdapt算法对移动手机的cpu进行调优，然后通过新的架构改进对其进行改进。本文开始探索自动化搜索算法和网络设计如何协同工作，利用互补的方法来提高整体STOA。通过这个过程，我们创建发布了两个新的MobileNet模型:MobileNetV3-Large和MobileNetV3-Small，它们分别针对高资源和低资源两种情况。然后将这些模型应用于目标检测和语义分割。针对语义分割(或任何密集像素预测)任务，我们提出了一种新的高效分割解码器Lite reduce Atrous Spatial Pyramid Pooling (LR-ASPP)。我们实现了移动分类、检测和分割的STOA。与MobileNetV2相比，MobileNetV3-Large在ImageNet分类上的准确率提高了3.2%，同时延迟降低了15%。与MobileNetV2相比，MobileNetV3-Small的准确率高4.6%，同时延迟降低了5%。MobileNetV3-Large检测速度比MobileNetV2快25%，在COCO检测上的精度大致相同。MobileNetV3-Large LR-ASPP的速度比MobileNetV2 R-ASPP快30%，在城市景观分割的精度类似。

## 1. 介绍
&emsp;高效的神经网络在移动应用程序中变得无处不在，从而实现全新的设备体验。它们也是个人隐私的关键推动者，允许用户获得神经网络的好处，而无需将数据发送到服务器进行评估。神经网络效率的进步不仅通过更高的精度和更低的延迟改善了用户体验，而且通过降低功耗有助于保持电池寿命。

&emsp;本文描述了我们开发MobileNetV3大型和小型模型的方法，以提供下一代高精度高效的神经网络模型来驱动设备上的计算机视觉。新的网络推动了STOA，并展示了如何将自动化搜索与新的体系结构进步结合起来，以构建有效的模型。

&emsp;本文的目标是开发最佳的移动计算机视觉架构，以优化在移动设备上的精度-延迟之间的权衡。为了实现这一点，我们引入了(1)互补搜索技术，(2)适用于移动设备的非线性的新高效版本，(3)新的高效网络设计，(4)一个新的高效分割解码器。我们提供了深入的实验，以证明每种技术在广泛的用例和移动手机上评估的有效性和价值。

&emsp;论文组织如下。我们从第2节中有关工作的讨论开始。第3节回顾了用于移动模型的高效构建块。第4节回顾了体系结构搜索以及MnasNet和NetAdapt算法的互补性。第5节描述了通过联合搜索提高模型效率的新型架构设计。第6节为分类、检测和分割提供了广泛的实验，以证明其有效性并了解不同元素的贡献。第7节包含结论和今后的工作。

## 2. 相关工作
&emsp;设计深度神经网络结构来实现精度和效率之间的最优平衡是近年来一个活跃的研究领域。无论是新颖的手工结构还是算法神经结构搜索，都在这一领域发挥了重要作用。

&emsp;SqueezeNet广泛使用1x1卷积与挤压和扩展模块，主要集中于减少参数的数量。最近的工作将关注点从减少参数转移到减少操作的数量(MAdds)和实际测量的延迟。MobileNetV1采用深度可分离卷积，大大提高了计算效率。MobileNetV2在此基础上进行了扩展，引入了一个具有反向残差和线性瓶颈的资源高效块。ShuffleNet利用组卷积和通道洗牌操作进一步减少MAdds。CondenseNet在训练阶段学习组卷积，以保持层与层之间有用的紧密连接，以便特征重用。ShiftNet提出了与逐点卷积交织的移位操作，以取代昂贵的空间卷积。

&emsp;为了使体系结构设计过程自动化，首先引入了增强学习(RL)来搜索具有竞争力的精度的高效体系结构。一个完全可配置的搜索空间可能会以指数级增长且难以处理。因此，早期的架构搜索工作主要关注单元级结构搜索，并且在所有层中重用相同的单元。最近，有文章探索了一个块级分层搜索空间，允许在网络的不同分辨率块上使用不同的层结构。为了降低搜索的计算成本，在一些研究中使用了基于梯度优化的可微体系结构搜索框架。针对现有网络适应受限移动平台的问题，有些文章中提出了更高效的自动化网络简化算法。

&emsp;量化是另一个重要的补充努力，通过降低精度的算法来提高网络效率。最后，知识蒸馏提供了一种附加的补充方法，在大型“教师”网络的指导下生成精确的小型“学生”网络。

## 3. 有效的移动构建块(Efficient Mobile Building Blocks)

&emsp;移动模型已经建立在越来越高效的构建块之上。MobileNetV1引入深度可分离卷积作为传统卷积层的有效替代。深度可分卷积通过将空间滤波与特征生成机制分离，有效地分解了传统卷积。深度可分卷积由两个独立的层定义:用于空间滤波的轻量级深度卷积和用于特征生成的较重的1x1逐点卷积。

&emsp;MobileNetV2引入了线性瓶颈和反向残差结构，以便利用问题的低秩性质使层结构更加有效。这个结构如下图所示，由1x1拓展卷积（注：即提升通道数）、深度卷积和1x1投影层定义。当且仅当输入和输出具有相同数量的通道时，才用残差链接连接它们。该结构在保持输入和输出的紧凑表示的同时，在内部扩展到高维特征空间，以增加每个通道非线性转换的表现力。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/MobileNetV2.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">MobileNetV2层(反向残差和线性瓶颈)。每个块由窄的输入和输出(瓶颈)组成，它们没有非线性，然后扩展到一个更高维度的空间并投影到输出。残差连接瓶颈(而不是扩展)</div>
</center>

&emsp;MnasNet建立在MobileNetV2结构上，在瓶颈结构中引入基于压缩和激励（squeeze and excitation）的轻量级注意力模块。注意，与[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)中提出的基于ResNet的模块相比，压缩和激励模块集成在不同的位置。此模块位于拓展的深度滤波器之后，以便注意力机制应用于最大的表示（representation），如下图所示。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/MobileNetV3.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">MobileNetV2+Squeeze-and-Excite。与SENet相比，我们在残差层施加压缩和激励。 我们根据层的不同使用不同的非线性，详见5.2节</div>
</center>

&emsp;对于MobileNetV3，我们使用这些层的组合作为构建块，以便构建最有效的模型。图层也通过修改swish非线性进行升级。压缩和激励以及swish非线性都使用了sigmoid，它的计算效率很低，而且很难在定点算法中保持精度，因此我们将其替换为硬sigmoid，如5.2节所讨论的。

## 4. 网络搜索

&emsp;网络搜索已被证明是发现和优化网络架构的一个非常强大的工具。对于MobileNetV3，我们使用平台感知的NAS通过优化每个网络块来搜索全局网络结构。然后，我们使用NetAdapt算法搜索每个层的过滤器数量。这些技术是互补的，可以结合起来为给定的硬件平台有效地找到优化模型。

### 4.1 用于块搜索的平台感知NAS

&emsp;我们采用平台感知神经结构方法来寻找全局网络结构。由于我们使用相同的基于RNN的控制器和相同的分解层次搜索空间，所以对于目标延迟在80ms左右的大型移动模型，我们发现了与[MnasNet](https://arxiv.org/abs/1807.11626v2)类似的结果。因此，我们只需重用相同的MnasNet-A1作为我们的初始大型移动模型，然后在其上应用[NetAdapt](https://arxiv.org/abs/1804.03230v2)和其他优化。

&emsp;然而，我们发现最初的奖励设计并没有针对小型手机模型进行优化。具体来说，它使用一个多目标奖励$ACC(m) \times [LAT(m)/TAR]^{w}$来近似帕累托最优解，根据目标延迟$TAR$对每个模型$m$平衡模型精度$ACC(m)$和延迟$LAT(m)$。我们观察到，对于小模型，精度随延迟变化更为显著；因此，我们需要一个较小的权重因子$\omega = - 0.15$(与MnasNet中原始的$\omega = - 0.07$相比)来补偿不同延迟下较大的精度变化。在新的权重因子$\omega$的增强下，我们从头开始一个新的架构搜索，以找到初始的种子模型，然后应用NetAdapt和其他优化来获得最终的MobileNetV3-Small模型。

$$a^2 + b^2 = c^2$$
