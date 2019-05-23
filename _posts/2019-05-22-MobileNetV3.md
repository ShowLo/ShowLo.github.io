---
layout:     post
title:      MobileNetV3
subtitle:   Third Version of MobileNet
date:       2019-05-22
author:     CJR
header-img: img/2019-05-22-MobileNetV3/post-bg-brain.jpg
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

# MobileNetV3
&emsp;谷歌在2019年5月在arxiv上公开了MobileNetV3的论文，原论文见 [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244v2)。

&emsp;下面先对论文做一个简单的翻译：

---

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
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/MobileNetV2.png">
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
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/MobileNetV3.png">
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

&emsp;然而，我们发现最初的奖励设计并没有针对小型手机模型进行优化。具体来说，它使用一个多目标奖励$ACC(m) \times [LAT(m)/TAR]^{w}$来近似帕累托最优解，根据目标延迟$TAR$对每个模型$m$平衡模型精度$ACC(m)$和延迟$LAT(m)$。我们观察到，对于小模型，精度随延迟变化更为显著；因此，我们需要一个较小的权重因子$\omega=-0.15$(与MnasNet中原始的$\omega=-0.07$相比)来补偿不同延迟下较大的精度变化。在新的权重因子$\omega$的增强下，我们从头开始一个新的架构搜索，以找到初始的种子模型，然后应用NetAdapt和其他优化来获得最终的MobileNetV3-Small模型。

### 4.2 NetAdapt用于分层搜索

&emsp;我们在架构搜索中使用的第二种技术是NetAdapt，这种方法是对平台感知NAS的补充:它允许以顺序的方式对单个层进行微调，而不是试图推断出粗糙但全局的体系结构。详情请参阅[原文](https://arxiv.org/abs/1804.03230v2)。 简而言之，这项技术的过程如下:

&ensp;1. 从平台感知NAS发现的种子网络体系结构开始。

&ensp;2. 对于每个步骤:

&emsp;(a)生成一组新proposal。每一个proposal都代表了一个架构的修改，与前一步相比，该架构至少能降低延迟$\delta$。

&emsp;(b)对于每一个proposal，我们使用前一步骤的预先训练的模型，并填充新提出的架构，适当地截断和随机初始化缺失的权重。对每个proposal进行T个步骤的微调，以获得对精度的粗略估计。

&emsp;(c)根据某种标准选择最佳proposal。

&ensp;3. 重复前面的步骤，直到达到目标延迟。

&emsp;在NetAdapt中，度量是为了最小化精度的变化。我们修改了这个算法，使延迟变化和精度变化的比例最小化。也就是说，对于每个NetAdapt步骤中生成的所有proposal，我们选择一个使得$\frac{\Delta Acc}{\Delta Latency}$最大的proposal，其中$\Delta Latency$满足2(a)中的约束。 直觉告诉我们，由于我们的proposal是离散的，所以我们更喜欢最大化权衡(trade-off)曲线斜率的proposal。

&emsp;这个过程重复进行，直到延迟达到目标，然后从头开始重新训练新的体系结构。 我们使用与在NetAdapt中用于MobilenetV2的相同的proposal生成器。 具体来说，我们允许以下两种proposal:

&ensp;1. 减小任何拓展层的尺寸;

&ensp;2. 减少共享相同瓶颈大小的所有块中的瓶颈—以维护残差连接。

&emsp;在我们的实验中，我们使用$T = 10000$，发现虽然它增加了proposal初始微调的准确性，但是当从零开始训练时，它并没有改变最终的准确性。 我们设置$\delta=0.01\|L\|$, L是种子模型的延迟。

## 5. 网络的改进

&emsp;除了网络搜索，我们还为模型引入了一些新的组件，以进一步改进最终模型。在网络的开始和结束阶段，我们重新设计了计算代价高的层。我们还引入了一种新的非线性，h-swish，它是最近的swish非线性的改进版本，计算速度更快，更易于量化。

### 5.1 重新设计Expensive的层

&emsp;一旦通过架构搜索找到模型，我们就会发现，一些最后的层以及一些较早的层比其他层更expensive。我们建议对体系结构进行一些修改，以减少这些较慢的层的延迟，同时保持准确性。这些修改超出了当前搜索空间的范围。

&emsp;第一个修改修改了网络的最后几层的交互方式，以便更有效地生成最终的特征。目前的模型基于MobileNetV2的反向瓶颈结构和变体，将1x1卷积作为最后一层以扩展到高维特征空间。这一层非常重要，因为它具有丰富特征用于预测。然而，这是以额外的延迟为代价的。

&emsp;为了减少延迟并保留高维特征，我们将该层移到最终的平均池化之后。最后一组特征现在以1x1空间分辨率计算，而不是7x7空间分辨率。这种设计选择的结果是，在计算和延迟方面，特征的计算变得几乎是free的。

&emsp;一旦降低了该特征生成层的成本，就不再需要以前的瓶颈投影层来减少计算量。该观察允许我们删除前一个瓶颈层中的投影和滤波层，从而进一步降低计算复杂度。原始阶段和优化后的最后一个阶段如下图所示。有效的最后一个阶段将延迟减少了10毫秒，即15%的运行时间，并将操作数量减少了3000万个MAdds，几乎没有损失精度。 第6节包含了详细的结果。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/original-efficient.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">原来的和有效的最后一个阶段的比较。这个更有效的最后阶段能够在不损失精度的情况下，在网络的末端丢弃三个expensive的层</div>
</center>

&emsp;另一个expensive的层是初始过滤器集。目前的移动模型倾向于在一个完整的3x3卷积中使用32个滤波器来构建初始滤波器库进行边缘检测。通常这些过滤器是彼此的镜像。我们尝试减少滤波器的数量，并使用不同的非线性来尝试减少冗余。我们决定对这一层使用硬swish非线性，因为它表现得和其他被测试的非线性一样好。我们能够将滤波器的数量减少到16个，同时保持与使用ReLU或swish的32个滤波器相同的精度。这节省了额外的3毫秒和1000万MAdds。

### 5.2 非线性

&emsp;一种称为swish的非线性，当作为ReLU的替代变量时，它可以显著提高神经网络的准确性。非线性定义为

$$swish(x)=x\cdot\sigma(x)$$

&emsp;虽然这种非线性提高了精度，但在嵌入式环境中，它的成本是非零的，因为在移动设备上计算sigmoid函数要昂贵得多。我们用两种方法处理这个问题。

&emsp;1. 我们将sigmoid函数替换为它的分段线性硬模拟:$\frac{RELU6(x+3)}{6}$。小的区别是我们使用RELU6而不是自定义的剪裁常量（custom clipping constant）。类似地，swish的硬版本也变成了

$$h-swish[x]=x\frac{RELU6(x+3)}{6}$$

&emsp;最近在某篇文章中也提出了类似的hard-swish版本。下图显示了sigmoid和swish非线性的软、硬版本的比较。我们选择常量的动机是简单，并且与原始的平滑版本很好地匹配。在我们的实验中，我们发现所有这些函数的硬版本在精度上没有明显的差异，但是从部署的角度来看，它们具有多种优势。首先，几乎所有的软件和硬件框架上都可以使用ReLU6的优化实现。其次，在量化模式下，它消除了由于近似sigmoid的不同实现而引起的潜在数值精度损失。最后，即使优化了量化的sigmoid实现，其速度也比相应的ReLU慢得多。在我们的实验中，用量化模式下的swish替换h-swish使推理延迟增加了15%。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/sigmoid-swish.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">sigmoid和swish非线性及其他“硬”对应物</div>
</center>

&emsp;2. 随着我们深入网络，应用非线性的成本会降低，因为每层激活内存通常在分辨率下降时减半。顺便说一句，我们发现swish的大多数好处都是通过只在更深的层中使用它们实现的。因此，在我们的架构中，我们只在模型的后半部分使用h-swish。我们参照表1和表2来获得精确的布局。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/table1.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表1. MobileNetv3-Large规范。SE表示该块中是否存在压缩和激励。NL表示使用的非线性类型。这里，HS表示h-swish，RE表示ReLU。NBN表示没有批量规范化。s表示步长。</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/table2.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表2. MobileNetv3-Small规范。符号见表1。</div>
</center>

&emsp;即使有了这些优化，h-swish仍然会带来一些延迟成本。然而，正如我们在第6节中所展示的，对准确性和延迟的净影响是积极的，它为进一步的软件优化提供了一个场所：一旦平滑的Sigmoid被逐段线性函数取代，大部分开销都是内存访问，可以通过将非线性与前一层融合来消除。

#### 5.2.1 Large squeeze-and-excite

&emsp;在Mnasnet中，压缩-激发瓶颈的大小与卷积瓶颈的大小有关。取而代之的是，我们将它们全部替换为固定为拓展层通道数的1/4。我们发现这样做可以在适当增加参数数量的情况下提高精度，并且没有明显的延迟成本。

### 5.3 MobileNetV3定义

&emsp;MobileNetV3被定义为两个模型:MobileNetV3-Large和MobileNetV3-Small。这些模型分别针对高资源用例和低资源用例。通过将平台感知的NAS和NetAdapt用于网络搜索，并结合本节定义的网络改进，可以创建模型。我们的网络的完整规范见表1和表2。

## 6. 实验

&emsp;我们提供了实验结果来证明新的MobileNetV3模型的有效性。我们报告分类、检测和分割的结果。我们也报告各种消融研究，以阐明各种设计决策的影响。

### 6.1 分类

&emsp;由于已经成为标准，我们在所有分类实验中都使用ImageNet，并将准确度与各种资源使用度量(如延迟和乘法加法(MAdds))进行比较。

#### 6.1.1 训练设置

&emsp;我们在4x4 TPU Pod上使用momentum=0.9的标准tensorflow RMSPropOp-timizer进行同步训练。我们使用初始学习率为0.1，批大小为4096(每个芯片128张图像)，学习率衰减率为0.01/每3个epoch。我们使用0.8的dropout, l2权值衰减为1e-5，图像预处理与[Inception](https://arxiv.org/abs/1602.07261v2)相同。最后，我们使用衰减为0.9999的指数移动平均。我们所有的卷积层都使用批处理归一化层，平均衰减为0.99。
