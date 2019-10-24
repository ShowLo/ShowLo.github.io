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
&emsp;谷歌在2019年5月在arxiv上公开了MobileNetV3的论文，原论文见 [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)。

&emsp;下面先对论文做一个简单的翻译：

---

## 摘要
&emsp;我们展示了基于互补搜索技术和新颖架构设计相结合的下一代MobileNet。MobileNetV3通过结合硬件感知的网络架构搜索（NAS）和NetAdapt算法对移动手机的cpu进行调优，然后通过新的架构改进对其进行改进。本文开始探索自动化搜索算法和网络设计如何协同工作，利用互补的方法来提高整体STOA。通过这个过程，我们创建发布了两个新的MobileNet模型:MobileNetV3-Large和MobileNetV3-Small，它们分别针对高资源和低资源两种情况。然后将这些模型应用于目标检测和语义分割。针对语义分割（或任何密集像素预测）任务，我们提出了一种新的高效分割解码器Lite reduce Atrous Spatial Pyramid Pooling （LR-ASPP）。我们实现了移动分类、检测和分割的STOA。与MobileNetV2相比，MobileNetV3-Large在ImageNet分类上的准确率提高了3.2%，同时延迟降低了15%。与MobileNetV2相比，MobileNetV3-Small的准确率高4.6%，同时延迟降低了5%。MobileNetV3-Large检测速度比MobileNetV2快25%，在COCO检测上的精度大致相同。MobileNetV3-Large LR-ASPP的速度比MobileNetV2 R-ASPP快30%，在Cityscapes分割的精度类似。

## 1. 介绍
&emsp;高效的神经网络在移动应用程序中变得无处不在，从而实现全新的设备体验。它们也是个人隐私的关键推动者，允许用户获得神经网络的好处，而无需将数据发送到服务器进行评估。神经网络效率的进步不仅通过更高的精度和更低的延迟改善了用户体验，而且通过降低功耗有助于保持电池寿命。

&emsp;本文描述了我们开发MobileNetV3大型和小型模型的方法，以提供下一代高精度高效的神经网络模型来驱动设备上的计算机视觉。新的网络推动了STOA，并展示了如何将自动化搜索与新的体系结构进步结合起来，以构建有效的模型。

&emsp;本文的目标是开发最佳的移动计算机视觉架构，以优化在移动设备上的精度-延迟之间的权衡。为了实现这一点，我们引入了（1）互补搜索技术，（2）适用于移动设备的非线性的新高效版本，（3）新的高效网络设计，（4）一个新的高效分割解码器。我们提供了深入的实验，以证明每种技术在广泛的用例和移动手机上评估的有效性和价值。

&emsp;论文组织如下。我们从第2节中有关工作的讨论开始。第3节回顾了用于移动模型的高效构建块。第4节回顾了体系结构搜索以及MnasNet和NetAdapt算法的互补性。第5节描述了通过联合搜索提高模型效率的新型架构设计。第6节为分类、检测和分割提供了广泛的实验，以证明其有效性并了解不同元素的贡献。第7节包含结论和今后的工作。

## 2. 相关工作
&emsp;设计深度神经网络结构来实现精度和效率之间的最优平衡是近年来一个活跃的研究领域。无论是新颖的手工结构还是算法神经结构搜索，都在这一领域发挥了重要作用。

&emsp;SqueezeNet广泛使用1x1卷积与挤压和扩展模块，主要集中于减少参数的数量。最近的工作将关注点从减少参数转移到减少操作的数量（MAdds）和实际测量的延迟。MobileNetV1采用深度可分离卷积，大大提高了计算效率。MobileNetV2在此基础上进行了扩展，引入了一个具有反向残差和线性瓶颈的资源高效块。ShuffleNet利用组卷积和通道洗牌操作进一步减少MAdds。CondenseNet在训练阶段学习组卷积，以保持层与层之间有用的紧密连接，以便特征重用。ShiftNet提出了与逐点卷积交织的移位操作，以取代昂贵的空间卷积。

&emsp;为了使体系结构设计过程自动化，首先引入了增强学习（RL）来搜索具有竞争力的精度的高效体系结构。一个完全可配置的搜索空间可能会以指数级增长且难以处理。因此，早期的架构搜索工作主要关注单元级结构搜索，并且在所有层中重用相同的单元。最近，有文章探索了一个块级分层搜索空间，允许在网络的不同分辨率块上使用不同的层结构。为了降低搜索的计算成本，在一些研究中使用了基于梯度优化的可微体系结构搜索框架。针对现有网络适应受限移动平台的问题，有些文章中提出了更高效的自动化网络简化算法。

&emsp;量化是另一个重要的补充努力，通过降低精度的算法来提高网络效率。最后，知识蒸馏提供了一种附加的补充方法，在大型“教师”网络的指导下生成精确的小型“学生”网络。

## 3. 有效的移动构建块（Efficient Mobile Building Blocks）

&emsp;移动模型已经建立在越来越高效的构建块之上。MobileNetV1引入深度可分离卷积作为传统卷积层的有效替代。深度可分卷积通过将空间滤波与特征生成机制分离，有效地分解了传统卷积。深度可分卷积由两个独立的层定义:用于空间滤波的轻量级深度卷积和用于特征生成的较重的1x1逐点卷积。

&emsp;MobileNetV2引入了线性瓶颈和反向残差结构，以便利用问题的低秩性质使层结构更加有效。这个结构如图3所示，由1x1拓展卷积（注：即提升通道数）、深度卷积和1x1投影层定义。当且仅当输入和输出具有相同数量的通道时，才用残差链接连接它们。该结构在保持输入和输出的紧凑表示的同时，在内部扩展到高维特征空间，以增加每个通道非线性转换的表现力。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/figure3.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图3. MobileNetV2层（反向残差和线性瓶颈）。每个块由窄的输入和输出（瓶颈）组成，它们没有非线性，然后扩展到一个更高维度的空间并投影到输出。残差连接瓶颈（而不是扩展）</div>
</center>

&emsp;MnasNet建立在MobileNetV2结构上，在瓶颈结构中引入基于压缩和激励（squeeze and excitation）的轻量级注意力模块。注意，与[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)中提出的基于ResNet的模块相比，压缩和激励模块集成在不同的位置。此模块位于拓展的深度滤波器之后，以便注意力机制应用于最大的表示（representation），如图4所示。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/figure4.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图4. MobileNetV2+Squeeze-and-Excite。与SENet相比，我们在残差层施加压缩和激励。 我们根据层的不同使用不同的非线性，详见5.2节</div>
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

&emsp;（a）生成一组新proposal。每一个proposal都代表了一个架构的修改，与前一步相比，该架构至少能降低延迟$\delta$。

&emsp;（b）对于每一个proposal，我们使用前一步骤的预先训练的模型，并填充新提出的架构，适当地截断和随机初始化缺失的权重。对每个proposal进行T个步骤的微调，以获得对精度的粗略估计。

&emsp;（c）根据某种标准选择最佳proposal。

&ensp;3. 重复前面的步骤，直到达到目标延迟。

&emsp;在NetAdapt中，度量是为了最小化精度的变化。我们修改了这个算法，使延迟变化和精度变化的比例最小化。也就是说，对于每个NetAdapt步骤中生成的所有proposal，我们选择一个使得$\frac{\Delta Acc}{\Delta Latency}$最大的proposal，其中$\Delta Latency$满足2（a）中的约束。 直觉告诉我们，由于我们的proposal是离散的，所以我们更喜欢最大化权衡（trade-off）曲线斜率的proposal。

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

&emsp;一旦降低了该特征生成层的成本，就不再需要以前的瓶颈投影层来减少计算量。该观察允许我们删除前一个瓶颈层中的投影和滤波层，从而进一步降低计算复杂度。原始阶段和优化后的最后一个阶段如图5所示。有效的最后一个阶段将延迟减少了10毫秒，即15%的运行时间，并将操作数量减少了3000万个MAdds，几乎没有损失精度。 第6节包含了详细的结果。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/figure5.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图5. 原来的和有效的最后一个阶段的比较。这个更有效的最后阶段能够在不损失精度的情况下，在网络的末端丢弃三个expensive的层</div>
</center>

&emsp;另一个expensive的层是初始过滤器集。目前的移动模型倾向于在一个完整的3x3卷积中使用32个滤波器来构建初始滤波器库进行边缘检测。通常这些过滤器是彼此的镜像。我们尝试减少滤波器的数量，并使用不同的非线性来尝试减少冗余。我们决定对这一层使用硬swish非线性，因为它表现得和其他被测试的非线性一样好。我们能够将滤波器的数量减少到16个，同时保持与使用ReLU或swish的32个滤波器相同的精度。这节省了额外的3毫秒和1000万MAdds。

### 5.2 非线性

&emsp;一种称为swish的非线性，当作为ReLU的替代变量时，它可以显著提高神经网络的准确性。非线性定义为

$$swish(x)=x\cdot\sigma(x)$$

&emsp;虽然这种非线性提高了精度，但在嵌入式环境中，它的成本是非零的，因为在移动设备上计算sigmoid函数要昂贵得多。我们用两种方法处理这个问题。

&emsp;1. 我们将sigmoid函数替换为它的分段线性硬模拟:$\frac{RELU6(x+3)}{6}$。小的区别是我们使用RELU6而不是自定义的剪裁常量（custom clipping constant）。类似地，swish的硬版本也变成了

$$h-swish[x]=x\frac{RELU6(x+3)}{6}$$

&emsp;最近在某篇文章中也提出了类似的hard-swish版本。图6显示了sigmoid和swish非线性的软、硬版本的比较。我们选择常量的动机是简单，并且与原始的平滑版本很好地匹配。在我们的实验中，我们发现所有这些函数的硬版本在精度上没有明显的差异，但是从部署的角度来看，它们具有多种优势。首先，几乎所有的软件和硬件框架上都可以使用ReLU6的优化实现。其次，在量化模式下，它消除了由于近似sigmoid的不同实现而引起的潜在数值精度损失。最后，在实践中，h-swish可以作为一个分段函数来实现，以减少内存访问次数，从而大大降低了延迟。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/figure6.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图6. sigmoid和swish非线性及其他“硬”对应物</div>
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

&emsp;即使有了这些优化，h-swish仍然会带来一些延迟成本。然而，正如我们在第6节中所演示的那样，当使用基于分段函数的优化实现时，对精度和延迟的净影响是正的，没有优化，并且非常显著。

#### 5.3 Large squeeze-and-excite

&emsp;在Mnasnet中，压缩-激发瓶颈的大小与卷积瓶颈的大小有关。取而代之的是，我们将它们全部替换为固定为拓展层通道数的1/4。我们发现这样做可以在适当增加参数数量的情况下提高精度，并且没有明显的延迟成本。

### 5.4 MobileNetV3定义

&emsp;MobileNetV3被定义为两个模型:MobileNetV3-Large和MobileNetV3-Small。这些模型分别针对高资源用例和低资源用例。通过将平台感知的NAS和NetAdapt用于网络搜索，并结合本节定义的网络改进，可以创建模型。我们的网络的完整规范见表1和表2。

## 6. 实验

&emsp;我们提供了实验结果来证明新的MobileNetV3模型的有效性。我们报告分类、检测和分割的结果。我们也报告各种消融研究，以阐明各种设计决策的影响。

### 6.1 分类

&emsp;由于已经成为标准，我们在所有分类实验中都使用ImageNet，并将准确度与各种资源使用度量（如延迟和乘法加法（MAdds））进行比较。

#### 6.1.1 训练设置

&emsp;我们在4x4 TPU Pod上使用momentum=0.9的标准tensorflow RMSPropOp-timizer进行同步训练。我们使用初始学习率为0.1，批大小为4096（每个芯片128张图像），学习率衰减率为0.01/每3个epoch。我们使用0.8的dropout, l2权值衰减为1e-5，图像预处理与[Inception](https://arxiv.org/abs/1602.07261v2)相同。最后，我们使用衰减为0.9999的指数移动平均。我们所有的卷积层都使用批处理归一化层，平均衰减为0.99。

#### 6.1.2 测量设置

&emsp;为了测量延迟，我们使用标准的谷歌Pixel手机，并通过标准的TFLite基准测试工具运行所有网络。我们在所有测量中都使用单线程大内核。我们没有报告多核推理时间，因为我们发现这种设置对移动应用程序不太实用。我们为tensorflow lite提供了一个h-swish的原子操作符，在最新版本中它是默认的。我们在图9展示了优化后的h-swish的影响。

### 6.2 结果

&emsp;如图1所示，我们的模型优于目前的STOA，如MnasNet、ProxylessNas和MobileNetV2。我们在表3中报告了不同Pixel手机上的浮点性能。我们在表4中包括量化结果。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/figure1.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图1. Pixel 1 延迟与ImageNet上top-1准确率之间的权衡。所有模型都使用输入分辨率224。V3 large和V3 small使用乘数0.75、1和1.25来显示最佳边界。使用TFLite在同一设备的单个大内核上测量所有延迟。MobileNetV3-Small和Large是我们提议的下一代移动模型</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/figure2.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图2. MAdds和top-1准确率之间的权衡。这允许比较针对不同硬件或软件框架的模型。所有MobileNetV3用于输入分辨率224，并使用乘数0.35、0.5、0.75、1和1.25。其他分辨率见第6节</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/table3.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表3. Pixel系列手机的浮点性能（“P-n”表示Pixel-n手机）。所有延迟都以毫秒为单位，并使用批大小为1的单个大核心来测量。top-1精度是在ImageNet上测得的。</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/table4.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表4. 量化性能。所有延迟都以毫秒为单位。推理延迟是在各自的Pixel 1/2/3设备上使用单个大核心来测量的</div>
</center>

&emsp;在图7中，我们展示了MobileNetV3性能权衡作为乘数和分辨率的函数。请注意，MobileNetV3-Small的性能比MobileNetV3-Large的性能好得多，其乘法器缩放到与性能匹配的倍数接近3%。另一方面，分辨率提供了比乘数更好的权衡。但是，需要注意的是，分辨率通常是由问题决定的（例如分割和检测问题通常需要更高的分辨率），因此不能总是用作可调参数。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/figure7.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图7. 作为不同乘数和分辨率函数的MobileNetV3性能。在我们的实验中，我们使用了固定分辨率为224同时乘数为0.35、0.5、0.75、1.0和1.25，以及固定深度乘数为1.0同时分辨率为96、128、160、192、224和256。top-1精度是在ImageNet上测量得到的，延迟以ms为单位。</div>
</center>

#### 6.2.1 模型简化测试（Ablation study）

##### 非线性的影响

&emsp;在表5中，我们研究了在何处插入h-swish非线性的选择，以及使用优化实现相对于简单实现的改进。可以看出，使用优化的h-swish实现可以节省6ms（超过运行时间的10%）。优化后的h-swish只比传统的ReLU多增加1ms。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/table5.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表5. 非线性对MobileNetV3-Large的影响。在h-swish@N中，N表示启用h-swish的第一层的通道数。第三列显示了没有优化h-swish的运行时间。top-1精度是在ImageNet上测量得到的，延迟以ms为单位。</div>
</center>

&emsp;图8显示了基于非线性选择和网络宽度的有效边界。MobileNetV3在网络中间使用h-swish，并且明显控制着ReLU。有趣的是，向整个网络添加h-swish略胜于扩展网络的插值边界。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/figure8.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图8. h-swish与ReLU对优化和非优化h-swish延迟的影响。曲线显示了使用深度乘数的边界。注意，将h-swish放置在具有80个或更多通道的所有层上（V3）为优化的h-swish和非优化的h-swish提供了最佳的权衡。top-1精度是在ImageNet上测量得到的，延迟以ms为单位。</div>
</center>

##### 其他组件的影响

&emsp;在图9中，我们展示了不同组件的引入是如何沿着延迟/准确率曲线移动的。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/figure9.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图9. 向网络体系结构中添加单个组件的影响。进展过程是向上和向左移动的。</div>
</center>

### 6.3 检测

&emsp;我们使用MobileNetV3作为SSDLite中骨干功能提取器的替代品，并与COCO数据集上的其他骨干网络进行比较。

&emsp;跟MobileNetV2一样，我们将第一层SSDLite附加到输出步长为16（注：比如输入分辨率为224，这里就指的是分辨率变为14）的最后一个特征提取器层，并将第二层SSDLite附加到输出步长为32的最后一个特征提取器层。根据检测文献，我们将这两个特征提取层分别称为C4和C5。对于MobileNetV3-Large, C4是第13个瓶颈块的拓展层。对于MobileNetV3-Small, C4是第9个瓶颈块的拓展层。对于这两个网络，C5都是池化之前的一层。

&emsp;此外，我们还将C4和C5之间的所有特征层的通道数减少了2倍。这是因为MobileNetV3的最后几层被调优为输出1000个类，当传输到只有90个类的COCO时，这可能是多余的。

&emsp;表6给出了COCO测试集的结果。随着通道减少，MobileNetV3-Large比具有几乎相同mAP的MobileNetV2快25%。与MobileNetV2和Mnasnet相比，通道减少的MobileNetV3-Small在相近延迟下的mAP也高出2.4和0.5。对于这两个MobileNetV3模型，通道减少技巧有助于在不丢失mAP的情况下减少大约15%的延迟，这表明ImageNet分类和COCO目标检测可能更喜欢不同的特征提取器形状。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/table6.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表6. 不同骨架的SSDLite在COCO测试集上的目标检测结果。†:C4和C5之间的块中的通道减少了2倍</div>
</center>

### 6.4 语义分割

&emsp;在本小节中，我们使用MobileNetV2和提出的MobileNetV3作为移动语义分割的网络骨架。此外，我们比较了两个分割头。第一个是在MobileNetV2中提出的R-ASPP。R-ASPP是一种无源空间金字塔池化模块的简化设计，它只采用由1×1卷积和全局平均池化操作组成的两个分支。在这项工作中，我们提出了另一种轻型分割头，称为Lite R-ASPP（或LR-ASPP），如图10所示。Lite R-ASPP是对R-ASPP的改进，它部署全局平均池的方式类似于压缩-激励模块，其中我们使用了一个具有较大步长的较大池化核（以节省一些计算），并且模块中只有一个1×1卷积。我们对MobileNetV3的最后一个块应用空洞卷积（atrous convolution）来提取更密集的特性，并进一步从底层特性添加一个跳跃（skip）连接来捕获更详细的信息。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/figure10.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图10. 在MobileNetV3的基础上，提出的分割头Lite R-ASPP提供了快速的语义分割结果，同时混合了多种分辨率的特征。</div>
</center>

&emsp;我们使用度量mIOU对Cityscapes数据集进行了实验，只使用了“fine”注释。 我们采用与MobileNetV2相同的训练方案。 我们所有的模型都是从零开始训练，没有在ImageNet上进行预训练，并使用单尺度输入进行评估。 与目标检测类似，我们发现我们可以在不显著降低性能的情况下，将网络骨干最后一块的通道减少2倍。 我们认为这是因为骨干网络是为具有1000类的ImageNet图像分类设计的，而Cityscapes只有19类，这意味着骨干网络中存在一定的通道冗余。

&emsp;我们在表7中报告了我们的Cityscapes验证集结果。如表所示，我们观察到：（1）将最后一个网络骨干块中的通道减少2倍，在保持类似性能的同时显著提高了速度（第1行与第2行，第5行与第6行）；（2）提出的分割头LR-ASSP略快于R-ASSP，而性能有所改善（第2行与第3行，第6行与第7行）；（3）将分割头中的滤波器从256减少到128可以提高速度，但性能稍差（第3行与第4行，第7行与第8行）；（4）当采用相同的设置时，MobileNetV3模型的变体获得了相似的性能，但比MobileNetV2对应的模型稍快（第1行对第5行，第2行对第6行，第3行对第7行，第4行对第8行）；（5）MobileNetV3-Small与MobileNetV2-0.5的性能相似，但速度更快；（6）MobileNetV3-Small明显优于MobileNetV2-0.35，但速度相似。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/table7.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表7. Cityscapes验证集的语义分割结果。RF2：将最后一个块中的滤波器减少2倍。V2 0.5和V2 0.35是MobileNetV2，深度乘数分别为0.5和0.35。SH：分割头，其中$\times$使用R-ASSP，而$\checkmark$使用建议的LR-ASSP。F：分割头中使用的滤波器数量。CPU（f）：在Pixel 3（浮点）用全分辨率输入（即1024×2048）的单个大核心上测量的CPU时间。CPU（h）：测量的CPU时间，用半分辨率输入（即512×1024）。第8行和第11行是我们的MobileNetv3细分候选</div>
</center>

&emsp;表8显示了我们的Cityscapes测试集结果。我们的以MobileNetV3为网络骨干的分割模型，性能分别比ESPNetv2、CCC2和ESPNetv1提高6.4%、10.6%、12.3%，同时在MAdds方面更快。在MobileNetV3的最后一个块中，当不使用空洞卷积提取密集特征图时，性能略有下降0.6%，但速度提高到1.98B（对于半分辨率输入），分别是ESPNetv2、CCC2和ESPNetv1快1.36倍、1.59倍和2.27倍。此外，我们的以MobileNetV3-Small作为骨干网络的模型仍然优于它们至少一个2.1%的健壮差额（healthy margin）。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/table8.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表8. Cityscapes测试集的语义分割结果。OS ：输出步长，输入图像空间分辨率与骨干输出分辨率之比。当OS=16时，在骨干的最后一块应用空洞卷积。当OS=32时，不使用空洞卷积。MAdds（f）：在全分辨率输入（即1024×2048）时测量的乘法-加法运算量。MADDS（h）：在半分辨率输入（即512×1024）时测量的乘法-加法运算量。CPU（f）：在Pixel 3（浮点）于全分辨率输入（即1024×2048）时的单个大核心上测量的CPU时间。CPU（h）：在半分辨率输入（即512×1024）时测量的CPU时间。†:MAdds是根据Espnetv2估计的，其中仅为输入尺寸224×224提供MAdds</div>
</center>

## 7. 结论与未来工作

&emsp;本文介绍了MobileNetV3的Large和Small模型，展示了移动分类、检测和分割的SOTA。我们描述了我们的努力，以利用多种网络架构搜索算法，以及先进的网络设计，以交付下一代移动模型。我们还展示了如何采用像swish这样的非线性，并以量化友好和有效的方式应用压缩和激励，将它们作为有效的工具引入到移动模型领域。我们还介绍了一种称为LR-ASSP的轻量级分割解码器。尽管如何最好地将自动搜索技术与人类的直觉结合起来仍然是一个悬而未决的问题，但我们很高兴地提出了这些初步的积极结果，并将继续在未来的工作中继续改进这些方法。

## 附录A 不同分辨率和乘数的性能表

&emsp;我们在表9中给出了包含乘法-加法、准确率、参数量和延迟的详细表。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/table9.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表9. Large和Small V3模型的浮点性能。P-1对应于Pixel 1上的大单核性能</div>
</center>
