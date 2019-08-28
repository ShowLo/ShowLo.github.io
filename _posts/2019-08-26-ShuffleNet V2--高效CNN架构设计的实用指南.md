---
layout:     post
title:      ShuffleNet V2--高效CNN架构设计的实用指南
subtitle:   ShuffleNet V2--Practical Guidelines for Efficient CNN Architecture Design
date:       2019-08-26
author:     CJR
header-img: img/2019-08-26-ShuffleNet V2--高效CNN架构设计的实用指南/post-bg.jpg
catalog: true
mathjax: true
tags:
    - Lightweight Network
    - ShuffleNet V2
    - CNN
---

# ShuffleNet

&emsp;前一篇文章介绍了ShuffleNet，然后Face++在2018年又发布了其升级版--ShuffleNet V2，文章也发表在了ECCV 2018上，原文可见[ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)。

---

## 摘要

&emsp;目前，神经网络体系结构设计主要以计算复杂度的间接度量为指导，即FLOPs。然而，速度等直接指标也取决于其他因素，如内存访问成本和平台特性。因此，这项工作建议评估目标平台上的直接度量，而不仅仅是考虑FLOPs。在一系列对照实验的基础上，提出了一些实用的网络设计准则。相应地，提出了一种新的体系结构，称为ShuffleNet V2。综合消融实验验证了该模型在速度和精度上的权衡是最先进的。

## 1. 引言

&emsp;深度卷积神经网络（CNNs）的体系结构已经发展多年，变得更加精确和快速。自AlexNet的里程碑式工作以来，ImageNet的分类精度得到了显著提高，其中包括VGG、GoogLeNet、ResNet、DenseNet、ResNeXt、SE-Net和自动神经架构搜索等新结构。

&emsp;除了精确度之外，计算复杂度也是一个重要的考虑因素。现实世界中的任务通常旨在在有限的计算预算下获得最佳的精度，这是由目标平台（例如，硬件）和应用场景（例如，自动驾驶需要较低的延迟）给出的。这激发了一系列致力于轻量级架构设计和更好的速度-精度权衡的工作，包括Xception、MobileNet、MobileNet V2、ShuffleNet和CondenseNet等等。分组卷积和深度卷积在这些工作中是至关重要的。

&emsp;为了度量计算复杂性，一个广泛使用的度量是浮点运算的数量，即FLOPs。然而，FLOPs是一个间接指标。这是一个近似，但通常不等于我们真正关心的直接度量，比如速度或延迟。这种差异在以前的工作中已经注意到了。例如，MobileNet V2比NASNET-A快得多，但它们也有类似的FLOPs。这一现象在图1(c)和(d)中得到了进一步的体现，它们表明具有相似FLOPs的网络具有不同的速度。因此，仅用FLOPs作为计算复杂度的度量是不够的，可能会导致次优设计。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-08-26-ShuffleNet V2--高效CNN架构设计的实用指南/figure1.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图1. 测量精度（ImageNet分类的验证集）、两个硬件平台上四个网络架构的速度和FLOPs，具有四个不同级别的计算复杂性。（a，c）GPU结果，批大小=8。（b，d）ARM结果，批大小=1。在所有情况下，我们提出的ShuffleNet V2均为最有效的算法，都位于右上角。</div>
</center>

&emsp;间接（FLOPs）和直接（速度）指标之间的差异可以归结为两个主要原因。首先，一些对速度有很大影响的重要因素没有被考虑到。其中一个因素就是内存访问成本（MAC）。在某些操作（如组卷积）中，这样的开销占运行时间的很大一部分。它可能成为计算能力强的设备（如gpu）的瓶颈。 在网络体系结构设计中不应简单地忽略这一成本。另一个是并行度。在相同的FLOPs下，并行度高的模型可能比并行度低的模型快得多。

&emsp;其次，根据平台的不同，具有相同FLOPs的操作可能会有不同的运行时间。例如，张量分解在早期的工作中被广泛应用来加速矩阵的乘法。然而，最近的工作发现，[Accelerating very deep convolutional networks for classification and detection]中的分解在GPU上更慢，尽管它减少了75%的FLOPs。我们调查了这个问题，发现这是因为最新的CUDNN库是专门针对3×3卷积优化过的，我们不能肯定认为3×3卷积比1×1卷积慢9倍。

&emsp;根据这些观察结果，我们建议在进行有效的网络架构设计时应考虑两个原则。首先，应该使用直接度量（例如，速度），而不是间接度量（例如，FLOPs)。其次，应该在目标平台上评估这些指标。

&emsp;在这项工作中，我们遵循这两个原则，并提出了一个更有效的网络架构。在第2节中，我们首先分析了两种具有代表性的最先进网络的运行时性能。在此基础上，我们提出了四种有效的网络设计准则，这些准则不仅仅是考虑FLOPs。虽然这些指导原则是独立于平台的，但是我们通过专门的代码优化在两个不同的平台（GPU和ARM）上执行一系列对照实验来验证它们，确保我们的结论是最先进的。

&emsp;在第三节中，我们根据准则设计了一个新的网络结构。因为它的灵感来自ShuffleNet，所以它被称为ShuffleNet V2。通过第4节的综合验证实验，证明了该方法在两种平台上都比以前的网络更快、更准确。图1(a)和(b)给出了比较的概述。例如，在计算复杂度预算为40M FLOPs的情况下，ShuffleNet V2的准确率分别比ShuffleNet V1和MobileNet V2高3.5%和3.7%。

## 2. 有效的网络设计实用指南

&emsp;我们的研究是在两个广泛采用的具有行业级优化的CNN库的硬件上进行的。我们注意到我们的CNN库比大多数开源库更有效。因此，我们保证我们的观察和结论是可靠的，对工业实践具有重要意义。

- *GPU.*&emsp;使用单个Nvidia GeForce GTX 1080Ti。卷积库为CUDNN 7.0。我们还激活了CUDNN的基准函数，分别为不同的卷积选择最快的算法。

- *ARM.*&emsp;高通骁龙810。我们使用一个高度优化的基于Neon的实现。一个线程用于评估。

&emsp;其他设置包括：打开全优化选项（例如张量融合，用于减少小操作的开销）。输入图像大小为224×224。每个网络被随机初始化并评估100次。使用平均运行时间。

&emsp;为了开始我们的研究，我们分析了两种最先进的网络ShuffleNet V1和MobileNet V2的运行时性能。它们在图像分类任务中既高效又准确。它们都广泛应用于手机等低端设备。虽然我们只分析了这两个网络，但我们注意到它们代表了当前的趋势。它们的核心是分组卷积和深度卷积，它们也是其他先进网络的关键组件，如ResNeXt、Xception、MobileNet和CondenseNet。

&emsp;根据不同的操作分解整个运行时间，如图2所示。我们注意到FLOPs度量只占卷积部分。虽然这部分花费的时间最多，但是其他操作包括数据I/O、数据重排和元素级操作（AddTensor、ReLU等）也会占用相当多的时间。因此，FLOPs对实际运行时的估计不够准确。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-08-26-ShuffleNet V2--高效CNN架构设计的实用指南/figure2.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图2. 在两个具有代表性的最先进的网络架构上的运行时间分解：ShuffeNet V1（1×，g=3）和MobileNet V2（1×）。</div>
</center>

&emsp;基于这一观察，我们从几个不同的方面对运行时间（或速度）进行了详细的分析，并为高效的网络架构设计得出了几个实用的指导原则。

**（G1）通道宽度相等，内存访问成本（MAC）最小。** 现代网络通常采用深度可分卷积，其中逐点卷积（即1×1卷积）占了复杂度的绝大部分。我们研究了1×1卷积的核形状。其形状由两个参数指定：输入通道数$c_{1}$和输出通道数$c_{2}$。设$h$和$w$为特征图的空间大小，1×1卷积的FLOPs为$B=hwc_{1}c_{2}$。

&emsp;为了简单起见，我们假设计算设备中的缓存足够大，可以存储全部特征图和参数。因此，内存访问成本（MAC）或内存访问操作的数量是$MAC=hw(c_{1}+c_{2})+c_{1}c_{2}$。注意，这两项分别对应于输入/输出特征图和卷积核权重的内存访问。

&emsp;从均值不等式，我们得到

$$
MAC\ge 2\sqrt{hwB}+\frac{B}{hw} \tag{1}
$$

&emsp;因此，MAC有一个由FLOPs给出的下界。输入通道和输出通道数目相等时，到达下界。

&emsp;结论是理论性的。实际上，许多设备上的缓存不够大。此外，现代计算库通常采用复杂的阻塞策略来充分利用缓存机制。因此，实际MAC可能会偏离理论MAC。为了验证上述结论，我们进行了如下实验。通过重复叠加10个构建块来构建基准网络。每个块包含两个卷积层。第一个包含$c_{1}$输入通道和$c_{2}$输出通道，第二个则不然。

&emsp;表1通过改变$c_{1}:c_{2}$的比例来报告运行速度，同时固定总FLOPs。很明显，当$c_{1}:c_{2}$接近$1:1$时，MAC变得更小，网络评估速度更快。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-08-26-ShuffleNet V2--高效CNN架构设计的实用指南/table1.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表1. 准则1的验证实验。测试了四种不同的输入/输出通道数比（$c_{1}$和$c_{2}$），而四种比率下的总FLOPs是通过改变通道数来固定的。输入图像大小为56×56。</div>
</center>

**（G2）过多的分组卷积增加了MAC。** 分组卷积是现代网络结构的核心。它通过将所有通道之间的密集卷积变为稀疏（仅在通道组内）来降低计算复杂度（FLOPs）。一方面，它允许在给定固定FLOPs的情况下使用更多的通道，并增加了网络容量（从而提高了准确性）。然而，另一方面，通道数量的增加导致更多的MAC。

&emsp;形式上，根据G1和公式$(1)$中的符号，1×1分组卷积的MAC和FLOPs之间的关系为

$$
MAC=hw(c_{1}+c_{2})+\frac{c_{1}c_{2}}{g}=hwc_{1}+\frac{Bg}{c_{1}}+\frac{B}{hw} \tag{2}
$$

&emsp;其中g为组数，$B=hwc_{1}c_{2}/g$为FLOPs。可以看出，给定固定的输入形状$c_{1}\times h\times w$，计算量B, MAC随着g的增大而增大。

&emsp;为了研究在实际应用中的影响，通过叠加10个分组逐点卷积层，建立了一个基准网络。表2报告了在固定总FLOPs时使用不同组数的运行速度。很明显，使用大组数会显著降低运行速度。例如，在GPU上使用8组比使用1组（标准密集卷积）慢两倍多，在ARM上慢30%。这主要是由于MAC的增加。我们注意到，我们的实现经过了特别的优化，比一组一组地计算卷积要快得多。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-08-26-ShuffleNet V2--高效CNN架构设计的实用指南/table2.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表2. 准则2的验证实验。测试了组数g的四个值，而四个值下的总FLOPs是通过改变总通道数c来固定的，输入图像大小为56×56。</div>
</center>

&emsp;因此，我们建议根据目标平台和任务仔细选择组数。使用较大的组号是不明智的，因为这可能使使用更多的通道成为可能，因为精度提高的好处很容易被快速增长的计算成本所抵消。

**（G3）网络碎片化降低了并行度。** 在GoogLeNet系列和自动生成体系结构中，每个网络块都广泛采用“多路径”结构。使用许多小的操作符（这里称为“片段操作符”），而不是一些大的操作符。例如，在NASNET-A中，碎片操作符的数量（即一个构建块中单个卷积或池化操作的数量）为13。相反，在像ResNet这样的常规结构中，这个数字是2或3。

&emsp;虽然这种碎片化的结构已被证明有利于提高精度，但它可能会降低效率，因为它对GPU等具有强大并行计算能力的设备不友好。它还引入了额外的开销，比如核启动和同步。

&emsp;为了量化网络碎片对效率的影响，我们评估了一系列具有不同碎片程度的网络块。具体来说，每个构建块由1到4个1×1的卷积组成，这些卷积按顺序或并行排列。块结构如附录所示。每个块重复堆叠10次。表3中的结果表明，GPU上的碎片化显著降低了速度，例如4-碎片结构比1-碎片慢3倍。在ARM上，减速相对较小。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-08-26-ShuffleNet V2--高效CNN架构设计的实用指南/table3.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表3. 准则3的验证实验。c表示1-fragment的通道数。调整其他碎片结构中的通道数，使FLOPs与1-fragment相同。输入图像大小为56×56。</div>
</center>

**（G4）元素级（element-wise）操作是不可忽略的。** 如图2所示，在轻量级模型中，元素操作占用了相当多的时间，尤其是在GPU上。这里的元素级操作符包括ReLU、AddTensor、AddBias等。他们有小的FLOPs，但相对较重的MAC。特别地，我们还将深度卷积看作是一种元素级操作，因为它也具有很高的MAC/FLOPs比。

&emsp;为了验证，我们实验了ResNet中的“瓶颈”单元（1×1 conv + 3×3 conv + 1×1 conv, ReLU + shortcut连接）。分别删除ReLU和shortcut操作。表4中报告了不同变体的运行时间。去除ReLU和shortcut后，GPU和ARM均获得了20%左右的加速。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-08-26-ShuffleNet V2--高效CNN架构设计的实用指南/table4.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表4. 准则4的验证实验。RELU和shortcut操作分别从“瓶颈”单元中删除。c是单元中的通道数。该单元重复堆叠10次，以对速度进行基准测试。</div>
</center>

**结论和讨论。** 基于上述指导思想和实证研究，我们认为有效的网络架构应该(1)使用“均衡”卷积（通道宽度相等）；(2)注意使用分组卷积的成本；(3)降低碎片化程度；(4)减少元素级操作。这些理想的特性依赖于平台特性（例如内存操作和代码优化），这些特性超出了理论FLOPs。在实际的网络设计中应考虑这些因素。

&emsp;轻量级神经网络体系结构的最新进展大多基于FLOPs度量，没有考虑上述特性。例如，ShuffleNet V1严重依赖于分组卷积（违反G2）和类似瓶颈的构建块（违反G1）。MobileNet V2使用了一个违反G1的反向瓶颈结构。它在“厚”特征图上使用深度卷积和ReLUs。这违反了G4。自动生成的结构高度碎片化，违反G3。

## 3. ShuffleNet V2：一个高效的架构

**回顾ShuffleNet V1。** ShuffleNet是一种最先进的网络架构。被广泛应用于低端设备，如手机。它激励着我们的工作。因此，本文首先对其进行了回顾和分析。

&emsp;根据[ShuffleNet]，轻量级网络的主要挑战是，在给定的计算预算（FLOPs）下，只能负担得起有限数量的特征通道。为了在不显著增加FLOPs的情况下增加通道数量，[ShuffleNet]采用了两种技术：分组逐点积和瓶颈式结构。然后引入“通道重排”操作，使不同的通道组之间能够进行信息通信，从而提高准确性。构建块如图3(a)和(b)所示。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-08-26-ShuffleNet V2--高效CNN架构设计的实用指南/figure3.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图3. ShuffleNet V1和这项工作的构建块。（a）：基本ShuffleNet单元；（b）空间下采样（2倍）的ShuffleNet单元；（c）我们的基本单元；（d）我们的空间下采样（2倍）的单元。DWConv：深度卷积。GConv：分组卷积。</div>
</center>

&emsp;如第2节所述，分组逐点卷积和瓶颈结构都增加了MAC（G1和G2）。这一成本是不可忽视的，特别是对轻量级模型。此外，使用太多组违反了G3。shortcut连接中的元素级“Add”操作也是不可取的（G4）。因此，为了获得较高的模型容量和效率，关键问题是如何在不使用密集卷积和过多组的情况下保持大量和等宽的通道。

**通道分割和ShuffleNet V2。** 为了实现上述目的，我们引入了一个简单的操作符，称为通道分割。如图3(c)所示。在每个单元的开始，将$c$特征通道的输入分为$c-c'$和$c'$通道两个分支。遵循G3，一个分支仍然作为恒等分支。另一个分支由三个具有相同输入和输出通道的卷积组成，以满足G1。与[ShuffleNet]不同，这两个1×1卷积不再是组-级（group-wise）的。部分原因是遵循G2，部分原因是分割操作已经生成了两个组。

&emsp;卷积后，将两个分支连接起来。因此，通道数保持不变（G1）。然后使用与[ShuffleNet]中相同的“通道重排”操作来支持两个分支之间的信息通信。

&emsp;重排之后，下一个单元开始。注意，ShuffleNet V1中的“Add”操作不再存在。像ReLU和深度卷积这样的元素级操作只存在于一个分支中。此外，连续的三个元素操作（“Concat”、“通道洗牌”和“通道分割”）合并为一个元素级操作。G4认为这些变化是有益的。

&emsp;对于空间下采样，对单元进行了微调，如图3(d)所示。移除通道分割操作符。因此，输出通道的数量增加了一倍。

&emsp;提议的构建块(c)(d)以及生成的网络称为ShuffleNet V2。基于以上分析，我们得出结论，该体系结构设计是高效的，因为它遵循了所有的准则。

&emsp;构建块被反复堆叠，以构建整个网络。为了简单起见，我们设$c'=c/2$。总体网络结构与ShuffleNet V1相似，如表5所示。唯一的区别是：在全局平均池化之前添加了一个额外的1×1卷积层来混合特征，这在ShuffleNet V1中是不存在的。与[ShuffleNet]类似，将每个块中的通道数进行缩放，生成不同复杂度的网络，标记为0.5×、1×等。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-08-26-ShuffleNet V2--高效CNN架构设计的实用指南/table5.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表5. ShuffleNet V2的总体架构，适用于四个不同级别的复杂度。</div>
</center>

**网络精度分析。** 为ShuffleNet v2不仅高效，而且准确。有两个主要原因。首先，每个构建块的高效率使得使用更多的特征通道和更大的网络容量成为可能。

&emsp;其次，在每个块中，有一半的特性通道（当$c'=c/2$时）直接穿过块并加入下一个块。这可以看作是一种特性重用，类似于DenseNet和CondenseNet。

&emsp;在DenseNet中，为了分析特征重用模式，绘制层间权重的l1范数，如图4(a)所示。很明显，相邻层之间的连接比其他层之间的连接更强。这意味着所有层之间的紧密连接可能会引入冗余。最近的CondenseNet也支持这一观点。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-08-26-ShuffleNet V2--高效CNN架构设计的实用指南/figure4.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图4. DenseNet和ShuffleNet V2特征重用中的模式说明。（a）（由[DenseNet]提供）模型中卷积层的平均绝对滤波器权重。像素的颜色（s，l）编码了连接层s到l的权重的l1范数。（b）像素的颜色（s，l）表示ShuffleNet V2中直接连接块s到块l的通道数。所有像素值都归一化为[0，1]。</div>
</center>

&emsp;在ShuffleNet V2中，很容易证明第$i$个和第$i+j$个构建块之间的“直接连接”的通道的数量为$r^{j}c$，其中$r=(1-c')/c$。换句话说，特征重用量随着两个块之间的距离呈指数衰减。在距离较远的块之间，特征重用变得非常弱。图4(b)绘制了与图(a)类似的可视化，其中$r=0.5$。注意(b)中的模式类似于(a)。

&emsp;因此，ShuffleNet V2的结构通过设计实现了这种类型的特征重用模式。它具有与DenseNet中类似的特征重用的优点，但如前所述，它的效率要高得多。这在实验中得到验证，见表8。

## 4. 实验

&emsp;我们的消融实验是在ImageNet 2012分类数据集上进行的。按照通常的做法，所有网络都有四个级别的计算复杂度，即大约40、140、300和500+ MFLOPs。这种复杂性在移动场景中非常典型。其他超参数和协议与ShuffleNet V1完全相同。

&emsp;我们与以下网络架构进行比较：

- *ShuffleNet V1.* 在[ShuffleNet]中，比较了一系列组数g。结果表明，$g=3$在精度和速度之间具有较好的平衡关系。这也符合我们的观察。在这项工作中，我们主要使用$g=3$。

- *MobileNet V2.* 它比MobileNet V1好。为了进行全面的比较，我们在原始论文和我们的复现中都报告了准确性，因为在原论文中缺少一些结果。

- *Xception.* 原始的Xception模型非常大（FLOPs>2G），超出了我们的比较范围。最近的工作提出了一种改进的轻量级Xception结构，该结构在精度和效率之间有更好的权衡。我们和这个变体进行比较。

- *DenseNet.* 原始工作只报告大型模型的结果（FLOPs>2G）。为了进行直接比较，我们按照表5中的架构设置重新实现了它，其中阶段2-4中的构建块由DenseNet块组成。我们调整通道的数量，以满足不同的目标复杂性。

&emsp;表8总结了所有结果。我们从不同的角度分析这些结果。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-08-26-ShuffleNet V2--高效CNN架构设计的实用指南/table8.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表8. 在两个平台和四个计算复杂度级别上，比较几种网络架构对分类错误（验证集、单中心裁剪）和速度的影响。结果按复杂度级别分组，以便更好地进行比较。GPU的批大小为8，ARM的批大小为1。图像大小为224×224，除了：[*]160×160和[**]192×192。由于目前缺乏有效的实现，我们不提供CondenseNets的速度测量。</div>
</center>

*准确率 vs. FLOPs* 很明显，所提出的ShuffleNet V2模型比所有其他网络都要出色得多，尤其是在较小的计算预算下。此外，我们注意到MobileNet V2在40MFLOPs级别上（224×224图像大小）表现较差。这可能是由于通道太少造成的。相反，我们的模型没有这个缺点，因为我们的高效设计允许使用更多的通道。此外，虽然我们的模型和DenseNet都重用了特征，但是我们的模型要有效得多，如第3节所讨论的。

&emsp;表8还将我们的模型与其他最先进的网络进行了比较，包括CondenseNet、IGCV2和IGCV3。我们的模型在不同的复杂度级别上表现得更好。

*推理速度 vs. FLOPs/准确率.* 对于四种精度较好的架构ShuffleNet V2、MobileNet V2、ShuffleNet V1和Xception，我们将它们的实际速度与FLOPs进行比较，如图1(c)和(d)所示。关于不同分辨率的更多结果见附录表1。

&emsp;ShuffleNet V2显然比其他三个网络更快，尤其是在GPU上。例如，在500MFLOPs时，ShuffleNet V2比MobileNet V2快58%，比ShuffleNet V1快63%，比Xception快25%。在ARM上，ShuffleNet V1、Xception和ShuffleNet V2的速度相当；然而，MobileNet V2要慢得多，尤其是在较小的FLOPs下。我们认为这是因为MobileNet V2有更高的MAC（见第二节G1和G4），这在移动设备上很重要。

&emsp;与MobileNet V1、IGCV2和IGCV3相比，我们有两个观察结果。首先，虽然MobileNet V1的精度不高，但是它在GPU上的速度比所有的同类产品都要快，包括ShuffleNet V2。我们认为这是因为它的结构满足大多数建议的准则（例如，对于G3, MobileNet V1的碎片甚至比ShuffleNet V2更少）。其次，IGCV2和IGCV3速度较慢。这是由于使用了太多的卷积组。这两项观察都符合我们提出的准则。

&emsp;最近，自动模型搜索成为CNN架构设计的一个很有前途的趋势。表8的底部部分评估了一些自动生成的模型。我们发现它们的速度相对较慢。我们认为这主要是因为使用了太多的碎片（参见G3）。尽管如此，这一研究方向仍然很有前景。例如，如果模型搜索算法与我们提出的准则相结合，并在目标平台上评估直接度量（速度），可能会得到更好的模型。

&emsp;最后，图1(a)和(b)总结了精度与直接度量--速度的结果。我们得出结论，ShuffeNet V2在GPU和ARM上都是最好的。

*与其他方法的兼容性.* ShuffeNet V2可以与其他技术相结合，进一步提高性能。当配备挤压-激励（SE）模块时，ShuffleNet V2的分类精度提高了0.5%，但速度有一定的损失。模块结构如附录图2(b)所示。结果见表8(底部部分)。

*对大型模型的泛化.* 虽然我们的消融实验主要用于轻量级的情况下，但ShuffleNet V2可以用于大型模型(比如FLOPs≥2G)。表6比较了50层ShuffleNet V2（详见附录）与对应版本的ShuffleNet V1和ResNet-50。ShuffleNet V2仍然在2.3GFLOPs上优于ShuffleNet V1，并且以40%的更少的FLOPs超过ResNet-50。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-08-26-ShuffleNet V2--高效CNN架构设计的实用指南/table6.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表6. 大型模型的结果。</div>
</center>

&emsp;对于非常深的ShuffleNet V2（例如超过100层），为了使训练更快地收敛，我们通过添加一个残差路径（详见附录）来稍微修改基本的ShuffleNet V2单元。表6给出了一个包含164层的ShuffleNet V2模型，其中包含SE组件（详见附录）。与以前的最先进的模型相比，它在更少的FLOPs下获得了更高的精度。

*目标检测.* 为了评估泛化能力，我们还测试了COCO目标检测任务。我们使用最先进的轻型探测器-Light-Head RCNN作为我们的框架，并遵循相同的训练和测试协议。只有骨干网络被我们的取代。模型在ImageNet上进行预训练，然后对检测任务进行微调。除了minival集中的5000张图片外，我们使用COCO中的train+val集进行训练，并使用minival集进行测试。精度指标为COCO标准mmAP，即在box IoU阈值0.5到0.95之间的平均mAPs。

&emsp;将ShuffleNet V2与其他三个轻量级模型:Xception、ShuffleNet V1和MobileNet V2在四个复杂度级别上进行比较。表7中的结果显示ShuffleNet V2的性能最好。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-08-26-ShuffleNet V2--高效CNN架构设计的实用指南/table7.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表7. COCO目标检测性能。输入图像尺寸为800×1200。FLOPs行列出了224×224输入大小下的复杂度级别。对于GPU速度评估，批大小为4。我们不测试ARM，因为[Light-head R-CNN]中所需的PSROI池化操作当前在ARM上不可用。</div>
</center>

&emsp;将检测结果（表7）与分类结果（表8）进行比较，有趣的是，在分类时，准确率等级为ShuffleNet V2 $\ge$ MobileNet V2 > ShuffeNet V1 > Xception，而在检测时，该等级变为ShuffleNet V2 > Xception $\ge$ ShuffleNet V1 $\ge$ MobileNet V2。这说明，Xception在检测任务上表现良好。这可能是由于Xception构建块的感受野大于其他构建块（7 vs. 3）。受此启发，我们还通过在每个构建块的第一个逐点卷积之前引入额外的3×3深度卷积，来扩大ShuffleNet V2的感受野。这个变体表示为ShuffleNet V2*。只有少数额外的FLOPs，它进一步提高准确性。

&emsp;我们还在GPU上测试运行时间。为了公平比较，批处理大小被设置为4，以确保GPU的充分利用。由于数据复制（分辨率高达800×1200）及其他检测专用操作（如PSRoI池化）的开销，不同模型之间的速度差距小于分类。尽管如此，ShuffleNet V2的性能仍然优于其他模型，例如，它比ShuffleNet V1快40%左右，比MobileNet V2快16%。

&emsp;此外，变体ShuffleNet v2*具有最好的精度，仍然比其他方法更快。这引发了一个实际的问题：如何增加感受野的大小？这对于高分辨率图像中的目标检测至关重要。我们以后会研究这个话题。

## 5. 结论

&emsp;我们建议网络架构设计应考虑速度等直接指标，而不是FLOPs等间接指标。我们提出了实用的指导方针和一个新的架构，ShuffleNet V2。综合实验验证了该模型的有效性。我们希望这项工作能够启发未来的网络架构设计工作，使其更具有平台意识和实用性。

## 附录

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-08-26-ShuffleNet V2--高效CNN架构设计的实用指南/a-figure1.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">附录图1. 实验中准则3用到的构建块。(a) 1-fragment. (b) 2-fragment-series. (c) 4-fragment-series. (d) 2-fragment-parallel. (e) 4-ragment-parallel.</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-08-26-ShuffleNet V2--高效CNN架构设计的实用指南/a-figure2.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">附录图2. 含有SE/残差的ShuffleNet v2构建块。(a)带有残差的ShuffleNet V2。(b)带有SE的ShuffleNet V2。(c)含有SE和残差的ShuffleNet V2。</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-08-26-ShuffleNet V2--高效CNN架构设计的实用指南/a-table1.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">附录表1. 表(a)比较了每个网络(整个体系结构)的速度。表(b)比较了每个网络单元的速度，我们将每个网络的10个网络单元堆叠起来；c的值表示ShuffleNet V2的通道数，我们调整通道数以保持其他网络单元的FLOPs不变。详情请参阅第4节。[∗]对于输入大小为320×320的40M FLOPs模型，为了保证GPU的利用率，将batchsize设置为8，否则我们将batchsize设置为1。</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-08-26-ShuffleNet V2--高效CNN架构设计的实用指南/a-table2.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">附录表2. 大型模型的架构。构建基块的卷积核形状和堆叠块数显示在括号中。降采样由conv3_1、conv4_1和conv5_1执行，步长为2。对于ShuffleNet V1-50和ResNet-50，瓶颈比率设置为1:4。对于SE-ShuffleNet V2-164，我们在残差add-ReLUs之前添加SE模块（详情见附录图2）；我们将SE模块中的神经元数设置为相应构建块中通道数的1/2。详见第4节。</div>
</center>

---

## 个人看法

&emsp;这篇文章是ShuffleNet的升级版，最主要的贡献是提出了FLOPs其实不足以衡量网络的速度，包括MAC和并行度等因素也需要加以考虑，然后据此提出了构建轻量级网络的四个准则，并根据四个准则，将ShuffleNet中不符合的部分进行修正，从而得到了ShuffleNetV2。

&emsp;再回过头看四个准则，准则1告诉我们输入和输出通道数一致的时候MAC最小，准则2告诉我们减少分组卷积可以减小MAC，准则3告诉我们模型中的分支数越少则模型的并行度越高从而速度越快，准则4告诉我们减少element-wise操作可以加快速度。根据四个准则，再看图3所示的构建块。对比(a)和(c)，可以看到，(c)在一开始的地方有一个channel split的操作以将$c$个输入通道给拆分成$c-c'$和$c'$个通道分别进入两条支路，文章中采用$c'=c/2$，使得最后的输入通道数与输出通道数相等，符合了准则1。而且(c)中也将1×1卷积层中的group操作去掉了符合准则2，其实前面的channel split操作可以看作是一种group操作。(c)里channel shuffle的操作也移到了concat后面，因为前面已经将1×1卷积层中的group操作去掉了，所以也不用在其后添加channel shuffle操作，同时这么做也减少了支路长度，符合准则3。而将element-wise的add操作替换成concat，则符合准则4。最后因为在构建网络的时候会将(c)重复堆叠，所以channel split、concat和channel shuffle这几个操作是可以合并在一起的。
