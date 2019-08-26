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

## 3. 方法

### 3.1 用于分组卷积的通道重排

&emsp;现代卷积神经网络通常由相同结构的重复构建块组成。其中，最先进的网络，如Xception和ResNeXt，在构建块中引入了高效的深度可分卷积或组卷积，从而在表示能力和计算成本之间取得了很好的权衡。然而，我们注意到这两种设计都没有完全考虑到1×1卷积（也称为逐点卷积），这需要相当大的复杂性。例如，在ResNeXt中，只有3×3层具有分组卷积。 因此，对于ResNeXt中的每个残差单元，逐点卷积占用93.4%的乘法-加法。在微型网络中，昂贵的逐点卷积会导致有限的通道来满足复杂度约束，这可能会严重损害精度。

&emsp;为了解决这个问题，一个简单的解决方案是在1×1层上应用通道稀疏连接，例如分组卷积。通过确保每个卷积只对对应的输入信道组进行运算，分组卷积显著降低了计算成本。然而，如果多个分组卷积叠加在一起，就会产生一个副作用：来自某个通道的输出只来自一小部分输入通道。图1(a)为两个叠加分组卷积层的情况。很明显，某个组的输出只与组内的输入相关。此属性会阻塞通道组之间的信息流并削弱表征能力。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-08-19-ShuffleNet--一个非常有效的移动卷积神经网络/figure1.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图1. 用两个叠加的分组卷积进行通道重排。GConv代表分组卷积。a）两个具有相同组数的叠加卷积层。每个输出通道仅与组内的输入通道相关。当GConv2在GConv1之后从不同的组中获取数据时，输入和输出通道完全相关；c）使用通道重排的与b）等效的实现。</div>
</center>

&emsp;如果我们允许组卷积得到来自不同组的输入数据（如图1(b)所示），那么输入和输出通道将是完全相关的。具体来说，对于前一组层生成的feature map，我们可以先将每组中的通道划分为几个子组，然后在下一层中为每一组提供不同的子组。这可以通过通道重排操作高效优雅地实现（图1(c)）：假设卷积层有g组，输出有g×n个通道；我们首先将输出通道维数reshape为（g，n），转置，然后将其展平，作为下一层的输入。注意，即使两个卷积的组数不同，操作仍然有效。此外，信道洗牌也是可微的，这意味着它可以嵌入到网络结构中进行端到端训练。此外，通道重排也是可微的，这意味着它可以嵌入到网络结构中进行端到端训练。

&emsp;通道重排操作使得构建具有多组卷积层的更强大的结构成为可能。在下一小节中，我们将介绍一种具有通道重排和分组卷积的高效网络单元。

### 3.2 ShuffleNet单元

&emsp;利用通道重排的优点，提出了一种专为小型网络设计的新型ShuffleNet单元。我们从图2(a)中瓶颈单元的设计原理出发，它是一个残差块。在其残差分支中，对于3×3层，我们在瓶颈特征图上应用了计算量较少的3×3深度卷积。然后，我们将第一个1×1层替换为分组逐点卷积，然后进行通道重排操作，形成一个ShuffleNet单元，如图2(b)所示。第二个分组逐点卷积的目的是恢复通道维数以匹配shortcut路径。为了简单起见，我们不在第二个逐点卷积层之后应用额外的通道重排操作，因为它已产生可比较的成绩。批归一化（BN）和非线性的使用与[Deep residual learning for image recognition, Aggregated residual transformations for deep neural networks]相似，只是我们没有按照[Xception: Deep learning with depthwise separable convolutions]建议的在深度卷积后使用ReLU。对于将stride应用于ShuffleNet的情况，我们只做了两个修改（见图2(c)）：(i)在shortcut路径上添加一个3×3的平均池化；(ii)用通道级联替换元素相加，这样可以很容易地扩大通道尺寸，而几乎无需额外的计算成本。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-08-19-ShuffleNet--一个非常有效的移动卷积神经网络/figure2.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图2. ShuffleNet单位。a）具有深度卷积（DWConv）的瓶颈单元；b）具有分组逐点卷积（GConv）和通道重排的ShuffleNet单元；c）具有stride=2的ShuffleNet单元。</div>
</center>

&emsp;由于分组逐点卷积与通道重排的结合，使得shuffle单元中的所有分量都能得到有效的计算。与ResNet(瓶颈设计)和ResNeXt相比，我们的结构在相同的设置下复杂度更低。例如，给定输入大小$c\times h\times w$，瓶颈通道数$m$, ResNet单元需要$hw(2cm+9m^{2})$FLOPs，ResNeXt需要$hw(2cm+9m^{2}/g)$FLOPs，而我们的ShuffleNet单元只需要$hw(2cm/g + 9m)$FLOPs。其中g表示卷积的组数。换句话说，给定计算预算，ShuffleNet可以使用更广的特征图。我们发现这对于小型网络非常重要，因为小型网络通常没有足够的通道来处理信息。

&emsp;此外，在ShuffleNet中，深度卷积只在瓶颈特征图上执行。虽然深度卷积通常具有非常低的理论复杂度，但是我们发现在低功耗的移动设备上很难有效地实现，这可能是因为与其他密集操作相比，它的计算/内存访问率更低。[Xception]中也提到了这种缺点，它有一个基于TensorFlow的运行时库。在ShuffleNet单元中，我们故意只在瓶颈上使用深度卷积，以尽可能地避免开销。

### 3.3 网络体系结构

&emsp;构建在ShuffleNet单元上，我们在表1中展示了整个ShuffleNet架构。该网络主要由一组分为三个阶段的ShuffleNet单元组成。每个阶段的第一个构建块使用stride=2。一个阶段中的其他超参数保持不变，下一个阶段的输出通道加倍。我们为每个ShuffleNet单元将瓶颈通道的数量设置为输出通道的1/4。我们的目的是提供一个尽可能简单的参考设计，尽管我们发现进一步的超参数调优可能会产生更好的结果。

&emsp;在ShuffleNet单元中，组数g控制逐点卷积的连接稀疏性。表1探索了不同的组数，我们调整了输出通道，以确保总体计算成本大致不变(∼140 MFLOPs)。显然，对于给定的复杂度约束，较大的组数会产生更多的输出通道(因此会产生更多的卷积滤波器)，这有助于编码更多的信息，但由于相应的输入通道有限，这也可能导致单个卷积滤波器的性能下降。在第4.1.1节中，我们将研究这个数字受不同计算约束的影响。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-08-19-ShuffleNet--一个非常有效的移动卷积神经网络/table1.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表1. ShuffleNet架构。复杂度用FLOPs来计算，即浮点乘法加法的数目。注意，对于第2阶段，我们不在第一个逐点卷积层上应用分组卷积，因为输入通道数相对较小。</div>
</center>

&emsp;要将网络自定义为所需的复杂性，我们可以简单地在通道的数量上应用一个比例因子s。例如，我们将表1中的网络表示为“ShuffleNet 1×”，那么“ShuffleNet s×”表示将ShuffleNet 1×中的滤波器数量乘以s倍，因此总体复杂度大约为ShuffleNet 1×的$s^{2}$倍。

## 4. 实验

&emsp;我们主要是在ImageNet 2012分类数据集上评估我们的模型。我们遵循[Aggregated residual transformations for deep neural networks]中使用的大多数训练设置和超参数，但有两个例外:(i)我们将权重衰减设置为4e-5而不是1e-5，采用线性衰减学习率策略(由0.5降至0)；(ii)我们使用稍微不那么激进的规模扩大（scale augmentation）来进行数据预处理。在[MobileNets]中也引用了类似的修改，因为这样的小网络通常存在拟合不足而不是过拟合。在4个gpu上训练一个3×105迭代的模型需要1 - 2天，批处理大小设置为1024。为了进行基准测试，我们比较了在ImageNet验证集上的single-crop top-1性能，即从256×输入图像裁剪224×224中心视图，并评估了分类精度。我们对所有模型使用完全相同的设置，以确保公平的比较。

### 4.1 消融实验

&emsp;ShuffleNet的核心思想在于分组逐点卷积和通道重排操作。在本小节中，我们分别对它们评估。

#### 4.1.1 分组逐点卷积

&emsp;为了评估分组逐点卷积的重要性，我们比较了具有相同复杂度的ShuffleNet模型，其组数从1到8不等。如果组数等于1，则不涉及分组逐点卷积，则ShuffleNet单元成为一个“Xception-like”结构。为了更好地理解，我们还将网络的宽度扩展到3种不同的复杂性，并分别比较它们的分类性能。结果如表2所示。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-08-19-ShuffleNet--一个非常有效的移动卷积神经网络/table2.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表2. 分类误差VS组数g（较小的数字代表更好的性能）</div>
</center>

&emsp;从结果中我们可以看出，有分组卷积（g>1）的模型始终比没有分组逐点卷积（g=1）的模型表现得更好，较小的模型往往从分组中获益更多。例如，对于ShuffleNet 1×最佳条目（g=8），它比对应条目好1.2%，而对于ShuffleNet 0.5×和0.25×，两者之间的差距分别为3.5%和4.4%。注意，对于给定的复杂性约束，分组卷积允许更多的feature map通道，因此我们假设性能的提高来自于更广泛的feature map，它有助于编码更多的信息。此外，更小的网络包含更薄的特征图，这意味着它从放大的特征图中获益更多。

&emsp;表2还显示，对于某些模型(如ShuffleNet 0.5×)，当组数变得相对较大时（例如g=8），分类分数饱和甚至下降。随着组数的增加（因此特征图的范围更广），每个卷积滤波器的输入通道变得更少，这可能会损害表示能力。有趣的是，我们也注意到，对于如ShuffleNet 0.25×这样较小的模型，较大的组数往往会得到更好的一致性结果，这表明更宽的特征图为较小的模型带来了更多的好处。

#### 4.1.2 通道重排 vs 不重排

&emsp;shuffle操作的目的是为多个分组卷积层实现跨组信息流。表3比较了具有/不具有通道重排的ShuffleNet结构的性能（组数设置为3或8）。评估是在三种不同的复杂程度下进行的。很明显，在不同的设置下，通道重排可以不断地提高分类分数。特别是当组数较大时（如g=8），通道重排模型的性能明显优于同类模型，说明了跨组信息交换的重要性。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-08-19-ShuffleNet--一个非常有效的移动卷积神经网络/table3.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表3. 具有/不具有通道重排的ShuffleNet（数字越小表示性能越好）</div>
</center>

### 4.2 与其他结构单元比较

&emsp;VGG、ResNet、GoogleNet、ResNeXt、Xception等最新领先的卷积单元，在大模型(如$\ge$1 GFLOPs)上追求最先进的结果，但没有充分探索低复杂度条件。在本节中，我们考察了各种构建块，并在相同的复杂度约束下与ShuffleNet进行了比较。

&emsp;为了进行公平的比较，我们使用了如表1所示的整体网络架构。我们将阶段2-4中的ShuffleNet单元替换为其他结构，然后调整通道的数量，以确保复杂性保持不变。我们研究的结构包括:

* *VGG-like.*&emsp;根据VGG net的设计原则，我们使用了一个两层的3×3卷积作为基本构建块。有所不同的是，我们在每次卷积之后都添加了一个批归一化层，使端到端的训练更加容易。

* *ResNet.*&emsp;实验中采用了“瓶颈”设计，在[Deep residual learning for image recognition]中得到了较好的验证。与之相同，瓶颈比率也是1:4。

* *Xception-like.*&emsp;在[Deep learning with depthwise separable convolutions]中提出的原始结构涉及不同阶段的花哨设计或超参数，我们发现很难在小模型上进行公平的比较。我们将分组逐点卷积和通道重排操作从ShuffleNet中去掉（也相当于ShuffleNet中g=1），得到的结构与其中“深度可分离卷积”的思想相同，在这里称为*Xception-like*结构。

* *ResNeXt.*&emsp;我们使用cardinality=16和瓶颈比=1:2的设置，如[Aggregated residual transformations for deep neural networks]中建议的那样。我们还研究了其他设置，例如瓶颈比=1:4，并得到了类似的结果。

&emsp;我们使用完全相同的设置来训练这些模型。结果如表4所示。在不同的复杂性下，我们的ShuffleNet模型比大多数其他模型有显著的优势。有趣的是，我们发现了特征图通道与分类精度之间的经验关系。有趣的是，我们发现了特征图通道与分类精度之间的经验关系。例如，在38 MFLOPs复杂度下，VGG-like、ResNet、ResNeXt、Xceplike、ShuffleNet模型第4阶段的输出通道（见表1）分别为50、192、192、288、576，这与精度的提高是一致的。由于ShuffleNet的高效设计，我们可以在给定的计算预算下使用更多的通道，从而通常可以获得更好的性能。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-08-19-ShuffleNet--一个非常有效的移动卷积神经网络/table4.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表4. 分类误差vs各种结构（%，数字越小表示性能越好）。我们不会在较小的网络上报告VGG-like结构，因为精度明显较差。</div>
</center>

&emsp;请注意，上述比较不包括GoogleNet或Inception系列。我们发现在小型网络中生成这样的Inception结构并非易事，因为Inception模块的设计涉及太多的超参数。作为参考，第一个GoogleNet版本有31.3%的top-1错误率，代价是1.5 GFLOPs(见表6)。更复杂的Inception版本更准确，但是涉及显著增加的复杂性。最近，Kim等人提出了一种轻量级网络结构PVANET，采用Inception单元。我们重新实现的PVANET（224×224输入大小）分类错误率为29.7%，计算复杂度为557 MFLOPs，而我们的ShuffleNet 2x模型（g=3）分类错误率为26.3%，计算复杂度为524 MFLOPs（见表6）。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-08-19-ShuffleNet--一个非常有效的移动卷积神经网络/table6.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表6. 复杂度比较。*由BVLC实现（https://github.com/bvlc/caffe/tree/master/models/bvlc_googlenet）</div>
</center>

### 4.3 与MobileNets和其他框架进行比较

&emsp;最近Howard等人提出了MobileNets，是主要针对移动设备的高效网络架构。MobileNet采用了[Xception]的深度可分离卷积的思想，并在小模型上取得了最先进的结果。

&emsp;表5比较了不同复杂度级别下的分类得分。很明显，我们的ShuffleNet模型在所有的复杂性上都优于MobileNet。虽然我们的ShuffleNet网络是专门为小模型（<150 MFLOPs）设计的，但我们发现对于较高的计算成本它仍然比MobileNet好，例如在500 MFLOPs的情况下，比MobileNet 1×更精确3.1%。对于较小的网络（∼40MFLOPs），ShuffleNet比MobileNet高出7.8%。注意，我们的ShuffleNet架构包含50层，而MobileNet只有28层。为了更好地理解，我们还尝试通过移除阶段2-4中的一半块（见表5中的“shufflenet 0.5×shallow（g=3）”）构造26层的架构。结果表明，浅模型仍然是明显好于相应的MobileNet，这意味着ShuffleNet的有效性主要是高效结构的结果，而不是深度。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-08-19-ShuffleNet--一个非常有效的移动卷积神经网络/table5.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表5. ShuffleNet vs. MobileNet （在ImageNet分类任务上）</div>
</center>

&emsp;表6将我们的ShuffleNet与一些流行的模型进行了比较。结果表明，在精度相近的情况下，ShuffleNet比其他模型更有效。例如，ShuffleNet 0.5×理论上比AlexNet快18倍，分类得分相当。我们将在第4.5节中评估实际运行时间。同样值得注意的是，简单的体系结构设计使ShuffleNet能够轻松地配备最新进展，如[Squeeze-and-excitation networks，Swish: a self-gated activation function]。例如，在[SENet]中，作者提出了挤压和激发（SE）块，在大型ImageNet模型上实现了最先进的结果。我们发现SE模块与ShuffleNet骨干相结合也起到了作用，例如，将ShuffleNet 2×的top-1错误率提升到24.7%（如表5所示）。有趣的是，虽然理论复杂度的增加可以忽略不计，但我们发现，在移动设备上使用SE模块的ShuffleNet通常比“原始”ShuffleNet慢25～40%，这意味着实际的加速评估对于低成本架构设计至关重要。在第4.5节中，我们将做进一步的讨论。

### 4.4 泛化能力

&emsp;为了评估迁移学习的泛化能力，我们以MS COCO目标检测为任务，对我们的ShuffleNet模型进行了测试。我们采用Faster-RCNN作为检测框架，使用公开发布的Caffe代码进行默认设置的训练。与[MobileNets]类似，模型是在COCO train+val数据集上训练的，剔除5000张minival图像，我们在minival集上进行测试。表7显示了在两种输入分辨率下训练和评估结果的比较。将ShuffleNet 2×与复杂度相当（524与569 MFLOPs）的MobileNet进行比较，我们的ShuffleNet 2×在两个分辨率上都大大超过MobileNet；我们的ShuffleNet 1×在600×分辨率上也达到了与MobileNet相当的结果，但是降低了约4倍复杂度。我们推测，这一显著的收益部分是由于Shuffleet简单的架构设计，没有其他花里胡哨的东西。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-08-19-ShuffleNet--一个非常有效的移动卷积神经网络/table7.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表7. ShuffleNet vs. MobileNet （在ImageNet分类任务上）</div>
</center>

### 4.5 实际加速评估

&emsp;最后，我们评估了基于ARM平台的移动设备上的ShuffleNet模型的实际推理速度。虽然具有较大组数(例如g=4或g=8)的ShuffleNet通常具有更好的性能，但是我们发现在当前实现中它的效率较低。根据经验，g=3通常在准确性和实际推理时间之间有一个适当的平衡。如表8所示，测试使用了三个输入分辨率。由于内存访问和其他开销，我们发现在实现中，每减少4倍理论上的复杂性，就会导致约2.6倍实际加速。然而，与AlexNet相比，我们的ShuffleNet 0.5×模型仍然在可比较的分类精度下达到约13倍实际加速（理论加速为18倍），这比以前的AlexNet级模型或加速方法快得多。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-08-19-ShuffleNet--一个非常有效的移动卷积神经网络/table8.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表8. 移动设备上的实际推理时间（数字越小表示性能越好）。该平台基于单个高通公司Snapdragon 820处理器。所有结果都用单线程进行评估。</div>
</center>

---

## 个人看法

&emsp;其实这篇文章中最主要的创新点还是通道重排（channel shuffle）以及对应的网络构建基本块（图2）。有趣的一点是我发现图2中的(a)所示的构建块其实就是MobileNetV2的基本构建块，而MobileNetV2的发表却是在这篇论文之后，也就是说MobileNetV2用这个基本构建块又得到了更好的结果，也不知道这是否有所借鉴还是纯属巧合。。。不过MobileNetV2里面对这个基本构建块的阐述也的确要更加的透彻，而且用构建块堆叠起来的整体网络架构也是有所不同。
