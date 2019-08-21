---
layout:     post
title:      ShuffleNet--一个非常有效的移动卷积神经网络
subtitle:   ShuffleNet--An Extremely Efficient Convolutional Neural Network for Mobile
date:       2019-08-19
author:     CJR
header-img: img/2019-08-19-ShuffleNet--一个非常有效的移动卷积神经网络/post-bg.jpg
catalog: true
mathjax: true
tags:
    - Lightweight Network
    - ShuffleNet
    - CNN
---

# ShuffleNet

&emsp;ShuffleNet是Face++在2017年发布的一个极有效率且可以运行在手机等移动设备上的网络结构，文章也发表在了CVPR2018上，原文可见[ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)。

---

## 摘要

&emsp;我们介绍了一个计算效率极高的CNN架构ShuffleNet，它是专门为计算能力非常有限的移动设备（例如，10-150 MFLOPs）而设计的。该结构利用分组逐点卷积（pointwise group convolution）和通道重排（channel shuffle）两种新的运算方法，在保证计算精度的同时，大大降低了计算成本。ImageNet分类和MS COCO目标检测实验表明，在40 MFLOPs计算预算下，ShuffleNet的性能优于其他结构，例如，在ImageNet分类任务上，与最近的MobileNet相比，top-1错误率（绝对7.8%）更低。在基于arm的移动设备上，ShuffleNet比AlexNet实现了约13倍的实际加速，同时保持了相当的准确性。

## 1. 介绍

&emsp;构建更深、更大的卷积神经网络(CNNs)是解决主要视觉识别任务的主要趋势。最精确的CNNs通常有数百层和数千个通道，因此需要数十亿次的计算。本报告研究了另一个极端：在非常有限的计算预算中，以数十或数百MFLOPS来追求最佳的精度，重点关注诸如无人机、机器人和智能手机等常见移动平台。请注意，许多现有的工作侧重于修剪、压缩或low-bit表示“基本”网络架构。在这里，我们的目标是探索一个高效的基础架构，专门为我们所需的计算范围设计。

&emsp;我们注意到，最先进的基础架构，如Xception和ResNeXt，由于代价高昂的密集1×1卷积，在极小的网络中变得效率更低。我们建议使用分组逐点卷积来降低1×1卷积的计算复杂度。为了克服分组卷积带来的副作用，我们提出了一种新的通道重排操作来帮助信息在特征通道间流动。基于这两种技术，我们构建了一个名为ShuffleNet的高效架构。与其他流行的结构相比，对于给定的计算复杂度预算，我们的ShuffleNet允许更多的feature map通道，这有助于编码更多的信息，并且对非常小的网络的性能尤其重要。

&emsp;我们评估了具有挑战性的ImageNet分类和MS COCO目标检测任务上的模型。通过一系列的控制实验，验证了该方法的有效性，并取得了较好的性能。与最先进的架构MobileNet相比，ShuffleNet的性能有了显著的提高，例如，在40 MFLOPs级别上，ImageNet top-1错误率降低了7.8%。

&emsp;我们还研究了实际硬件的加速，即基于现成的ARM计算核心。ShuffleNet模型比AlexNet获得了约13倍的实际加速（理论加速为18倍），同时保持了相当的精度。

## 2. 相关工作
**高效模型设计**&emsp;近年来，深度神经网络在计算机视觉任务中取得了成功，其中模型设计起着重要作用。在嵌入式设备上运行高质量的深度神经网络的需求日益增长，促使了对高效模型设计的研究。例如，与简单地叠加卷积层相比，GoogLeNet以更低的复杂度增加了网络的深度。SqueezeNet在保持精度的同时，显著降低了参数和计算量。ResNet利用高效的瓶颈结构实现了令人印象深刻的性能。SENet引入了一种架构单元，它可以以很少的计算成本提高性能。与我们同时，最近的一项工作利用强化学习和模型搜索来探索有效的模型设计。提出的移动NASNet模型的性能与我们的同类ShuffleNet模型相当（由于ImageNet分类错误，分别为26.0%@564 MFLOPs和26.3%@524 MFLOPs）。但是NASNet报告极小模型的结果（例如复杂度小于150 MFLOPs），也不会评估移动设备上的实际推理时间。SqueezeNet广泛使用1x1卷积与挤压和扩展模块，主要集中于减少参数的数量。最近的工作将关注点从减少参数转移到减少操作的数量(MAdds)和实际测量的延迟。MobileNetV1采用深度可分离卷积，大大提高了计算效率。MobileNetV2在此基础上进行了扩展，引入了一个具有反向残差和线性瓶颈的资源高效块。ShuffleNet利用组卷积和通道洗牌操作进一步减少MAdds。CondenseNet在训练阶段学习组卷积，以保持层与层之间有用的紧密连接，以便特征重用。ShiftNet提出了与逐点卷积交织的移位操作，以取代昂贵的空间卷积。

**分组卷积**&emsp;分组卷积的概念最早出现在AlexNet中，用于将模型分布到两个GPU上，在ResNeXt和DeepRoots中得到了很好的证明。Xception中提出的深度可分离卷积概括了Inception系列中可分离卷积的思想。最近，MobileNet采用了深度可分离卷积，并在轻量级模型中获得了最先进的结果。我们的工作以一种新的形式推广了分组卷积和深度可分离卷积。

**通道重排操作**&emsp;据我们所知，在之前的高效模型设计工作中，虽然CNN库cuda-convnet支持“随机稀疏卷积”层，这相当于随机通道重排后接一分组卷积层，但通道重排操作的思想在之前的工作中很少被提及。这种“随机重排”操作有不同的目的，以后很少被利用。最近，另一个并行工作[Interleaved group convolutions for deep neural networks]也采用了这一思想的两阶段卷积。然而，其并没有专门研究通道重排本身的有效性及其在微小模型设计中的应用。

**模型加速**&emsp;这个方向的目的是在保持预训练模型精度的同时，加速推理。修剪网络连接或通道可以在保持性能的同时减少预训练模型中的冗余连接。文献中提出量化和因式分解来减少计算中的冗余，加快推理速度。在不修改参数的情况下，由FFT等方法实现的优化卷积算法在实际应用中降低了时间消耗。蒸馏将知识从大型模型转移到小型模型，这使得训练小型模型更加容易。

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

#### 6.1.2 测量设置

&emsp;为了测量延迟，我们使用标准的谷歌Pixel手机，并通过标准的TFLite基准测试工具运行所有网络。我们在所有测量中都使用单线程大内核。我们没有报告多核推理时间，因为我们发现这种设置对移动应用程序不太实用。

### 6.2 结果

&emsp;如下图所示，我们的模型优于目前的STOA，如MnasNet、ProxylessNas和MobileNetV2。我们在表3中报告了不同Pixel手机上的浮点性能。我们在表4中包括量化结果。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/MobileNetV3-small-large.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Pixel 1 延迟与ImageNet上top-1准确率之间的权衡。所有模型都使用输入分辨率224。V3 large和V3 small使用乘数0.75、1和1.25来显示最佳边界。使用TFLite在同一设备的单个大内核上测量所有延迟。MobileNetV3-Small和Large是我们提议的下一代移动模型</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/Accuracy-MADDS.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">MAdds和top-1准确率之间的权衡。这允许比较针对不同硬件或软件框架的模型。所有MobileNetV3用于输入分辨率224，并使用乘数0.35、0.5、0.75、1和1.25。其他分辨率见第6节</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/table3.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表3. Pixel系列手机的浮点性能（“P-n”表示Pixel-n手机）。所有延迟都以毫秒为单位。推理延迟是使用批量大小为1的单个大核心来测量的</div>
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

&emsp;在下图中，我们展示了MobileNetV3性能权衡作为乘数和分辨率的函数。请注意，MobileNetV3-Small的性能比MobileNetV3-Large的性能好得多，其乘法器缩放到与性能匹配的倍数接近3%。另一方面，分辨率提供了比乘数更好的权衡。但是，需要注意的是，分辨率通常是由问题决定的(例如分割和检测问题通常需要更高的分辨率)，因此不能总是用作可调参数。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/MobileNetV3-V2.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">作为不同乘数和分辨率函数的MobileNetV3性能。在我们的实验中，我们使用了固定分辨率为224同时乘数为0.35、0.5、0.75、1.0和1.25，以及固定深度乘数为1.0同时分辨率为96、128、160、192、224和256。彩色效果最佳</div>
</center>

#### 6.2.1 模型简化测试（Ablation study）

##### 非线性的影响

&emsp;在表5和下图中，我们展示了在何处插入h-swish的决定如何影响延迟。特别重要的是，我们注意到，在整个网络上使用h-swish会导致精度略有提高(0.2)，同时增加了近20%的延迟，并再次回到效率边界（efficient frontier）。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/table5.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表5. 非线性对MobileNetV3-Large的影响。在h-swish@N中，N表示启用h-swish的第一层的通道数</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/h-swish.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">h-swish vs swish vs ReLU对延迟的影响。曲线显示了使用深度乘数的边界。请注意，将h-swish放置在112个或更多通道的所有层，沿着最佳边界移动</div>
</center>

&emsp;另一方面，与ReLU相比，使用h-swish提高了效率边界，尽管仍然比ReLU计算代价高了12%左右。最后，我们注意到，当h-swish通过融合到卷积运算符中得到优化时，我们预计h-swish和ReLU之间的延迟差距即使没有消失，也会显著减小。然而，h-swish和swish之间不可能有这样的改进，因为计算sigmoid本来代价就比较高。

##### 其他组件的影响

&emsp;在下图中，我们展示了不同组件的引入是如何沿着延迟/准确率曲线移动的。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/progression.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">向网络体系结构中添加单个组件的影响</div>
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

&emsp;在本小节中，我们使用MobileNetV2和提出的MobileNetV3作为移动语义分割的网络骨架。此外，我们比较了两个分割头。第一个是在MobileNetV2中提出的R-ASPP。R-ASPP是一种无源空间金字塔池化模块的简化设计，它只采用由1×1卷积和全局平均池化操作组成的两个分支。在这项工作中，我们提出了另一种轻型分割头，称为Lite R-ASPP(或LR-ASPP)，如下图所示。Lite R-ASPP是对R-ASPP的改进，它部署全局平均池的方式类似于压缩-激励模块，其中我们使用了一个具有较大步长的较大池化核(以节省一些计算)，并且模块中只有一个1×1卷积。我们对MobileNetV3的最后一个块应用空洞卷积（atrous convolution）来提取更密集的特性，并进一步从底层特性添加一个跳跃（skip）连接来捕获更详细的信息。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-05-22-MobileNetV3/MobileNetV3-Segmentation.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">在MobileNetV3的基础上，提出的分割头Lite R-ASPP提供了快速的语义分割结果</div>
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

&emsp;表8显示了我们的Cityscapes测试集结果。我们的以MobileNetV3为网络骨干的分割模型，表现分别比ESPNetv2、CCC2和ESPNetv1超出10.5%、10.6%、12.3%，同时在MAdds方面更快。在MobileNetV3的最后一个块中，当不使用空洞卷积提取密集特征图时，性能略有下降0.6%，但速度提高到1.98B(对于半分辨率输入)，分别是ESPNetv2、CCC2和ESPNetv1的1.7倍、1.59倍和2.24倍。此外，我们的以MobileNetV3-Small作为骨干网络的模型仍然优于它们至少一个6.2%的健壮差额（healthy margin）。我们最快的模型版本比ESPNetv2-small好13.6%，推理速度稍微快一些。

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

&emsp;本文介绍了MobileNetV3的Large和Small模型，展示了移动分类、检测和分割的STOA。我们已经描述了我们利用多种类型的网络架构搜索以及网络设计的进步来交付下一代移动模型的努力。我们还展示了如何采用像swish这样的非线性，并以量化友好和有效的方式应用压缩和激励，将它们作为有效的工具引入到移动模型领域。我们还介绍了一种称为LR-ASSP的轻量化分割译码器。尽管如何最好地将自动搜索技术与人类的直觉结合起来仍然是一个悬而未决的问题，但我们很高兴地提出了这些第一个积极的结果，并将继续在未来的工作中改进方法。

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
