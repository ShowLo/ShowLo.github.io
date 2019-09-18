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

&emsp;构建更深、更大的卷积神经网络(CNNs)是解决主要视觉识别任务的主要趋势。最精确的CNNs通常有数百层和数千个通道，因此需要数十亿次的计算。本报告研究了另一个极端：在非常有限的计算预算中，以数十或数百MFLOPs来追求最佳的精度，重点关注诸如无人机、机器人和智能手机等常见移动平台。请注意，许多现有的工作侧重于修剪、压缩或low-bit表示“基本”网络架构。在这里，我们的目标是探索一个高效的基础架构，专门为我们所需的计算范围设计。

&emsp;我们注意到，最先进的基础架构，如Xception和ResNeXt，由于代价高昂的密集1×1卷积，在极小的网络中变得效率更低。我们建议使用分组逐点卷积来降低1×1卷积的计算复杂度。为了克服分组卷积带来的副作用，我们提出了一种新的通道重排操作来帮助信息在特征通道间流动。基于这两种技术，我们构建了一个名为ShuffleNet的高效架构。与其他流行的结构相比，对于给定的计算复杂度预算，我们的ShuffleNet允许更多的feature map通道，这有助于编码更多的信息，并且对非常小的网络的性能尤其重要。

&emsp;我们评估了具有挑战性的ImageNet分类和MS COCO目标检测任务上的模型。通过一系列的控制实验，验证了该方法的有效性，并取得了较好的性能。与最先进的架构MobileNet相比，ShuffleNet的性能有了显著的提高，例如，在40 MFLOPs级别上，ImageNet top-1错误率降低了7.8%。

&emsp;我们还研究了实际硬件的加速，即基于现成的ARM计算核心。ShuffleNet模型比AlexNet获得了约13倍的实际加速（理论加速为18倍），同时保持了相当的精度。

## 2. 相关工作
**高效模型设计**&emsp;近年来，深度神经网络在计算机视觉任务中取得了成功，其中模型设计起着重要作用。在嵌入式设备上运行高质量的深度神经网络的需求日益增长，促使了对高效模型设计的研究。例如，与简单地叠加卷积层相比，GoogLeNet以更低的复杂度增加了网络的深度。SqueezeNet在保持精度的同时，显著降低了参数和计算量。ResNet利用高效的瓶颈结构实现了令人印象深刻的性能。SENet引入了一种架构单元，它可以以很少的计算成本提高性能。与我们同时，最近的一项工作利用强化学习和模型搜索来探索有效的模型设计。提出的移动NASNet模型的性能与我们的同类ShuffleNet模型相当（ImageNet分类错误率分别为26.0%@564 MFLOPs和26.3%@524 MFLOPs）。但是NASNet报告极小模型的结果（例如复杂度小于150 MFLOPs），也不会评估移动设备上的实际推理时间。SqueezeNet广泛使用1x1卷积与挤压和扩展模块，主要集中于减少参数的数量。最近的工作将关注点从减少参数转移到减少操作的数量(MAdds)和实际测量的延迟。MobileNetV1采用深度可分离卷积，大大提高了计算效率。MobileNetV2在此基础上进行了扩展，引入了一个具有反向残差和线性瓶颈的资源高效块。ShuffleNet利用组卷积和通道重排操作进一步减少MAdds。CondenseNet在训练阶段学习组卷积，以保持层与层之间有用的紧密连接，以便特征重用。ShiftNet提出了与逐点卷积交织的移位操作，以取代昂贵的空间卷积。

**分组卷积**&emsp;分组卷积的概念最早出现在AlexNet中，用于将模型分布到两个GPU上，在ResNeXt和DeepRoots中得到了很好的证明。Xception中提出的深度可分离卷积概括了Inception系列中可分离卷积的思想。最近，MobileNet采用了深度可分离卷积，并在轻量级模型中获得了最先进的结果。我们的工作以一种新的形式推广了分组卷积和深度可分离卷积。

**通道重排操作**&emsp;据我们所知，在之前的高效模型设计工作中，虽然CNN库cuda-convnet支持“随机稀疏卷积”层，这相当于随机通道重排后接一分组卷积层，但通道重排操作的思想在之前的工作中很少被提及。这种“随机重排”操作有不同的目的，之后很少被利用。最近，另一个并行工作[Interleaved group convolutions for deep neural networks]也采用了这一思想的两阶段卷积。然而，其并没有专门研究通道重排本身的有效性及其在微小模型设计中的应用。

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

&emsp;如果我们允许组卷积得到来自不同组的输入数据（如图1(b)所示），那么输入和输出通道将是完全相关的。具体来说，对于前一组层生成的feature map，我们可以先将每组中的通道划分为几个子组，然后在下一层中为每一组提供不同的子组。这可以通过通道重排操作高效优雅地实现（图1(c)）：假设卷积层有g组，输出有g×n个通道；我们首先将输出通道维数reshape为（g，n），转置，然后将其展平，作为下一层的输入。注意，即使两个卷积的组数不同，操作仍然有效。此外，通道重排也是可微的，这意味着它可以嵌入到网络结构中进行端到端训练。此外，通道重排也是可微的，这意味着它可以嵌入到网络结构中进行端到端训练。

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

&emsp;我们主要是在ImageNet 2012分类数据集上评估我们的模型。我们遵循[Aggregated residual transformations for deep neural networks]中使用的大多数训练设置和超参数，但有两个例外:(i)我们将权重衰减设置为4e-5而不是1e-5，采用线性衰减学习率策略(由0.5降至0)；(ii)我们使用稍微不那么激进的规模扩大（scale augmentation）来进行数据预处理。在[MobileNets]中也引用了类似的修改，因为这样的小网络通常存在拟合不足而不是过拟合。在4个gpu上训练一个$3\times 10^{5}$迭代的模型需要1 - 2天，批处理大小设置为1024。为了进行基准测试，我们比较了在ImageNet验证集上的single-crop top-1性能，即从256×输入图像裁剪224×224中心视图，并评估了分类精度。我们对所有模型使用完全相同的设置，以确保公平的比较。

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

&emsp;VGG、ResNet、GoogleNet、ResNeXt、Xception等最新领先的卷积单元，在大模型(如$\ge$ 1 GFLOPs)上追求最先进的结果，但没有充分探索低复杂度条件。在本节中，我们考察了各种构建块，并在相同的复杂度约束下与ShuffleNet进行了比较。

&emsp;为了进行公平的比较，我们使用了如表1所示的整体网络架构。我们将阶段2-4中的ShuffleNet单元替换为其他结构，然后调整通道的数量，以确保复杂性保持不变。我们研究的结构包括:

* *VGG-like.*&emsp;根据VGG net的设计原则，我们使用了一个两层的3×3卷积作为基本构建块。有所不同的是，我们在每次卷积之后都添加了一个批归一化层，使端到端的训练更加容易。

* *ResNet.*&emsp;实验中采用了“瓶颈”设计，在[Deep residual learning for image recognition]中得到了较好的验证。与之相同，瓶颈比率也是1:4。

* *Xception-like.*&emsp;在[Deep learning with depthwise separable convolutions]中提出的原始结构涉及不同阶段的花哨设计或超参数，我们发现很难在小模型上进行公平的比较。我们将分组逐点卷积和通道重排操作从ShuffleNet中去掉（也相当于ShuffleNet中g=1），得到的结构与其中“深度可分离卷积”的思想相同，在这里称为*Xception-like*结构。

* *ResNeXt.*&emsp;我们使用cardinality=16和瓶颈比=1:2的设置，如[Aggregated residual transformations for deep neural networks]中建议的那样。我们还研究了其他设置，例如瓶颈比=1:4，并得到了类似的结果。

&emsp;我们使用完全相同的设置来训练这些模型。结果如表4所示。在不同的复杂性下，我们的ShuffleNet模型比大多数其他模型有显著的优势。有趣的是，我们发现了特征图通道与分类精度之间的经验关系。例如，在38 MFLOPs复杂度下，VGG-like、ResNet、ResNeXt、Xceplike、ShuffleNet模型第4阶段的输出通道（见表1）分别为50、192、192、288、576，这与精度的提高是一致的。由于ShuffleNet的高效设计，我们可以在给定的计算预算下使用更多的通道，从而通常可以获得更好的性能。

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
    padding: 2px;">表7. MS COCO上的目标检测结果（数字越大表示性能越好）。对于MobileNets，我们比较了两个结果：1）由[MobileNets]报告的COCO检测分数；2）从我们重新实现的MobileNets进行微调，其训练和微调设置与ShuffleNets完全相同</div>
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

&emsp;其实这篇文章中最主要的创新点还是通道重排（channel shuffle）以及对应的网络构建基本块（图2）。~~有趣的一点是我发现图2中的(a)所示的构建块其实就是MobileNetV2的基本构建块，而MobileNetV2的发表却是在这篇论文之后，也就是说MobileNetV2用这个基本构建块又得到了更好的结果，也不知道这是否有所借鉴还是纯属巧合。。。不过MobileNetV2里面对这个基本构建块的阐述也的确要更加的透彻，而且用构建块堆叠起来的整体网络架构也是有所不同。~~ 2333，今天发现原来图2（a）的结构最早是ResNet中提出来的，其实MobileNetV2是在其基础上作了一点小修改。
