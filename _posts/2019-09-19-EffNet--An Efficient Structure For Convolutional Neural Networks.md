---
layout:     post
title:      EffNet--卷积神经网络的一种高效结构
subtitle:   EffNet--An Efficient Structure For Convolutional Neural Networks
date:       2019-09-19
author:     CJR
header-img: img/2019-09-19-EffNet/post-bg.jpg
catalog: true
mathjax: true
tags:
    - Lightweight Network
    - EffNet
    - CNN
---

## MixNet

&emsp;这篇文章主要是在MobileNet之上做修改，增加了一个空间可分离卷积，发表在2018 25th IEEE International Conference on Image Processing (ICIP)上。原文可见[EffNet: An Efficient Structure for Convolutional Neural Networks](https://arxiv.org/abs/1801.06434)。

---

## 摘要

&emsp;随着卷积神经网络在客户产品中的应用日益广泛，对模型在嵌入式、移动硬件上高效运行的需求也日益显现。因此，从二值网络到修正卷积层，用各种各样的方法使得模型变得更加精细，成为研究的热点。我们对后者做出了贡献，并提出了一种新的卷积块，它在大大降低计算负担的同时，超过了目前的SOTA。我们的模型，被称为EffNet，是针对一开始就比较瘦的模型进行优化的，旨在解决现有模型（如MobileNet和ShuffleNet）中的问题。

## 1. 引言

&emsp;随着最近工业上对人工神经网络对产品性能的好处的认识，对高效算法的需求出现了，这种算法需要在成本低廉的硬件上实时运行。这在某种程度上与几乎平行的大学研究相矛盾。后者在执行周期和硬件方面享有相对的自由，而前者受市场压力和产品需求的影响。

&emsp;多年来，多篇论文提出了在小型硬件上进行实时推理的不同方法。一个例子是对训练过的网络进行剪枝。另一个是32位网络到二进制模型的定点转换。最近的研究集中在神经元的相互连接和普通卷积层的本质。普通卷积层的核心是一个四维张量，它以如下格式\[行、列、输入通道、输出通道\]扫过输入信号，产生四倍分量乘法，从而将计算成本放大四倍。

&emsp;由于$3\times 3$卷积现在是一种标准，它们自然成为优化的候选项。论文如\[MobileNet\]和\[ShuffleNet\]通过将计算按不同的维度分开来解决这个问题。然而，在他们的方法有两个问题没有解决。首先，两篇论文都报道了将大型网络变得更小、更高效。当将他们的模型应用于更小的网络时，结果会有所不同。其次，两种模型都为网络中的数据流造成了一个严重的瓶颈。这种瓶颈在高冗余模型中可能被证明是无关紧要的，但正如我们的实验所示，它对较小的模型具有破坏性的影响。

&emsp;因此，我们提出了一个替代系列，它保留了计算量的大部分比例的下降，而对精度几乎没有影响。我们通过优化数据流并忽略在这个独特领域中被证明有害的实践来实现这一改进。我们的新型卷积模块允许我们将更大的网络部署到低容量硬件上，或者提高现有模型的效率。

## 2. 相关工作

&emsp;该领域的许多工作都集中在超参数优化上。这类算法在目标算法和优化目标方面都比较通用。\[Practical bayesian optimization of machine learning algorithms\]通过最大化提高模型精度的概率，提出了像CNNs和和SVMs这样的黑箱算法的贝叶斯优化框架。这可以与\[Multi-objective parameter configuration of machine learning algorithms using model-based optimization\]中的多目标优化相结合，从而优化计算复杂度。当初始化正确时，这些方法大都能很好地工作，而且许多方法在受制于搜索空间。利用强化学习，\[Neural architecture search with reinforcement learning\]训练了LSTM来优化超参数，以提高精度和速度。这与最近的进化方法一起对搜索空间的限制更小，但是由于需要额外的步骤，使得开发变得更加复杂。

&emsp;另一种方法包括通过后处理的方式减少大型模型的大小。\[Noiseout: A simple way to prune neural networks\]、\[Learning to prune deep neural networks via layer-wise optimal brain surgeon\]、\[Pruning convolutional neural networks for resource efficient transfer learning\]等文献提出了精度代价最小的剪枝算法。然而，修剪会导致几个问题。开发过程需要一个额外的阶段，具有需要优化的专用超参数。此外，随着网络架构的改变，模型需要额外的微调。

&emsp;后处理压缩的另一种方法是将模型定点量化为小于常用32位浮点和二进制网络的基元。量化模型虽然要快得多，但与基线相比，它们的精度始终在下降，因此吸引力更小。

&emsp;最后也是最类似于这项工作的，论文如\[Xception\]，\[Mobilenets\]和\[ShuffleNet\]重新讨论了普通卷积算子的本质。这涉及到卷积算子的维数分离，如\[Speeding up convolutional neural networks with low rank expansions\]中所讨论的。这里，使用明显更少的FLOPs来近似原始操作。[Rethinking the inception architecture for computer vision]将$3\times 3$卷积核分成$3\times 1$和$1\times 3$两种形状的连续卷积核。MobileNet模型更进一步，将通道与空间卷积分离开来，空间卷积也只应用于深度，见图1b。通过这样做，大多数计算转移到逐点卷积层，可以显著减少FLOPs。最后，ShuffleNet模型以类似于\[AlexNet\]的方式将逐点卷积层分组，从而解决了逐点卷积层中FLOPs的堆积问题。这导致了极大地减少了FLOPs，但对精度的影响很小，请参见\[ShuffleNet\]中的图1和图1c。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-09-19-EffNet/figure1.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图1. MobileNet和ShuffleNet与EffNet块的比较。“dw”表示深度卷积，“mp”表示最大池化，“ch”表示输出通道数，“gc”表示分组卷积。</div>
</center>

&emsp;方法的多样性说明了成功压缩CNN的方法是多种多样的。然而，大多数方法都假定一个大型开发模型，并根据效率进行调整。因此，当应用于一开始就很小的网络时，它们通常似乎达到了它们的极限。由于许多嵌入式系统有一个有限的规格，模型通常是在这些限制下设计的，而不是优化大型网络。在这样的环境下，\[MobileNet\]和\[ShuffleNet\]的局限性变得更加明显，从而为我们的EffNet模型奠定了基础，即使应用于较浅和较窄的模型，该模型也显示出相同的能力。

&emsp;最后，请注意，上面的方法并不相互排斥。例如，我们的模型也可以转换为定点，剪枝和对最佳超参数集进行优化。

## 3. 提高模型效率的构建块

&emsp;本节讨论提高效率的最常见实践。给出的结果有助于识别以前技术中的弱点，并以统一的EffNet块的形式构造合适的解决方案。由于实际原因，我们避免详细介绍下面实验的具体设置。相反，我们将在第5节中讨论它们的结果并展示它们的整体效果。

&emsp;多任务、竞争成本和交互式运行时间的组合对工业应用程序的模型大小有严格的限制。事实上，这些要求常常导致使用更为经典的计算机视觉算法，这些算法经过优化，能够非常快速地运行特定的任务。此外，监管限制常常禁止一个单一网络解决方案，因为它们需要备用系统和高度可解释的决策过程。因此，降低项目中所有小分类器的计算成本，可以将计算能力重新分配到更关键的位置，也可以实现更大容量的更深、更广的模型。

&emsp;探索以前工作的局限性发现，模型越小，在转换为MobileNet或ShuffleNet时损失的精度就越大，参见第5节。在分析这些建议修改的性质时，我们遇到了几个问题。

**瓶颈结构**&emsp;\[SqueezeNet\]中讨论的瓶颈结构将缩减因子8应用于块中输入通道数量与输出通道数量的关系。 一个ShuffleNet块使用4的缩减因子。然而，窄模型往往没有足够的通道来实现如此大幅度的缩减。在我们所有的实验中，我们都看到了准确性的下降，而不是适度的下降。因此，我们建议使用瓶颈因子2。此外，使用深度乘数为2的空间卷积（见下一段）也很有效，即第一个深度卷积层也使通道数量翻倍。

**步长和池化**&emsp;MobileNet和ShuffleNet模型都在其块中对深度空间卷积层应用了步长2。我们的实验表明这种做法存在两个问题。首先，与最大池化相比，我们多次看到精度下降。这在某种程度上是预期的，因为跨步卷积容易产生混叠。

&emsp;此外，将最大池化应用于空间卷积层不允许网络在将数据压缩到其传入大小的四分之一之前对数据进行正确编码。尽管如此，早期的池化意味着在块中接下来的层更cheap。为了在保持早期池化的优点的同时也放松数据压缩，我们建议使用可分离池化。与可分离卷积相似，我们首先在第一个空间卷积层之后应用一个$2\times 1$的池化核（具有相应的步长）。然后，池化的第二阶段跟在块最后的逐点卷积之后。

**可分离卷积**&emsp;由[Rethinking the inception architecture for computer vision]提出，但在其他方面经常被忽略，我们重新考虑连续可分离空间卷积的想法，即使用$3\times 1$和$1\times 3$层而不是单个$3\times 3$层。分离空间卷积可能只会在FLOPs方面产生很小的差异，但是，结合我们的池化策略，它变得更加重要。

**残差连接**&emsp;最初由[Deep residual learning for image recognition]提出，并很快被许多人采用，残差连接已成为标准实践。然而，论文也表明，残差连接大多有益于更深的网络。在使用残差连接的整个实验过程中，我们扩展了这一说法，并报告了准确性的持续下降。我们将此解释为对小型网络无法很好地处理大型压缩因子的主张的支持。

**分组卷积**&emsp;继\[ShuffleNet\]的有希望的结果之后，我们也用相似的结构进行了实验。最极端的设置是原始的ShuffleNet，最轻松的设置是将块中的最后一个逐点层进行分组。结果显示准确度明显下降。因此，我们避免使用分组卷积，尽管它具有诱人的计算优势。

**解决第一层的问题**&emsp;MobileNet和ShuffleNet都避免替换第一层。他们声称以这一层为开始很cheap。我们恭敬地表示反对，并相信每一个优化都很重要。在优化了网络中的所有其他层之后，第一层成比例地变大。在我们的实验中，用EffNet块替换第一个层可以节省大约30%的计算量。

## 4. EffNet模型

### 4.1 数据压缩

&emsp;通过分析第3节中讨论的各种方法的效果，我们发现小型网络对数据压缩非常敏感。在整个实验过程中，每一个导致更大瓶颈的实践也损害了准确性。为了更好地理解数据流概念，表1列出了通过Cifar10网络的不同阶段的输入维数。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-09-19-EffNet/table1.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表1. 选定模型中的数据流。人们可以直观地理解在早期阶段进行积极的数据压缩会如何损害准确性。压缩因子为4或4以上用红色标记。gc4表示分组数为4的分组卷积。</div>
</center>

### 4.2 EffNet块

&emsp;我们提出了一个高效的卷积块，它既解决了数据压缩问题，又实现了第3节的观点。我们将此块设计为一个通用结构，以无缝地替换瘦网络中的普通卷积层，但不限于此。

&emsp;我们以与[Inception V2]类似的方式开始，将$3\times 3$深度卷积分解为两个线性层。这允许我们在第一个空间层之后进行池化操作，从而为第二层节省计算。

&emsp;然后我们沿着空间维度对子采样进行分割。如表1和图1所示，我们在第一次深度卷积后应用$1\times 2$的最大池化核。对于第二个子采样，我们选择用$2\times 1$的核和相应的步长来代替普通的逐点卷积。这实际上有相同的FLOPs，但准确性略好。

&emsp;在第3节的初步实验之后，我们决定放松第一个逐点卷积的瓶颈因子。我们不使用四分之一的输出通道，而是采用0.5的因子（最小通道数量为6）作为首选。

## 5. 实验

&emsp;对于评估部分，我们选择了符合我们一般设置的数据集：少量的类和相对较小的输入分辨率。从\[MobileNet\]和\[ShuffleNet\]的结果来看，它们显示出与baseline相当的准确度，我们没有理由相信EffNet的表现会有很大差异。因此，我们专注于更小的模型。对于每个数据集，我们都要快速手动搜索baseline可能的超参数，以满足需求：两到三个隐藏层和少量的通道。其他体系结构则简单地替换卷积层，而不改变超参数。

&emsp;每个实验重复五次，以抵消随机初始化的影响。

&emsp;我们既没有使用数据增强，也没有对[The unreasonable effectiveness of noisy data for fine-grained recognition]提出的额外数据进行预训练。超参数也没有优化，因为我们的目标是用EffNet块替换每个给定网络中的卷积层。

&emsp;我们使用Tensorflow和使用Adam优化器进行训练，学习率为0.001，$\beta=0.75$。

&emsp;作为补充实验，我们评估了一个更大的EffNet模型，其FLOPs与baseline大致相同。在下面的表中，它被称为large，由表2和表4中的两个附加层和更多通道（或者表3中的更多通道）组成。我们还训练了ShuffleNet和MobileNet的版本，使用更多的通道来匹配我们的EffNet模型的FLOPs，从而评估架构的可比性。

### 5.1 Cifar10

&emsp;Cifar10是计算机视觉中一个简单、基本的数据集，是我们要改进的任务类型的一个很好的例子。它的图像很小，只代表有限的类。与MobileNet和ShuffleNet相比，我们取得了显著的改进，且只需要比baseline少7倍的浮点运算（表2）。我们将这种改进与网络的额外深度联系起来，这意味着EffNet块模拟的是一个更大、更深的网络，它不像其他模型那样不适合。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-09-19-EffNet/table2.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表2. Cifar10数据集上的模型比较</div>
</center>

### 5.2 Street View House Numbers

&emsp;与Cifar10类似，SVHN基准也是评估简单网络的常用数据集。该数据由32×32像素的patch组成，patch以数字为中心，并带有相应的标签。表3显示了实验结果，无论在精度还是在FLOPs方面都支持我们的EffNet模型。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-09-19-EffNet/table3.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表3. SVHN数据集的模型比较</div>
</center>

### 5.3 German Traffic Sign Recognition Benchmark

&emsp;GTSRB数据集是一个稍微较旧的数据集，但它在大多数当前的驾驶员辅助应用中仍然非常相关。它有超过50,000张图片和大约43个类，它呈现了一个相当小的任务，数据变化很大，因此是一个有趣的基准。由于即使是很小的网络也很快开始对这些数据进行过拟合，因此我们将输入图像的大小调整为$32\times 32$，并在输出层之前使用dropout，其概率为50%。结果如表4所示，也支持我们的EffNet模型。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-09-19-EffNet/table4.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表4. GTSRB数据集上的模型比较</div>
</center>

## 6. 与MobileNet V2比较

&emsp;随着[MobileNet V2]和我们的工作同时出现，我们扩展了这项工作并编写了一个快速比较。最后，我们展示了如何使用一些小的调整，我们在精度方面超过[MobileNet V2]，同时计算起来也同样昂贵。

### 6.1 体系结构的比较

&emsp;[MobileNet V2]和这项工作都是沿着卷积运算的某些维数进行分离，以节省计算。与[MobileNet V2]不同，我们还将空间的二维卷积核分离为两个一维卷积核。通过这样做，我们确实注意到在整个实验中，精确度下降了0.5%左右，但是它允许一个更有效的实现，并且需要更少的计算。

&emsp;为了解决数据压缩问题，[MobileNet V2]建议通过将输入通道的数量乘以4-10的因子来显著增加整个块中的数据量。这使得压缩相对于相应块的输入来说不那么具有侵略性，同时还将其移动到块的末尾，即反向瓶颈。他们进一步认识到ReLU函数的一个有趣的、常常被忽视的特性。当使用批归一化层时，ReLU将一半的数据设置为零，从而进一步压缩数据。为了解决这个问题，[MobileNet V2]在每个块的末尾使用一个线性逐点卷积。实际上，它们得到一个线性层，接着另一个非线性逐点层，即B∗(A∗x)，其中x是输入，A是第一层，B是第二层。移除层A只是简单地迫使网络将层B作为函数B*A来学习。我们的实验也显示出对层A的存在不关心。然而，我们显示在层A的顶部使用leaky ReLU显著地提高了性能。

### 6.2 EffNet调整

&emsp;考虑到最新的实验，我们通过引入三个小的调整来修改我们的架构。首先，考虑瓶颈结构，我们将第一个逐点层的输出通道定义为块的输入通道而不是输出通道的函数。与[MobileNet V2]相似，但不那么极端，通道的数量由下式给出：

$$
\lfloor\frac{inputChannels*expansionRate}{2}\rfloor
$$

&emsp;其次，空间卷积中的深度乘数，我们之前只是在某些情况下增加了它，现在已经自然地集成到我们的架构中并设置为2。

&emsp;最后，我们用一个leaky ReLU替换逐点层上的ReLU。

&emsp;请注意，无论是深度卷积的激活函数还是网络的第一层，实验结果都不是决定性的。为了简单起见，我们使用了一个ReLU，并注意到leaky ReLU和线性空间卷积偶尔都更可取。在接下来的实验中，第一个层是带有最大池化的普通卷积层。

### 6.3 实验

&emsp;我们使用与第5节相同的数据集，但针对的是另一种比较。现在我们评估三个模型。

1. 修正后的EffNet模型

2. 原始的MobileNet V2模型

3. 我们建议对MobileNet V2模型进行的修改，用池化来代替跨步卷积，用leaky ReLU来代替线性瓶颈。下表中称为mob_imp（移动改进）。

&emsp;模型分别以2、4和6种不同的拓展率进行评估，mob_imp仅以6的拓展率进行测试。5、6和7显示了我们修改后的体系结构如何在大多数设置中实现比[MobileNet V2]更好的准确性，而只有略微更多的FLOPs。此外，尽管mob_imp模型的性能优于我们的模型，但是它的计算成本要高得多。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-09-19-EffNet/table5.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表5. 不同扩展率下Cifar10数据集上MobileNet V2和EffNet的比较</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-09-19-EffNet/table6.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表6. 不同扩展率下SVHN数据集上MobileNet V2和EffNet的比较</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-09-19-EffNet/table7.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表7. 不同扩展率下GTSRB数据集上MobileNet V2和EffNet的比较</div>
</center>

## 7. 结论

&emsp;我们提出了一种新的CNNs卷积块，称为EffNet，它承诺在保持甚至超过baseline精度的同时，显著减少计算量。我们的统一块设计是为了确保在嵌入式和移动硬件的应用程序中安全替换普通卷积层。由于网络被减少到baseline的FLOPs的一小部分，我们的方法有两个优点，第一个是更快的推理，第二个是可以应用更大、更深的网络。我们还展示了这样一个更大的网络在需要类似数量的操作时明显优于baseline。

---

## 个人看法

&emsp;emmmm讲道理，这篇文章看得我迷迷糊糊的，作者讲起来感觉乱乱的，果然C会的文章跟顶会的距离还是有点大啊。文章最主要的创新点还是在于提出的EffNet构建块，不仅用上了深度卷积（这是把通道与空间两个维度给分割开来了），还用上了空间可分离卷积（把宽和高两个维度给分割开来）。它把原来的深度卷积分成在宽（用$1\times 3$卷积核）和高（用$3\times 1$卷积核）两个维度上分别进行，而且在宽度维度进行完卷积后还做了个宽度方向的池化（用$1\times 2$池化），那么下面再进行高度方向的卷积时就有了计算量上的减少。
