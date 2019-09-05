---
layout:     post
title:      SqueezeNet--AlexNet级精度，参数减少50倍，模型大小小于0.5MB
subtitle:   SqueezeNet--AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size
date:       2019-09-03
author:     CJR
header-img: img/2019-09-03-SqueezeNet/post-bg.jpg
catalog: true
mathjax: true
tags:
    - Lightweight Network
    - SqueezeNet
    - CNN
---

## SqueezeNet

&emsp;SqueezeNet是较早出现的轻量化网络之一，该文章2016年的时候就挂了出来，然后投了ICLR 2017，不过被拒了。。。原文可见[SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)。

---

## 摘要

&emsp;最近对深度卷积神经网络（CNNs）的研究主要集中在提高精度上。对于给定的精度级别，通常可以找到多个达到该精度级别的CNN架构。同样的精确度，较小的CNN架构至少提供了三个优势：（1）较小的CNN在分布式训练中需要较少的跨服务器通信。（2）较小的CNN需要更少的带宽将新模型从云端导出到自动驾驶汽车上。（3）较小的CNN更适合部署在FPGAs等内存有限的硬件上。为了提供所有这些优势，我们提出了一个小型CNN架构SqueezeNet。SqueezeNet以少50倍的参数在ImageNet上实现了AlexNet级的精度。此外，通过模型压缩技术，我们可以将SqueezeNet压缩到小于0.5MB（比AlexNet小510倍）。

>SqueezeNet架构可以在这里下载：<https://github.com/DeepScale/SqueezeNet>

## 1. 引言与动机

&emsp;最近对深度卷积神经网络（CNNs）的研究主要集中在提高计算机视觉数据集的准确性。对于给定的精度级别，通常存在多个CNN架构来实现该精度级别。在精度相当的情况下，参数较少的CNN架构具有以下几个优点：

* **更高效的分布式训练.**&emsp;服务器之间的通信是限制分布式CNN训练可扩展性的因素。对于分布式数据并行训练，通信开销与模型中参数的数量成正比。简而言之，小模型训练更快，因为需要更少的通信。

* **减少向客户端导出新模型时的开销.**&emsp;对于自动驾驶，特斯拉等公司会定期将新模型从服务器上复制到客户的汽车上。这种做法通常被称为无线更新。《消费者报告》发现，特斯拉的自动驾驶仪的半自动驾驶功能的安全性随着最近的无线更新而不断提高。然而，今天典型的CNN/DNN模型的无线更新可能需要大量的数据传输。使用AlexNet，这将需要从服务器到汽车的240MB通信。较小的模型需要较少的通信，使得频繁的更新更加可行。

* **可行的FPGA和嵌入式部署.**&emsp;FPGA通常只有不到10MB的片内内存，没有片外内存或存储器。对于推理来说，一个足够小的模型可以直接存储在FPGA上，而不是被内存带宽所限制，同时视频帧可以实时流经FPGA。此外，当在专用集成电路（ASIC）上部署CNN时，一个足够小的模型可以直接存储在芯片上，而较小的模型可以使ASIC适合一个较小的芯片。

&emsp;正如您所看到的，较小的CNN架构有几个优点。考虑到这一点，我们将直接关注发现CNN架构的问题，与已知模型相比，该架构的参数更少，但精度相当。我们发现了这样一种结构，我们称之为SqueezeNet。此外，我们还尝试了一种更有规律的方法来搜索新的CNN架构的设计空间。

&emsp;本文的其余部分组织如下。在第二部分中，我们回顾了相关的工作。然后，在第3和第4节中，我们描述和评价了SqueezeNet体系结构。之后，我们将注意力转向理解CNN架构设计的选择如何影响模型的大小和准确性。通过对类SqueezeNet架构的设计空间的探索，我们得到了这一认识。在第5节中，我们在CNN微架构上进行空间探索设计，我们将其定义为各个层和模块的组织和维数。在第6节中，我们对CNN宏架构进行了空间探索设计，我们将其定义为CNN中的高层层次组织。最后，我们在第7节中总结。简而言之，第3节和第4节对于CNN的研究人员以及那些只想将SqueezeNet应用于新应用程序的实践者都是有用的。其余部分针对的是打算设计自己的CNN架构的高级研究人员。

## 2. 相关工作

### 2.1 模型压缩

&emsp;我们工作的首要目标是找到一个只有很少参数的模型，同时保持准确性。为了解决这个问题，一个明智的方法是使用一个现有的CNN模型，并以有损的方式压缩它。事实上，围绕模型压缩这个主题已经出现了一个研究社区，并且已经报道了几种方法。Denton等人的一种相当直接的方法是将奇异值分解（SVD）应用到一个经过预训练的CNN模型中。Han等人开发了网络剪枝，从一个预训练的模型开始，用0代替阈值以下的参数，形成稀疏矩阵，最后对稀疏CNN进行几次迭代训练。最近，Han等人通过将网络剪枝与量化（小于等于8比特）和哈夫曼编码相结合，扩展了它们的工作，以创建一种称为深度压缩的方法，并进一步设计了一种称为EIE的硬件加速器，该加速器直接在压缩模型上运行，实现了大幅加速和节能。

### 2.2 CNN微体系架构

&emsp;卷积在人工神经网络中已经使用了至少25年；LeCun等人在20世纪80年代末推动了CNNs在数字识别应用中的普及。在神经网络中，卷积滤波器通常是三维的，以高度、宽度和通道为关键维度。当应用于图像时，CNN滤波器通常在其第一层中有3个通道（即RGB），并且在随后的每一层$L_{i}$中，滤波器的通道数与$L_{i-1}$层具有的滤波器的通道数相同。LeCun等人的早期工作使用$5\times 5\times Channels$滤波器，而最近的VGG架构广泛使用$3\times 3$滤波器。Network-in-Network和GoogLeNet体系结构家族等模型的部分层使用$1\times 1$滤波器。

>注：这里以及后面的滤波器其实就是指我们常说的卷积

&emsp;随着设计深度CNNs的趋势，为每一层手工选择滤波器尺寸变得很麻烦。为了解决这一问题，提出了由具有特定固定组织的多个卷积层组成的各种更高层次的构建块或模块。例如，GoogLeNet论文提出了Inception模块，它由许多不同尺寸的滤波器组成，通常包括$1\times 1$和$3\times 3$，有时是$5\times 5$，有时是$1\times 3$和$3\times 1$。然后，许多这样的模块被组合在一起，可能还有额外的ad-hoc层，以形成一个完整的网络。我们使用术语CNN微架构来表示各个模块的特定组织和维度。

### 2.3 CNN宏体系架构

&emsp;CNN微体系结构是指单个的层和模块，而我们将CNN宏体系结构定义为多个模块在系统级组织成一个端到端CNN体系结构。

&emsp;也许在最近的文献中，CNN宏观架构研究最广泛的话题是网络深度（即层数）的影响。Simoyan和Zisserman提出了具有12到19层的CNNs的VGG家族，并报道了更深的网络在ImageNet-1k数据集上产生更高的精度。K. He等人提出了更深层次的CNNs，最多可达30层，可以提供更高的ImageNet精度。

&emsp;跨多层或多个模块连接的选择是CNN宏结构研究的一个新兴领域。残差网络（ResNet）和Highway网络都建议使用跨多个层的连接，例如将第3层激活与第6层激活相加。我们将这些连接称为旁路连接。ResNet的作者提供了一个有和没有旁路连接的34层CNN的A/B比较；添加旁路连接可使ImageNet的top-5精度提高2个百分点。

### 2.4 神经网络设计空间探索

&emsp;神经网络（包括深度和卷积神经网络）有很大的设计空间，对于微架构、宏架构、求解器和其他超参数有很多的选择。对于这些因素如何影响神经网络的准确性（即设计空间的形状），社区想要获得直觉似乎是很自然的。神经网络的设计空间探索（DSE）的大部分工作都集中在开发自动化的方法来寻找提供更高精度的神经网络体系结构。这些自动化的DSE方法包括贝叶斯优化、模拟退火、随机搜索和遗传算法。值得赞扬的是，每一篇论文都提供了一个案例，其中提出的DSE方法生成了一个神经网络体系结构，与具有代表性的baseline相比，该体系结构具有更高的精度。然而，这些论文并没有试图对神经网络设计空间的形状提供直观的认识。在本文的后面，我们避开了自动化方法——相反，我们重构CNNs的方式是，我们可以进行有原则的A/B比较，以研究CNN架构决策如何影响模型大小和准确性。

&emsp;在接下来的部分中，我们首先提出并评估进行和不进行模型压缩的SqueezeNet体系结构。然后，我们探讨了微体系结构和宏体系结构中的设计选择对类SqueezeNet CNN体系结构的影响。

## 3. SqueezeNet：少参数保精度

&emsp;在这一部分中，我们首先概述了我们的CNN架构的设计策略，这些架构的参数很少。然后，我们介绍了Fire模块，这是我们构建CNN架构的新模块。最后，利用我们的设计策略构建了以Fire模块为主的SqueezeNet。

### 3.1 架构设计策略

&emsp;在本文中，我们的首要目标是找到具有较少参数的CNN架构，同时保持具有竞争力的准确性。为此，我们在设计CNN架构时采用了三种主要策略：

&emsp;*策略1.* **用$1\times 1$滤波器替换$3\times 3$滤波器.**&emsp;给定一定数量卷积滤波器的预算，我们将选择使这些滤波器中的大多数为$1\times 1$，因为$1\times 1$滤波器的参数比$3\times 3$滤波器少9倍。

&emsp;*策略2.* **减少$3\times 3$滤波器的输入通道数.**&emsp;考虑一个完全由$3\times 3$滤波器组成的卷积层。该层的参数总数为(输入通道数)\*(滤波器数)\*(3\*3)。因此，为了在CNN中保持一个小的参数总数，不仅要减少$3\times 3$滤波器的数量（见上面的策略1），而且也要减少$3\times 3$滤波器的输入通道数。我们使用squeeze层减少$3\times 3$滤波器的输入通道数，这将在下一节中描述。

&emsp;*策略3.* **在网络的后期下采样，这样卷积层就有了大的激活图.**&emsp;在卷积网络中，每个卷积层生成一个输出激活图，其空间分辨率至少为$1\times 1$，并且通常比$1\times 1$大得多。这些激活图的高度和宽度由以下因素控制：（1）输入数据的大小（如$256\times 256$大小的图像）和（2）在CNN架构中进行下采样的层的选择。最常见的是，通过在一些卷积或池层中设置（stride>1），将下采样设计到CNN架构中。如果网络的早期层有很大的stride，那么大多数层都会有小的激活图。相反，如果网络中的大多数层的stride为1，且大于1的stride集中在网络的末端，那么网络中的许多层将具有较大的激活图。我们的直觉是，在其他条件相同的情况下，大的激活图（由于下采样的延迟）可以导致更高的分类精度。的确，K. He和H. Sun将延迟下采样应用于四种不同的CNN架构，在每种情况下，延迟下采样都会导致更高的分类精度。

&emsp;策略1和策略2是在试图保持准确性的同时，明智地减少CNN中的参数数量。策略3是关于在有限的参数预算下最大化精度。接下来，我们描述Fire模块，它是我们CNN架构的构建模块，使我们能够成功地使用策略1、2和3。

### 3.2 Fire模块

&emsp;我们定义Fire模块如下。Fire模块包括：压缩（squeeze）卷积层（只有$1\times 1$滤波器），注入由$1\times 1$和$3\times 3$卷积滤波器组合而成的扩展（expand）层；我们在图1中对此进行了说明。在Fire模块中广泛使用$1\times 1$滤波器是3.1节中策略1的一个应用。我们在Fire模块中公开了三个可调维（超参数）：$s_{1\times 1}$、$e_{1\times 1}$和$e_{3\times 3}$。在Fire模块中，$s_{1\times 1}$是squeeze层中滤波器的数量（全部为$1\times 1$），$e_{1\times 1}$是expand层中$1\times 1$滤波器的数量，$e_{3\times 3}$是expand层中$3\times 3$滤波器的数量。当我们使用Fire模块时，我们将$s_{1\times 1}$设置为小于$(e_{1\times 1}+e_{3\times 3})$，因此squeeze层有助于限制$3\times 3$滤波器的输入通道数，如3.1节中的策略2所示。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-09-03-SqueezeNet/figure1.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图1. 微体系结构视图：Fire模块中卷积滤波器的组织。在本例中，$s_{1\times 1}=3$、$e_{1\times 1}=4$和$e_{3\times 3}=4$。我们展示了卷积滤波器而不是激活</div>
</center>

### 3.3 SqueezeNet架构

&emsp;我们现在描述SqueezeNet CNN架构。我们在图2中演示了SqueezeNet，从一个独立的卷积层（conv1）开始，然后是8个Fire模块（fire2-9），最后是一个卷积层（conv10）。从网络的开始到结束，我们逐渐增加每个fire模块的滤波器数量。SqueezeNet在conv1、fire4、fire8和conv10层之后执行stride为2的最大池化；这些相对较迟的池化操作遵循3.1节中的策略3。我们在表1中给出了完整的SqueezeNet架构。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-09-03-SqueezeNet/figure2.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图2. 我们的SqueezeNet架构的宏观视图。左：SqueezeNet（第3.3节）；中：带简单旁路的SqueezeNet（第6节）；右：带复杂旁路的SqueezeNet（第6节）。</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-09-03-SqueezeNet/table1.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表1. SqueezeNet架构</div>
</center>

#### 3.3.1 SqueezeNet的其他细节

&emsp;为了简洁起见，我们省略了表1和图2中关于SqueezeNet的许多细节和设计选择。我们在下面提供这些设计选择。这些选择背后的直觉可以在下面引用的论文中找到。

* 为了使从$1\times 1$和$3\times 3$滤波器的输出激活具有相同的高度和宽度，我们在expand模块的$3\times 3$滤波器的输入数据中添加一个填充0的1像素边框。

* ReLU用于squeeze和expand层的激活。

* 比例为50%的dropout用于fire9模块之后。

* 注意SqueezeNet缺少全连接层；这个设计选择的灵感来自NiN架构。

* 当训练SqueezeNet时，我们从0.04的学习率开始，并在整个训练过程中线性降低学习率，如[Systematic evaluation of cnn advances on the imagenet]所述。有关训练方案的详细信息（例如批处理大小、学习率、参数初始化），请参考我们的与caffe兼容的配置文件：<https://github.com/DeepScale/SqueezeNet>。

* Caffe框架本身并不支持包含多个滤波器分辨率（如$1\times 1$和$3\times 3$）的卷积层。为了解决这个问题，我们使用两个独立的卷积层来实现expand层：一个带有$1\times 1$滤波器的层和一个带有$3\times 3$滤波器的层。然后，我们在通道维度中将这些层的输出连接在一起。这在数值上等价于实现一个包含$1\times 1$和$3\times 3$滤波器的层。

&emsp;我们以Caffe CNN框架定义的格式发布了SqueezeNet配置文件。然而，除了Caffe，出现了其他几个CNN框架，包括MXNet、Chainer、Keras和Torch。每一个都有自己的原生格式来表示CNN架构。也就是说，大多数这些库使用相同的底层计算后端，如cuDNN和MKL-DNN。为了与其他CNN软件框架兼容，研究团体已经移植了SqueezeNet CNN架构：

* SqueezeNet的MXNet移植：<https://github.com/hariag/SqueezeNet/commit/0cf57539375fd5429275af36fc94c774503427c3>

* SqueezeNet的Chainer移植：<https://github.com/ejlb/squeezenet-chainer>

* SqueezeNet的Keras移植：<https://github.com/DT42/squeezenet_demo>

* SqueezeNet的Fire模块的Torch移植：<https://github.com/Element-Research/dpnn/blob/master/FireModule.lua>

## 4. 评估SqueezeNet

&emsp;现在我们将注意力转向评估SqueezeNet。在2.1节中回顾的每一篇CNN模型压缩论文中，目标都是压缩AlexNet模型，该模型使用ImageNet（ILSVRC 2012）数据集进行训练以对图像进行分类。因此，在评估SqueezeNet时，我们使用AlexNet和相关的模型压缩结果作为比较的基础。

&emsp;在表2中，我们根据最近的模型压缩结果回顾了SqueezeNet。基于SVD的方法能够将预训练的AlexNet模型压缩5倍，同时将top-1精度降低到56.0%。网络剪枝在保持ImageNettop-1精度为57.2%，top-5精度为80.3%的基础上，模型尺寸减少了9倍。深度压缩在保持baseline精度水平的同时，模型尺寸减少了35倍。现在，有了SqueezeNet，我们的模型尺寸比AlexNet缩小了50倍，同时达到或超过了AlexNet的top-1和top-5的精度。我们在表2中总结了上述所有结果。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-09-03-SqueezeNet/table2.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表2. 比较SqueezeNet和模型压缩方法。model size指的是在训练模型中存储所有参数所需的字节数。</div>
</center>

&emsp;我们似乎已经超越了模型压缩的最先进成果：即使使用未压缩的32比特值来表示模型，SqueezeNet的模型大小也比模型压缩的最好效果要小1.4倍，同时保持或超过baseline精度。到目前为止，一个悬而未决的问题是：小模型是否适合压缩，或者小模型是否“需要”密集浮点值提供的所有表示能力？为了找出答案，我们使用33%的稀疏性和8比特量化对SqueezeNet进行深度压缩。这产生了一个0.66MB的模型（比32比特AlexNet小363倍），其精度与AlexNet相当。此外，在SqueezeNet上应用6比特量化的深度压缩和33%的稀疏性，我们得到了一个0.47MB（510倍小于32比特AlexNet）的模型，具有相同的精度。**我们的小模型确实经得起压缩。**

&emsp;此外，这些结果表明，深度压缩不仅适用于具有许多参数的CNN架构（如AlexNet和VGG），而且能够压缩已经紧凑的、完全卷积的SqueezeNet架构。经过深度压缩压缩后的SqueezeNet在保持baseline精度的同时压缩了10倍。总之：通过将CNN架构创新（SqueezeNet）与最先进的压缩技术（深度压缩）相结合，我们实现了模型尺寸的510倍缩减，与baseline相比，精确度没有降低。

&emsp;最后，请注意深度压缩使用codebook作为其将CNN参数量化到6比特或8比特精度的方案的一部分。因此，在大多数商用处理器上，使用在深度压缩中开发的方案实现$\frac{32}{8}=4$倍（8比特量化）或$\frac{32}{6}=5.3$倍（6比特量化）的加速并非易事。然而，Han等人开发了定制的硬件——高效推理引擎（EIE）——可以更有效地计算codebook-量化的CNNs。此外，在我们发布SqueezeNet后的几个月里，P. Gysel开发了一个名为Ristretto的策略，用于将SqueezeNet线性量化到8比特。具体来说，Ristretto用8比特进行计算，并以8比特数据类型存储参数和激活。在SqueezeNet推理中使用Ristretto策略进行8比特计算时，Gysel发现当使用8比特而不是32比特数据类型时，精度下降了不到1%。

## 5. CNN微架构设计空间探索

&emsp;到目前为止，我们已经提出了小模型的架构设计策略，并按照这些原则创建了SqueezeNet，发现SqueezeNet比AlexNet小50倍，且精度相当。然而，SqueezeNet等模型存在于CNN架构的一个广泛而未被探索的设计空间中。现在，在第5和第6节中，我们将探讨设计空间的几个方面。我们将架构探索分为两个主要主题：微架构探索（每个模块的层维度和配置）和宏架构探索（模块和其他层的高层端到端组织）。

&emsp;在本节中，我们设计和执行实验的目的是根据我们在3.1节中提出的设计策略，对微架构设计空间的形状提供直观的认识。请注意，我们这里的目标不是在每个实验中都最大化精确度，而是理解CNN架构选择对模型大小和精确度的影响。

### 5.1 CNN微架构元参数

&emsp;在SqueezeNet中，每个Fire模块都有我们在3.2节中定义的三维超参数$s_{1\times 1}$、$e_{1\times 1}$和$e_{3\times 3}$。SqueezeNet有8个Fire模块，共24维超参数。为了对类SqueezeNet结构的设计空间进行广泛的扫描，我们定义了一组更高级别的元参数，它控制CNN中所有Fire模块的尺寸。我们将$base_{e}$定义为CNN中第一个Fire模块中expand滤波器的数量。在每$freq$个Fire模块之后，我们将expand滤波器的数量增加$incr_{e}$。换句话说，对于Fire模块i，expand滤波器的数量是$e_{i}=base_{e}+(incr_{e}*\lfloor \frac{i}{freq}\rfloor)$。在一个Fire模块的第$freq$个expand层中，有的滤波器为$1\times 1$，有的滤波器为$3\times 3$；我们定义$e_{i}=e_{i,1\times 1}+e_{i, 3\times 3}$和$pct_{3\times 3}$（在[0, 1]范围内，在所有Fire模块上共享）为$3\times 3$的expand滤波器的百分比。也就是说，$e_{i, 3\times 3}=e_{i}*pct_{3\times 3}$，$e_{i, 1\times 1}=e_{i}*(1-pct_{3\times 3})$。最后，我们使用一个称为压缩（squeeze）比（SR）的元参数定义Fire模块squeeze层中的滤波器数量（同样在[0, 1]范围内，为所有Fire模块共享）：$s_{i,1\times 1}=SR*e_{i}$（或等效为$s_{i,1\times 1}=SR*(e_{i,1\times 1}+e_{i, 3\times 3})$）。SqueezeNet（表1）是我们使用前面提到的一组元参数生成的一个示例架构。具体来说，SqueezeNet有以下元参数：$base_{e}=128$、$incr_{e}=128$、$pct_{3\times 3}=0.5$、$freq=5$和$SR=0.125$。

### 5.2 压缩（squeeze）比

&emsp;在3.1节中，我们提出通过使用squeeze层来减少$3\times 3$滤波器看到的输入通道数量，从而减少参数的数量。我们将压缩比（SR）定义为squeeze层中滤波器的数量与expand层中滤波器的数量之比。我们现在设计了一个实验来研究压缩比对模型尺寸和精度的影响。

&emsp;在这些实验中，我们使用SqueezeNet（图2）作为出发点。与SqueezeNet一样，这些实验使用了以下元参数：$base_{e}=128$、$incr_{e}=128$、$pct_{3\times 3}=0.5$和$freq=5$。 我们训练多个模型，其中每个模型具有不同的压缩比（SR），范围在[0.125, 1.0]。在图3（a）中，我们展示了这个实验的结果，图中的每个点都是从零开始训练的独立模型。图中SqueezeNet为$SR=0.125$的点。从图中可以看出，将SR提高到0.125以上，可以进一步提高ImageNet上的top-5准确率，从4.8MB模型的80.3%（即AlexNet级别）提高到19MB模型的86.0%。精度稳定在86.0%且$SR=0.75$（19MB模型），设置$SR=1.0$在不提高精度的情况下进一步增加了模型大小。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-09-03-SqueezeNet/figure3.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图3. 微体系架构设计空间探索</div>
</center>

### 5.3 权衡$1\times 1$和$3\times 3$滤波器

&emsp;在3.1节中，我们提出了用$1\times 1$滤波器代替$3\times 3$滤波器来减少CNN中参数的数量。一个悬而未决的问题是，CNN滤波器的空间分辨率有多重要？

&emsp;VGG架构在大多数层的滤波器中都具有$3\times 3$的空间分辨率；GoogLeNet和Network-in-Network（NiN）在某些层中有$1\times 1$滤波器。在GoogLeNet和NiN中，作者只是提出了一个特定数量的$1\times 1$和$3\times 3$过滤器，而没有进一步分析。在这里，我们试图阐明$1\times 1$和$3\times 3$滤波器的比例如何影响模型的大小和精度。

&emsp;在本实验中，我们使用了以下元参数：$base_{e}=incr_{e}=128$、$freq=2$、$SR=0.500$，$pct_{3\times 3}$的变化范围从1%到99%。换句话说，每个Fire模块的expand层都有一个预定义的滤波器数量，这些滤波器划分为$1\times 1$和$3\times 3$，这里我们将这些滤波器的旋钮从“大部分$1\times 1$”切换到“大部分$3\times 3$”。与之前的实验一样，这些模型有8个Fire模块，遵循图2中相同的层组织。实验结果如图3(b)所示。注意，图3（a）和图3（b）中的13MB模型是相同的体系结构：$SR=0.500$和$pct_{3\times 3}=50\%$。从图3（b）中可以看出，使用50%的$3\times 3$滤波器，top-5准确率为85.6%，进一步增加$3\times 3$滤波器的比例，导致模型尺寸更大，但在ImageNet上没有提高精度。

## 6. CNN宏架构设计空间探索

&emsp;到目前为止，我们已经探索了微架构层面的设计空间，即CNN各个模块的内容。现在，我们将探讨与Fire模块之间的高层连接有关的宏架构级别的设计决策。受ResNet的启发，我们探索了三种不同的架构：

* 普通的SqueezeNet（根据前面的章节）

* 在一些Fire模块之间采用简单的旁路连接的SqueezeNet

* 其余Fire模块之间具有复杂旁路连接的SqueezeNet

&emsp;我们在图2中演示了这三种类型的SqueezeNet。

&emsp;我们的简单旁路架构围绕Fire模块3、5、7和9添加旁路连接，要求这些模块学习输入和输出之间的残差函数。正如在ResNet中的一样，为了实现围绕Fire3的旁路连接，我们将Fire4的输入设置为等于（Fire2的输出+ Fire3的输出），其中+操作符是element-wise加法。这改变了这些Fire模块参数的正则化，并且，根据ResNet，可以提高最终的精度和/或训练整个模型的能力。

&emsp;一个限制是，在简单的情况下，输入通道的数量和输出通道的数量必须相同；因此，只有一半的Fire模块可以实现简单的旁路连接，如图2的中间图所示。当不能满足“相同数量的通道”需求时，我们使用一个复杂的旁路连接，如图2右边图所示。虽然简单旁路“只是一根线”，但我们将复杂旁路定义为包括$1\times 1$卷积层的旁路，其中滤波器数量设置为所需的输出通道数量。注意，复杂的旁路连接会向模型添加额外的参数，而简单的旁路连接则不会。

&emsp;除了改变正则化之外，我们还可以直观地看到，添加旁路连接将有助于缓解squeeze层引入的表征瓶颈。在SqueezeNet中，压缩比（SR）为0.125，这意味着每一个squeeze层的输出通道比相应的expand层少8倍。由于这种严重的维数减少，有限的信息可以通过squeeze层。然而，通过向SqueezeNet添加旁路连接，我们为信息在挤压层周围流动开辟了道路。

&emsp;我们使用图2中的三种宏架构对SqueezeNet进行了训练，并在表3中对精度和模型大小进行了比较。在整个宏体系架构探索过程中，我们按照表1中描述的那样，固定了与SqueezeNet匹配的微体系架构。复杂的旁路连接和简单的旁路连接都比传统的SqueezeNet结构获得了精度上的提高。有趣的是，简单旁路比复杂旁路具有更高的准确度。在不增加模型尺寸的情况下，添加简单的旁路连接在top-1精度上增加了2.9%，在top-5精度上增加了2.2%。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-09-03-SqueezeNet/table3.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表3. 使用不同宏架构配置的SqueezeNet精度和模型大小</div>
</center>

## 7. 结论

&emsp;在这篇论文中，我们提出了一个更有规律的方法来探索卷积神经网络的设计空间。为了实现这一目标，我们提出了SqueezeNet，这是一种CNN架构，它的参数比AlexNet少50倍，并且在ImageNet上保持了AlexNet级别的精度。我们还将SqueezeNet压缩到小于0.5MB，或者在不压缩的情况下比AlexNet小510倍。自2016年我们以技术报告的形式发表这篇论文以来，韩松和他的合作者们对SqueezeNet和模型压缩进行了进一步的实验。使用一种称为Dense-Sparse-Dense（DSD）的新方法，han等人在训练期间使用模型压缩作为正则化器来进一步提高精度，生成压缩后的SqueezeNet参数集，在ImageNet-1k上其精度提高1.2个百分点，并且生成未压缩的SqueezeNet参数集，其精度提高4.3个百分点（与表2中的结果相比）。

&emsp;我们在本文的开头提到，小模型更适合于FPGA上的片上实现。自我们发布了SqueezeNet模型以来，Gschwend开发了一个SqueezeNet的变体，并在FPGA上实现了它。正如我们所预料的，Gschwend能够将一个类SqueezeNet模型的参数完全存储在FPGA中，并且消除了对加载模型参数的片外内存访问的需要。

&emsp;在本文中，我们将ImageNet作为目标数据集。然而，将经过ImageNet训练的CNN表示应用于细粒度对象识别、图像中的标识识别、生成关于图像的句子等多种应用已成为一种常见的实践。ImageNet训练过的CNNs也被应用于一些与自动驾驶相关的应用，包括图像和视频中的行人和车辆检测，以及对道路形状的分割。我们认为SqueezeNet将成为适用于各种应用的良好候选CNN架构，尤其是那些小模型尺寸非常重要的应用。

&emsp;SqueezeNet是我们在广泛探索CNN架构设计空间的过程中发现的几个新的CNNs之一。我们希望SqueezeNet能够启发读者去思考和探索CNN架构设计空间的广阔可能性，并以一种更加系统的方式进行探索。

---

## 个人看法

&emsp;这篇文章相对而言较老了，其主要思想是较简单地通过大量的$1\times 1$卷积代替$3\times 3$卷积、同时减少$3\times 3$卷积对应的输入通道数达到减少参数量的目的，对应的核心就是Fire模块，模块中的squeeze层主要控制通道数，expand层则是大量利用$1\times 1$卷积代替$3\times 3$卷积。

&emsp;再参考这篇博客--[2.1 SqueezeNet V1思考](https://www.jianshu.com/p/6153cc19d6b7)，提到了SqueezeNet的几个缺点：

* SqueezeNet的侧重的应用方向是嵌入式环境，目前嵌入式环境主要问题是实时性。SqueezeNet通过更深的深度置换更少的参数量，虽然能网络的参数量少了，但是其丧失了网络的并行能力，测试时间反而会更长，这与目前的主要挑战是背道而驰的。

* 纸面上是减少了50倍的参数，但是问题的主要症结在于AlexNet本身全连接节点过于庞大（SqueezeNet中用全局平均池化代替），50倍参数的减少和SqueezeNet的设计并没有过大的关系，考虑去掉全连接之后3倍参数量的减少更为合适。

* SqueezeNet得到的模型是5MB左右，0.5MB的模型还要得益于Deep Compression。虽然Deep Compression也是这个团队的文章，但是将0.5MB这个模型大小列在文章的题目中显然不是很合适。

&emsp;而且文章的对比对象是AlexNet，在现在看来显然有点过时了。。。
