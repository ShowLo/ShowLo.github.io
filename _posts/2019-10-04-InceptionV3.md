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

&emsp;然而，Inception架构的复杂性使得对网络进行更改变得更加困难。如果天真地将体系结构按比例放大，大量的计算收益可能会立即丢失。此外，[Going deeper with convolutions]并没有对导致GoogLeNet体系结构的各种设计决策的影响因素提供清晰的描述。这使得在保持其效率的同时使其适应新的用例变得更加困难。例如，如果认为有必要增加某些Inception样式模型的容量，则只需将所有滤波器组大小的数量加倍的简单转换将导致计算成本和参数数量增加4倍。在许多实际情况下，这可能被证明是禁止或不合理的，尤其是在相关收益不大的情况下。在这篇文章中，我们从描述一些一般的原则和优化思想开始，这些原则和优化思想被证明对于以有效的方式扩展卷积网络是有用的。尽管我们的原则不仅限于Inception类型的网络，但由于Inception样式构建模块的通用结构足够灵活，可以自然地合并这些约束，因此在这种情况下更易于观察。这是通过大量使用维度缩减和Inception模块的并行结构来实现的，这样可以减轻结构更改对附近组件的影响。尽管如此，这样做仍然需要谨慎，因为应该遵循一些指导原则来保持模型的高质量。

## 2. 总体设计原则

&emsp;在这里，我们将描述一些基于大规模实验的设计原则，这些设计原则使用了卷积网络的各种架构选择。在这一点上，以下原则的效用是推测性的，未来需要额外的实验证据来评估其有效范围。尽管如此，严重偏离这些原则往往会导致网络质量的恶化，并在发现这些偏离的情况下修复这些情况，从而改进体系结构。

1. 避免表征瓶颈，特别是在网络的早期。前馈网络可以用从输入层到分类器或回归器的无环图来表示。这为信息流定义了一个清晰的方向。对于将输入与输出分开的任何剪切，都可以访问通过剪切的信息量。应该避免极端压缩的瓶颈。通常，表征形式的大小应从输入到输出逐渐减小，直到达到用于当前任务的最终表征形式。理论上，信息内容不能仅仅通过表征的维度来评价，因为它丢弃了重要的因素，如相关结构；维度仅仅提供了对信息内容的粗略估计。

2. 高维表征更容易在网络中进行局部处理。增加卷积网络中每个块的激活允许更多的解纠缠特征。由此产生的网络将训练得更快。

3. 空间聚合可以通过较低维度的嵌入来完成，而不会损失太多或任何表征能力。例如，在执行更广泛的卷积（如$3\times 3$）之前，可以在空间聚合之前减少输入表征的维数，而不会产生严重的不良影响。我们假设这样做的原因是，如果在空间聚合环境中使用输出，则在降维期间相邻单元之间的强相关性将导致更少的信息丢失。考虑到这些信号应该是容易压缩的，降维甚至可以促进更快的学习。

4. 平衡网络的宽度和深度。通过平衡每个阶段的滤波器数量和网络的深度，可以达到网络的最佳性能。增加网络的宽度和深度可以提高网络的质量。然而，如果两者并行增加，则可以达到对恒定计算量的最优改进。因此，计算预算应该在网络的深度和宽度之间以一种平衡的方式分配。

&emsp;虽然这些原则可能有意义，但使用它们来提高开箱即用网络的质量并不容易。我们的想法是，只有在模棱两可的情况下才能明智地使用它们。

## 3. 大滤波器尺寸卷积的分解

&emsp;GoogLeNet网络的许多最初的收获来自于对维度缩减的大量使用，就像Lin等人的“Network in Network”架构一样。这可以看作是一种以计算高效方式分解卷积的特殊情况。在视觉网络中，邻近激活的输出是高度相关的。因此，我们可以期望在聚合之前减少它们的激活，从而产生类似的表达性局部表征。

&emsp;在这里，我们探索了在各种情况下分解卷积的其他方法，特别是为了提高解的计算效率。由于Inception网络是完全卷积的，每个权值对应于每个激活的一个乘法。因此，任何计算成本的降低都会导致参数量的减少。这意味着通过适当的因式分解，我们可以得到更多的解纠缠（disentangled）参数，从而得到更快的训练。同样，我们可以使用节省的计算和内存来增加网络的滤波器组大小，同时保持在单台计算机上训练每个模型副本的能力。

### 3.1 分解成更小的卷积

&emsp;在计算方面，使用较大的空间滤波器（例如$5\times 5$或$7\times 7$）的卷积往往非常昂贵。例如，在具有m个滤波器的网格上，具有n个滤波器的$5\times 5$卷积比具有相同滤波器数量的$3\times 3$卷积的计算成本高$25/9=2.78$倍。当然，$5\times 5$过滤器可以捕获较早层中距离较远的单元的激活之间的信号之间的相关性，因此减小滤波器的几何尺寸会带来较大的表达成本。然而，我们可以提出这样一个问题，即一个$5\times 5$的卷积是否可以被一个参数更少、输入尺寸和输出深度相同的多层网络所取代。如果放大$5\times 5$卷积的计算图，我们会看到每个输出看起来像一个小型的全连接网络，在其输入上的$5\times 5$块上滑动（见图1）。由于我们正在构建视觉网络，因此再次利用平移不变性并用两层卷积体系结构代替全连接的组件似乎是很自然的：第一层是$3\times 3$卷积，第二层是在第一层$3\times 3$输出网格之上的全连接层（见图1）。在输入激活网格上滑动此小型网络归结为用两层$3\times 3$卷积替换$5\times 5$卷积（将图4与5进行比较）。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-10-04-InceptionV3/figure1.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图1. 小型网络取代了5×5卷积</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-10-04-InceptionV3/figure4.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图4. 原始的Inception模块，如[Going deeper with convolutions]中所述。</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-10-04-InceptionV3/figure5.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图5. 在Inception模块中，每个5×5的卷积被两个3×3的卷积所取代，如第2节的原则3所建议的。</div>
</center>

&emsp;这种设置通过共享相邻块之间的权重来减少参数量。为了分析预期的计算成本节省，我们将作出一些简化假设，适用于典型的情况：我们可以假设$n=\alpha m$，也就是说，我们希望通过常数因子$\alpha$来改变每个单元的激活数目。由于$5\times 5$卷积是聚合的，$\alpha$通常略大于1（在GoogLeNet的情况下大约为1.5）。将$5\times 5$层替换为两层，分两步进行扩展似乎是合理的：在两步中均将滤波器数量增加$\sqrt{\alpha}$。为了通过选择$\alpha=1$（无扩展）简化我们的估算，可以用两个$3\times 3$卷积层来表示滑动该网络，其重用相邻图块之间的激活。这样，通过这种分解，我们最终得到了$\frac{9+9}{25}$倍的净减少，相对增益为28％。由于每个参数在每个单元的激活计算中恰好使用一次，因此对参数量的节省完全相同。然而，此设置引发了两个一般性问题：此替换是否会导致表达能力损失？如果我们的主要目标是分解计算的线性部分，是否不建议在第一层保留线性激活？我们已经进行了几个控制实验（例如，请参见图2），在分解的所有阶段，使用线性激活总是不如使用线性整流单元。我们将此增益归因于网络可以学习到的变化空间的增强，特别是当我们对输出激活进行批归一化时。当对降维组件使用线性激活时，可以看到类似的效果。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-10-04-InceptionV3/figure2.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图2. 两个Inception模型之间的几个控制实验之一，其中一个使用线性+ReLU层的分解，另一个使用两个ReLU层。在进行了386万次操作后，前者稳定在76.2％，而后者在验证集上达到了77.2％的top-1准确性。</div>
</center>

### 3.2 空间分解为非对称卷积

&emsp;以上结果表明，使用大于$3\times 3$的滤波器进行卷积可能通常没有用，因为它们总是可以减少为$3\times 3$卷积层的序列。我们仍然可以问是否应该把它们分解成更小的，例如$2\times 2$的卷积。然而，事实证明，使用非对称卷积甚至可以做得比$2\times 2$更好，例如$n\times 1$。例如，使用$3\times 1$卷积再加上$1\times 3$卷积等效于滑动具有与$3\times 3$卷积相同的感受野的两层网络（请参见图3）。如果输入和输出滤波器的数量相等，则对于相同数量的输出滤波器，两层解决方案仍便宜33％。相比之下，将一个$3\times 3$的卷积分解成两个$2\times 2$的卷积只节省了11%的计算量。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-10-04-InceptionV3/figure3.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图3. 小型网络取代了3×3卷积。该网络的下层由具有3个输出单元的3×1卷积组成。</div>
</center>

&emsp;理论上，我们可以更进一步地论证，可以用$1\times n$卷积和$n\times 1$卷积来代替任何$n\times n$卷积，并且随着n的增长，计算成本节省显著增加（见图6）。在实践中，我们发现，采用这种分解在早期层上不能很好地工作，但是它对中等网格大小（$m\times m$特征图，其中m在12和20之间的范围）给出了非常好的结果。在该层次上，通过使用$1\times 7$卷积，然后是$7\times 1$卷积，可以实现非常好的结果。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-10-04-InceptionV3/figure6.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图6. n×n卷积分解后的Inception模块。在我们提出的体系结构中，我们为17×17网格选择n=7。（使用原则3选择滤波器尺寸）</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-10-04-InceptionV3/figure7.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图7. 具有扩展滤波器组输出的Inception模块。该架构用于最粗糙（8×8）网格，以促进高维表征，如第2节的原则2所建议的。我们只在最粗糙的网格上使用此解决方案，因为与空间聚合相比，生成高维稀疏表示是最关键的，因为局部处理（1×1卷积）的比率增加了。</div>
</center>

## 4. 辅助分类器的效用

&emsp;[Going deeper with convolutions]引入了辅助分类器的概念，以提高非常深的网络的收敛性。最初的动机是将有用的梯度推到较低的层，使它们立即有用，并通过在非常深的网络中解决梯度消失问题来提高训练的收敛性。Lee等人也认为辅助分类器促进了更稳定的学习和更好的收敛。有趣的是，我们发现辅助分类器并未在训练初期改善收敛性：在两个模型都达到高精度之前，带有和不带有side-head的网络的训练进展看起来几乎是相同的。在训练快要结束时，带有辅助分支的网络开始超越没有任何辅助分支的网络的精度，并达到略高的平稳期。

&emsp;此外，[Going deeper with convolutions]在网络的不同阶段使用了两个side-head。下层辅助支路的拆除对网络的最终质量没有任何不利影响。连同上一段中的较早观察，这意味着[Going deeper with convolutions]的原假设很可能是错误的，这些假设认为这些分支有助于演化低级特征。相反，我们认为辅助分类器起到了正则化的作用。如果侧分支是批归一化的或有一个dropout层，则网络的主分类器性能更好，这一事实支持了这一点。这也为批归一化充当正则化器的猜想提供了一个薄弱的支持证据。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-10-04-InceptionV3/figure8.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图8. 最后17×17层顶部的辅助分类器。side head各层的批归一化可使top-1准确度获得0.4%的绝对增益。</div>
</center>

## 5. 有效减少网格大小

&emsp;传统上，卷积网络使用一些池化操作来减小特征图的网格大小。为了避免出现表征瓶颈，在应用最大或平均池化之前，将扩展网络滤波器的激活维度。例如，从一个具有k个滤波器的$d\times d$网格开始，如果我们想得到一个具有2k滤波器的$\frac{d}{2}\times\frac{d}{2}$网格，我们首先需要计算具有2k个滤波器的stride为1的卷积，然后应用额外的池化步骤。这意味着总体计算成本主要由使用$2d^{2}k^{2}$次操作的较大网格上的昂贵卷积所控制。一种可能是切换到使用卷积的池化，从而变为$2(\frac{d}{2})^{2}k^{2}$，将计算成本减少四分之一。但是，这会造成表征瓶颈，因为表征的整体维数下降到$(\frac{d}{2})^{2}k$，导致网络的表征能力更差（请参见图9）。与其这样做，我们建议另一种变体，进一步降低计算成本，同时消除表征瓶颈（见图10）。我们可以使用两个并行的stride为2的块：P和C。P是一个池化层（平均池化或最大池化激活），它们的stride都为2，其滤波器组如图10所示级联在一起。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-10-04-InceptionV3/figure9.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图9. 减少网格大小的两种可选方法。左边的解决方案违反了原则1，即不引入第2节中的表征瓶颈。右边的版本在计算上要贵3倍。</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-10-04-InceptionV3/figure10.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图10. Inception模块，减少网格大小，同时扩大滤波器组。它既便宜又避免了原则1所提出的表征瓶颈。右边的图表示相同的解决方案，但从网格大小的角度来看，而不是操作。</div>
</center>

## 6. Inception-v3

&emsp;在这里，我们将上面的点连接起来，并提出了一个新的架构，该架构在ILSVRC 2012分类基准上具有改进的性能。我们的网络布局如表1所示。注意，我们根据3.1节中描述的相同思想，将传统的$7\times 7$卷积分解为3个$3\times 3$卷积。对于网络的Inception部分，我们在35×35处有3个传统的Inception模块，每个模块有288个滤波器。使用第5节所述的网格缩减技术，可以将其缩小为带有768个滤波器的$17\times 17$网格。接下来是如图6所示的分解Inception模块的5个实例。使用图10所示的网格缩减技术，可以将其缩减为$8\times 8\times 1280$的网格。在最粗的$8\times 8$层，我们有两个如图7所示的Inception模块，每个块的级联输出滤波器组大小为2048。该网络的详细结构，包括Inception模块内部的滤波器组的大小，在补充材料中给出，在此提交的tar文件中的model.txt中给出。然而，我们已经观察到，只要遵守第2节的原则，网络的质量在变化时是相对稳定的。虽然我们的网络有42层，但是我们的计算成本只比GoogLeNet高2.5倍左右，而且它的效率仍然比VGGNet高很多。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-10-04-InceptionV3/table1.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表1. 提出的网络架构概要。每个模块的输出大小是下一个模块的输入大小。只要适用，我们将使用图10所示的缩减技术的变体来减小Inception块之间的网格大小。我们用零填充标记了卷积，这是用来保持网格大小的。零填充也用在那些不减少网格大小的Inception模块中。所有其他层不使用填充。选择不同的滤波器组大小以遵循第2节中的原则4。</div>
</center>

## 7. 通过标签平滑实现模型正则化

&emsp;在这里，我们提出了一种机制，通过估计训练过程中标签丢失的边缘化作用来对分类器层进行正则化。

&emsp;对于每个训练样本x，我们的模型计算每个标签$k\in\{1\ldots K\}$的概率：$p(k|x)=\frac{exp(z_{k})}{\sum_{i=1}^{K}exp(z_{i})}$。其中$z_{i}$是对数或未归一化的对数概率。在本训练样本中，考虑标签$q(k|x)$上的真实分布，因为归一化有$\sum_{k}q(k|x)=1$。为简洁起见，我们省略p和q对样本x的依赖关系。我们将样本的损失定义为交叉熵：$\ell=-\sum_{k=1}^{K}log\left(p(k)\right)q(k)$。将其最小化等效于最大化标签的期望对数似然，其中标签是根据真实分布$q(k)$选择的。交叉熵损失相对于对数$z_{k}$是可微的，因此可以用于深度模型的梯度训练。梯度有一个相当简单的形式：$\frac{\partial\ell}{\partial z_{k}}=p(k)-q(k)$，其限定在-1和1之间。

&emsp;考虑单个真实标签y的情况，因此对于所有$k\ne y$，$q(y)=1$且$q(k)=0$。在这种情况下，最小化交叉熵相当于最大化正确标签的对数似然。对于具有标签y的特定样本，对数似然最大化为$q(k)=\delta_{k,y}$，其中$\delta_{k,y}$为狄拉克函数，$k=y$时等于1，否则为0。对于有限的$z_{k}$，无法达到此最大值，但是对于所有$k\ne y$来说，如果$z_{y}\gg z_{k}$，可以接近最大值，即对应于真实标签的对数比所有其他对数大得多时，可以接近该最大值。然而，这可能导致两个问题。首先，它可能导致过拟合：如果模型学会为每个训练样本分配全部概率给groundtruth标签，则不能保证将其泛化能力。其次，它鼓励最大对数和所有其他对数之间的差异变大，这与有界梯度$\frac{\partial\ell}{\partial z_{k}}$相结合，降低了模型的适应能力。直观地说，这是因为模型对其预测过于自信。

&emsp;我们提出了一个机制来鼓励模型少一点自信。如果目标是最大化训练标签的对数似然，这么做可能不是所期望的，但它确实使模型正则化并使其更具适应性。方法很简单。考虑独立于训练样本x的标签的分布$u(k)$和平滑参数$\epsilon$。对于真实标签为y的训练样本，我们将标签分布$q(k|x)=\delta_{k,y}$替换为

$$
q'(k|x)=(1-\epsilon)\delta_{k,y}+\epsilon u(k)
$$

它是原始真实分布$q(k|x)$和固定分布$u(k)$的混合，权重分别为$1-\epsilon$和$\epsilon$。这可以看作是按如下方式获得的标签k的分布：首先，将其设置为真实标签$k=y$；然后，以概率q用从分布$u(k)$中抽取的样本替换k。我们建议使用标签上的先验分布作为$u(k)$。在我们的实验中，我们使用均匀分布$u(k)=1/K$，因此

$$
q'(k|x)=(1-\epsilon)\delta_{k,y}+\frac{\epsilon}{K}
$$

我们将真实标签分布的这种变化称为标签平滑正则化或LSR。

&emsp;请注意，LSR实现了防止最大对数比其他所有对数大得多的预期目标。事实上，如果这种情况发生，那么一个$q(k)$将趋近于1，而其他所有$q(k)$将趋近于0。这将导致与$q'(k)$产生较大的交叉熵，因为与$q(k)=\delta_{k,y}$不同，所有$q'(k)$都有一个正的下界。

&emsp;通过考虑交叉熵可以得到LSR的另一种解释：

$$
H(q',p)=-\sum_{k=1}^{K}log\,p(k)q'(k)=(1-\epsilon)H(q,p)+\epsilon H(u,p)
$$

因此，LSR相当于将一个单一的交叉熵损失$H(q,p)$替换为一对这样的损失$H(q,p)$和$H(u,p)$。第二个损失惩罚了预测标签分布$p$与先验$u$的偏差，相对权重为$\frac{\epsilon}{1-\epsilon}$。注意，由于$H(u,p)=D_{KL}(u||p)+H(u)$和$H(u)$是固定的，因此可以用KL散度等效地捕获此偏差。当$u$是均匀分布时，$H(u,p)$是预测的分布$p$与均匀分布的不相似程度的度量，也可以用负熵$-H(p)$来度量（但不相等）；我们还没有尝试过这种方法。

&emsp;在我们使用$K=1000$类的ImageNet实验中，我们使用$u(k)=1/1000$和$\epsilon=0.1$。对于ILSVRC 2012，我们发现top-1错误率和top-5错误率都有0.2%的绝对改善（参见表3）。

## 8. 训练方法

&emsp;我们已经使用TensorFlow分布式机器学习系统以随机梯度训练了我们的网络，该系统使用了50个副本，每个副本在NVidia Kepler GPU上运行，批大小为32，共100个epoch。我们早期的实验使用了衰减为0.9的momentum，而我们最好的模型是使用衰减为0.9和$\epsilon=1.0$的RMSProp实现的。我们使用的学习率为0.045，每两个epoch使用0.94的指数衰减。此外，使用阈值2.0的梯度裁剪被发现对稳定训练是有用的。使用随时间计算的参数的运行平均值执行模型评估。

## 9. 低分辨率输入的性能

&emsp;视觉网络的一个典型用例是用于检测的后分类，例如在多框环境中。这包括分析图像的一个相对较小的区域，该区域包含具有某些背景的单个目标。任务是确定区域的中心部分是否与某个目标相对应，如果是，则确定该目标的类。挑战在于物体往往相对较小且分辨率较低。这就提出了如何正确处理低分辨率输入的问题。

&emsp;普遍的看法是，采用更高分辨率感受野的模型往往会显著提高识别性能。 然而，重要的是要区分提高第一层感受野分辨率的影响和更大的模型容量和计算的影响。如果我们只是改变输入的分辨率而不进一步调整模型，那么我们最终会使用计算成本更低的模型来解决更困难的任务。当然，由于减少了计算工作量，这些解决方案已经变得松散，这是很自然的。为了做出准确的评估，模型需要分析模糊的迹象，以便能够“幻想”出精细的细节。这在计算上是很昂贵的。因此，问题仍然是：如果计算量保持不变，那么更高的输入分辨率有多大帮助。确保相等计算量的一个简单方法是，在输入分辨率较低的情况下减少前两层的步长，或者简单地删除网络的第一个池化层。

&emsp;为此，我们进行了以下三个实验：

1. $299\times 299$感受野，步长2，第一层后最大池化。

2. $151\times 151$感受野，步长1，第一层后最大池化。

3. $79\times 79$感受野，步长1，第一层后无池化。

&emsp;这三种网络的计算成本几乎相同。尽管第三个网络稍微便宜一些，但池化层的成本是很小的（在总成本的1%以内）。在每种情况下，对网络进行训练，直到收敛并在ImageNet ILSVRC 2012分类基准的验证集上测量其质量。结果见表2。虽然低分辨率的网络需要更长的时间来训练，但最终结果的质量与高分辨率的网络相当接近。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-10-04-InceptionV3/table2.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表2. 当感受野大小不同，但计算代价不变时，识别性能的比较。</div>
</center>

&emsp;但是，如果仅根据输入分辨率天真地降低网络大小，那么网络的性能会差得多。但是，这将是不公平的比较，因为我们将在较困难的任务上比较便宜16倍的模型。此外，表2的这些结果表明，可以考虑在R-CNN环境中使用专用的高成本低分辨率网络来处理较小的目标。

## 10. 实验结果与比较

&emsp;表3显示了关于第6节中描述的我们所提出的架构（Inception-v3）的识别性能的实验结果。每个Inception-v3行显示累积更改的结果，包括突出显示的新修改加上所有先前的修改。Label Smoothing是指第7节中描述的方法。Factorized $7\times 7$包含将第一个$7\times 7$卷积层分解为$3\times 3$卷积层序列的更改。BN-auxiliary是指辅助分类器的全连接层也是批归一化的，而不仅仅是卷积。我们将表3的最后一行中的模型称为Inception-v3，并在multi-crop和集成设置下评估其性能。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-10-04-InceptionV3/table3.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表3. single-crop实验结果，比较了对各种影响因素的累积影响。我们将我们的数据与Ioffe等人发表的最佳single-crop推断进行了比较。对于“Inception-v3-”行，更改是累积的，并且后续的每一行除了前面的更改外，还包括新的更改。最后一行是指所有更改，以下称为“Inception-v3”。不幸的是，He等人只报告了10-crop的评价结果，而不是single-crop的结果，如下表4所示。</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-10-04-InceptionV3/table4.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表4. 单模型、multi-crop的实验结果，比较了各种影响因素的累积影响。我们将我们的数据与在ILSVRC 2012分类基准上发布的最佳单模型推理结果进行比较。</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-10-04-InceptionV3/table5.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表5. 集成评估结果，比较多模型、multi-crop报告的结果。我们的数据与ILSVRC 2012分类基准上发布的最佳集成推断结果进行了比较。*除了top-5的结果外，其他所有结果都是在验证集上得到的。集成在验证集上产生3.46％的top-5错误率。</div>
</center>

&emsp;我们的所有评估都是在ILSVRC-2012验证集上的48238个非黑名单样本上完成的，如[Imagenet large scale visual recognition challenge]所建议的。我们还对所有50000个样本进行了评估，结果top-5错误率的结果大约差0.1％，top-1错误率的结果差约0.2％。在本文即将发布的版本中，我们将在测试集上验证整体结果，但是在春季我们对BN-Inception的最后评估时，表明测试和验证集误差之间的相关性非常好。

## 11. 结论

&emsp;我们提供了一些设计原则来扩大卷积网络的规模，并在Inception体系结构的背景下对它们进行了研究。这种指导可以产生高性能的视觉网络，与更简单、更单一的架构相比，它的计算成本相对较低。在ILSVR 2012分类中，针对single-crop评估，我们质量最高的Inception-v3版本达到了21.2％的top-1和5.6％的top-5错误率，开创了新的技术水平。与Ioffe等人所述的网络相比，这是通过相对适度（2.5倍）的计算成本增加来实现的。尽管如此，我们的解决方案比基于更密集网络的最佳公布结果使用的计算量要少得多：我们的模型比He等人的结果要好得多——将top-5（top-1）误差相对减少25%（14%），同时计算成本降低了6倍，使用的参数（估计）也至少减少了5倍。较低的参数量和附加的正则化与批归一化辅助分类器和标签平滑相结合，允许在相对较小的训练集上训练高质量的网络。

---

## 个人看法

&emsp;文章继承了BN-Inception中将$5\times 5$卷积分解为两个$3\times 3$卷积的思路以减少计算量，同时还引入了空间分离卷积，也就是将$n\times n$卷积给分解成$1\times n$和$n\times 1$两个卷积，可以大大减少计算量，特别是对于像$5\times 5$和$7\times 7$这样的大卷积核。这两个思路对于轻量级网络的设计还是有一定的借鉴的，接下来可以做做看，毕竟今年好多篇NAS搜出来的网络都用到了不小于$5\times 5$的卷积核。
