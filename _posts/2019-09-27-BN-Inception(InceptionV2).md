---
layout:     post
title:      批归一化--通过减少Internal Covariate Shift来加速深度网络训练
subtitle:   Batch Normalization--Accelerating Deep Network Training by Reducing Internal Covariate Shift
date:       2019-09-27
author:     CJR
header-img: img/2019-09-27-BN-Inception(InceptionV2)/post-bg.jpg
catalog: true
mathjax: true
tags:
    - Batch Normalization
    - Inception
    - CNN
---

## BN-Inception/Inception V2

&emsp;这篇文章是Inception系列的第二篇，虽然在之后的第三篇中提出了官方定义的Inception V2，不过大家貌似更喜欢把这篇文章提出的BN-Inception当作Inception V2，文章发表在了ICML 2015。原文可见[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)。

---

## 摘要

&emsp;由于训练过程中各层输入的分布随前一层参数的变化而变化，使得训练深度神经网络变得复杂。这降低了训练的速度，因为需要更低的学习速度和更仔细的参数初始化，并且使得用饱和非线性来训练模型变得非常困难。我们将这种现象称为Internal Covariate Shift，并通过归一化层输入来解决这个问题。我们的方法将归一化作为模型体系结构的一部分，并对每个训练小批量执行归一化，从而获得了它的优势。批归一化使我们可以使用更高的学习率，而对初始化则不必那么小心。它还作为一个正则化器，在某些情况下消除了dropout的需要。批归一化应用于最先进的图像分类模型，以少了14倍的训练步骤即可达到相同的精度，并且在很大程度上击败了原始模型。使用一组批归一化网络，我们对ImageNet分类的最佳发表结果进行了改进：达到4.9%的top-5验证误差（和4.8%的测试误差），超过了人类评分者的准确度。

## 1. 引言

&emsp;深度学习极大地提高了视觉、语言和许多其他领域的技术水平。随机梯度下降（SGD）已被证明是训练深度网络的一种有效方法，SGD变体如momentum和Adagrad已被用来实现最先进的性能。SGD优化网络参数$\Theta$，以减少损失

$$
\Theta=\mathop{\arg\min}_{\Theta}\frac{1}{N}\sum_{i=1}^{N}\ell\left(x_{i},\Theta\right)
$$

其中，$x_{1\ldots N}$为训练数据集。使用SGD，训练分步骤进行，在每个步骤中，我们考虑一个尺寸为m的小批量$x_{1\ldots m}$。小批量用于估计损失函数相对于参数的梯度，通过计算

$$
\frac{1}{m}\frac{\partial\ell(x_{i},\Theta)}{\partial\Theta}
$$

使用小批量的样本，而不是一次一个样本，在很多方面都是有帮助的。首先，小批量上损失的梯度是训练集上梯度的估计，其质量随着批量的增加而提高。其次，由于现代计算平台提供的并行性，批量的计算可能比单个样本的$m$次计算效率高得多。

&emsp;虽然随机梯度简单有效，但它需要对模型超参数进行仔细的调整，特别是优化中使用的学习率，以及模型参数的初值。由于每个层的输入都受到前面所有层的参数的影响，因此网络参数的微小变化会随着网络变得更深而放大，这使得训练变得更加复杂。

&emsp;层输入分布的变化带来了一个问题，因为层需要不断地适应新的分布。当一个学习系统的输入分布发生变化时，它被称为经历covariate shift。这通常是通过领域自适应来处理的。然而，covariate shift的概念可以扩展到整个学习系统之外，应用到它的各个部分，例如子网络或层。

&emsp;考虑一个网络计算

$$
\ell=F_{2}\left(F_{1}\left(u,\Theta_{1}\right),\Theta_{2}\right)
$$

其中$F_{1}$和$F_{2}$是任意变换，并且需要学习参数$\Theta_{1}$，$\Theta_{2}$以最小化损失$\ell$。可以将学习$\Theta_{2}$看作是将输入$x=F_{1}\left(u,\Theta_{1}\right)$馈入子网络

$$
\ell=F_{2}\left(x,\Theta_{2}\right)
$$

例如，梯度下降步骤

$$
\Theta_{2}\leftarrow\Theta_{2}-\frac{\alpha}{m}\sum_{i=1}^{m}\frac{\partial F_{2}\left(x,\Theta_{2}\right)}{\partial \Theta_{2}}
$$

（批量大小$m$和学习率$\alpha$）与具有输入x的独立网络$F_{2}$的完全相等。因此，使训练更有效的输入分布属性（例如，在训练和测试数据之间具有相同的分布）也适用于训练子网络。因此，随着时间的推移，x的分布保持不变是有利的。那么，$\Theta_{2}$不必重新调整以补偿x分布的变化。

&emsp;子网络输入的固定分布也会对子网络外部的层产生积极的影响。考虑一个具有sigmoid激活函数$z=g(Wu+b)$的层，其中u为层输入，权重矩阵W和偏置向量b为要学习的层参数，$g(x)=\frac{1}{1+exp(-x)}$。随着$|x|$的增大，$g(x)$趋于零。这意味着对于$x=Wu+b$的所有维度，除了绝对值较小的维度，流向u的梯度将消失，模型将缓慢训练。但是，由于x受W、b及以下各层参数的影响，在训练过程中，这些参数的变化可能会使x的多个维度进入非线性饱和状态，从而减慢收敛速度。这种效应随着网络深度的增加而增强。在实践中，饱和问题和由此产生的梯度消失通常通过使用线性整流单元$ReLU(x)=max(x,0)$，仔细的初始化和小的学习率。然而，如果我们能够确保非线性输入的分布在网络训练时保持更稳定，那么优化器就不太可能陷入饱和状态，训练就会加速。

&emsp;我们将深度网络内部节点在训练过程中的分布变化称为Internal Covariate Shift。消除它可以提供更快的训练。我们提出了一种新的机制，我们称之为批归一化，它向减少Internal Covariate Shift迈出了一步，并在此过程中显著加快了深度神经网络的训练。它通过固定层输入的均值和方差的归一化步骤来完成此操作。批处理归一化还通过减少梯度对参数或其初始值的依赖关系，对网络中的梯度流产生有益的影响。这使得我们可以使用更高的学习率，而不存在出现发散的风险。此外，批归一化正则化了模型，减少了dropout的需要。最后，批归一化通过防止网络陷入饱和模式，使使用饱和非线性成为可能。

&emsp;在第4.2节中，我们对性能最好的ImageNet分类网络应用了批归一化，结果表明，我们仅使用7%的训练步骤就可以匹配它的性能，并且可以进一步大大超过它的精度。使用经过批归一化训练的这类网络的集成，我们实现了在ImageNet分类的最著名结果的基础上提高的top-5错误率。

## 2. Towards Reducing Internal Covariate Shift

&emsp;我们将Internal Covariate Shift定义为训练过程中网络参数的变化引起的网络激活分布的变化。为了提高训练质量，我们寻求减少Internal Covariate Shift。随着训练的进行，通过固定层输入x的分布，我们期望提高训练速度。人们早就知道，如果网络训练的输入被白化，即线性变换为零均值和单位方差，并去相关，则网络训练收敛得更快。由于每一层都观察到下面各层所产生的输入，因此对每一层的输入进行相同的白化是有利的。通过对每一层的输入进行白化，我们将朝着实现输入的固定分布迈出一步，从而消除Internal Covariate Shift的不良影响。

&emsp;我们可以考虑在每一个训练步骤或某一间隔，通过直接修改网络或通过改变优化算法的参数来依赖于网络激活值来考虑白化激活。但是，如果这些修改与优化步骤穿插在一起，那么梯度下降步骤可能会尝试以需要更新归一化的方式更新参数，从而降低梯度步骤的效果。例如，考虑一个具有输入u的层，该层将学习到的偏差b相加，并通过减去在训练数据上计算的激活平均值来归一化结果：$\hat{x}=x-E[x]$，其中$x=u+b$（注：这里原论文是这个公式，不过实际应该是$\hat{x}=Wu+b$），$\mathcal{X}=\{x_{1\ldots N}\}$是训练集上x的值集，且$E[x]=\frac{1}{N}\sum_{i=1}^{N}x_{i}$。如果梯度下降步骤忽略了$E[x]$对b的依赖性，那么它将更新$b\leftarrow b+\Delta b$，其中$\Delta b\propto-\partial\ell/\partial\hat{x}$。然后$u+(b+\Delta b)-E[u+(b+\Delta b)]=u+b-E[u+b]$。因此，对b的更新和随后的归一化更改的组合不会导致层的输出发生更改，因此也不会导致损失发生改变。随着训练的继续，b将会无限期的增长，而损失则会保持不变。如果归一化不仅集中而且缩放了激活范围，那么这个问题可能会变得更加严重。在初始实验中，当归一化参数在梯度下降步骤外计算时，模型会爆炸，这是我们的经验观察结果。

&emsp;上述方法的问题在于梯度下降优化未考虑归一化发生的事实。为了解决这个问题，我们希望确保对于任何参数值，网络总是生成具有所需分布的激活。这样做将允许损失相对于模型参数的梯度考虑到归一化及其对模型参数的依赖性。再次将x视为层输入，将其视为向量，并将$\mathcal{X}$设为训练数据集上这些输入的集合。然后可以将归一化写成一个变换

$$
\hat{x}=Norm(x,\mathcal{X})
$$

这不仅依赖于给定的训练样本x，还依赖于所有样本$\mathcal{X}$——如果x是由另一层生成的，则每个样本都依赖于$\Theta$。对于反向传播，我们需要计算雅可比矩阵$\frac{\partial Norm(x,\mathcal{X})}{\partial x}$和$\frac{\partial Norm(x,\mathcal{X})}{\partial\mathcal{X}}$；忽视后一项将导致上文所述的爆炸。在此框架下，对层输入进行白化是昂贵的，因为它需要计算协方差矩阵$Cov[x]=E_{x\in\mathcal{X}}[xx^{T}]-E[x]E[x]^{T}$及其平方根倒数，从而产生白化后的激活$Cov[x]^{-1/2}(x-E[x])$，以及这些变换的导数用于反向传播。这促使我们寻找一种替代方法，以可微的方式执行输入归一化，并且不需要在每次更新参数之后分析整个训练集。

&emsp;以前的一些方法使用在单个训练样本上计算的统计数据，或者在图像网络的情况下，使用给定位置上的不同feature map计算的统计数据。但是，这会通过放弃激活的绝对规模来改变网络的表示能力。我们希望通过将训练样本中的激活相对于整个训练数据的统计进行归一化，从而保护网络中的信息。

## 3. 通过小批量统计进行归一化

&emsp;由于每一层输入的完全白化是昂贵的，并且不是处处可微的，所以我们做了两个必要的简化。首先，我们不需要对层输入和输出中的特征进行联合白化，而是通过使其均值为0，方差为1，独立地对每个标量特征进行归一化。对于具有d维输入$x=(x^{(1)}\ldots x^{(d)})$的层，我们将对每个维度进行归一化

$$
\hat{x}^{(k)}=\frac{x^{(k)}-E[x^{(k)}]}{\sqrt{Var[x^{(k)}]}}
$$

其中期望和方差是在训练数据集上计算的。如[Efficient backprop]所示，即使特征不去相关，这种标准化也能加快收敛速度。

&emsp;注意，对层的每个输入进行标准化可能会改变层所能表示的内容。例如，对一个sigmoid的输入进行标准化会将它们限制在非线性的线性范围内。为了解决这个问题，我们确保插入到网络中的转换可以表示恒等转换。为了实现这一点，我们为每个激活$x^{(k)}$引入一对参数$\gamma^{(k)}$，$\beta^{(k)}$，该参数缩放并平移归一化值：

$$
y^{(k)}=\gamma^{(k)}\hat{x}^{(k)}+\beta^{(k)}
$$

这些参数与原始模型参数一起学习，恢复网络的表征能力。事实上,通过设置$\gamma^{(k)}=\sqrt{Var[x^{(k)}]}$和$\beta^{(k)}=E[x^{(k)}]$，那么我们可以恢复原始激活，如果那是最佳选择。

&emsp;在batch设置中，每个训练步骤都基于整个训练集，我们将使用整个训练集对激活进行归一化。然而，在使用随机优化时，这是不切实际的。因此，我们进行了第二个简化：由于我们在随机梯度训练中使用了小批量，所以每个小批量生成每个激活的均值和方差的估计值。这样，用于归一化的统计量就可以充分参与梯度的反向传播。请注意，通过计算每维方差而不是联合协方差可以启用小批量；在联合的情况下，需要正则化，因为小批量的大小可能小于被白化的激活数量，从而导致奇异协方差矩阵。

&emsp;考虑一个大小为m的小批量$\mathcal{B}$。由于归一化是独立地应用于每个激活的，所以让我们将重点放在一个特定的激活$x^{(k)}$上，为了清楚起见省略k。在小批量中有这个激活的m个值，

$$
\mathcal{B}=\{x_{1\ldots m}\}
$$

令归一化后的值为$\hat{x}_{1\ldots m}$，它们的线性变换为$y_{1\ldots m}$。我们将转换

$$
BN_{\gamma,\beta}:x_{1\ldots m}\rightarrow y_{1\ldots m}
$$

称为批归一化转换。我们在算法1中给出了BN变换。在该算法中，$\epsilon$是一个常量，它被添加到小批量方差中以保证数值稳定性。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-09-27-BN-Inception(InceptionV2)/algorithm1.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">算法1. 批归一化变换，应用于小批量激活x</div>
</center>

&emsp;BN转换可以添加到网络中来操作任何激活。在符号$y=BN_{\gamma,\beta}(x)$中，我们指出要学习参数$\gamma$和$\beta$，但应注意，在每个训练样本中BN变换都不会独立处理激活。相反，$BN_{\gamma,\beta}(x)$既依赖于训练样本，也依赖于小批量中的其他样本。缩放和平移的值$y$被传递到其他网络层。归一化的激活$\hat{x}$是我们转换的内部，但它们的存在至关重要。只要每个小批量的元素都是从同一个分布中抽取的，如果忽略$\epsilon$，则任意$\hat{x}$的值分布的期望值为0，方差为1。这可以通过观察$\sum_{i=1}^{m}\hat{x}_{i}=0$和$\frac{1}{m}\sum_{i=1}^{m}\hat{x}_{i}^{2}=1$，并考虑期望值来看到。每个归一化的激活$\hat{x}$可以看作是由线性变换$y^{(k)}=\gamma^{(k)}\hat{x}^{(k)}+\beta^{(k)}$组成的子网的输入，然后是原始网络完成的其他处理。这些子网络的输入都有固定的均值和方差，虽然这些归一化的$\hat{x}$的联合分布在训练过程中可能会发生变化，但我们希望归一化输入的引入能够加快子网络的训练，从而加快整个网络的训练。

&emsp;在训练过程中，我们需要通过该变换反向传播损失$\ell$的梯度，并计算与BN变换参数有关的梯度。我们使用链式规则，如下所示（简化前）：

$$
\frac{\partial\ell}{\partial\hat{x}_{i}}=\frac{\partial\ell}{\partial y_{i}}\cdot\gamma
$$
$$
\frac{\partial\ell}{\partial\sigma_{\mathcal{B}}^{2}}=\sum_{i=1}^{m}\frac{\partial\ell}{\partial\hat{x}_{i}}\cdot\left(x_{i}-\mu_{\mathcal{B}}\right)\cdot\frac{-1}{2}\left(\sigma_{\mathcal{B}}^{2}+\epsilon\right)^{-3/2}
$$
$$
\frac{\partial\ell}{\partial\mu_{\mathcal{B}}}=\left(\sum_{i=1}^{m}\frac{\partial\ell}{\partial\hat{x}_{i}}\cdot\frac{-1}{\sqrt{\sigma_{\mathcal{B}}^{2}+\epsilon}}\right)+\frac{\partial\ell}{\partial\sigma_{\mathcal{B}}^{2}}\cdot\frac{\sum_{i=1}^{m}-2\left(x_{i}-\mu_{\mathcal{B}}\right)}{m}
$$
$$
\frac{\partial\ell}{\partial x_{i}}=\frac{\partial\ell}{\partial \hat{x}_{i}}\cdot\frac{1}{\sqrt{\sigma_{\mathcal{B}}^{2}+\epsilon}}+\frac{\partial\ell}{\partial\sigma_{\mathcal{B}}^{2}}\cdot\frac{2\left(x_{i}-\mu_{\mathcal{B}}\right)}{m}+\frac{\partial\ell}{\partial\mu_{\mathcal{B}}}\cdot\frac{1}{m}
$$
$$
\frac{\partial\ell}{\partial\gamma}=\sum_{i=1}^{m}\frac{\partial\ell}{\partial y_{i}}\cdot\hat{x}_{i}
$$
$$
\frac{\partial\ell}{\partial\beta}=\sum_{i=1}^{m}\frac{\partial\ell}{\partial y_{i}}
$$

因此，BN变换是一个可微变换，它将归一化的激活引入到网络中。这确保了在模型进行训练时，各层可以继续学习呈现较少Internal Covariate Shift的输入分布，从而加快训练。此外，应用于这些归一化激活的学习仿射变换允许BN变换表示恒等变换并保持网络容量。

### 3.1 批归一化网络的训练与推理

&emsp;为了批归一化网络，我们根据算法1指定激活的子集，并为它们中的每一个插入BN变换。以前接收$x$作为输入的任何层，现在接收$BN(x)$。采用批归一化的模型可以使用批梯度下降或小批量大小$m>1$的随机梯度下降或其任何变体（如Adagrad）进行训练。依赖于小批量的激活的归一化允许有效的训练，但在推理期间既没有必要也不可取；我们想让输出只依赖于输入。为此，一旦对网络进行了训练，我们将使用总体（而不是小批量）统计数据来使用归一化

$$
\hat{x}=\frac{x-E[x]}{\sqrt{Var[x]+\epsilon}}
$$

忽略$\epsilon$，这些归一化的激活具有与训练过程中相同的均值0和方差1。我们使用无偏方差估计$Var[x]=\frac{m}{m-1}\cdot E_{\mathcal{B}}[\sigma_{\mathcal{B}}^{2}]$，其中期望是在大小为$m$的训练小批量上的以及$\sigma_{\mathcal{B}}^{2}$是他们的样本方差。用移动平均代替，我们可以跟踪模型训练时的精度。由于在推理过程中均值和方差是固定的，归一化只是对每个激活应用线性变换。它可以进一步由缩放$\gamma$和平移$\beta$组成，以产生一个替代$BN(x)$的线性变换。算法2总结了训练批归一化网络的过程。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-09-27-BN-Inception(InceptionV2)/algorithm2.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">算法2. 训练一个批归一化网络</div>
</center>

### 3.2 批归一化卷积网络

&emsp;批归一化可以应用于网络中的任何一组激活。在这里，我们重点介绍由仿射变换后跟元素级非线性组成的变换：

$$
z=g\left(Wu+b\right)
$$

其中W和b是模型的学习参数，而$g(\cdot)$是非线性，例如sigmoid或ReLU。这个公式包括全连通层和卷积层。我们通过对$x=Wu+b$进行归一化，在非线性之前加上BN变换。我们也可以对层输入u进行归一化，但是由于u很可能是另一个非线性的输出，所以在训练过程中，它的分布形状很可能会发生变化，限制它的一阶矩和二阶矩并不能消除covariate shift。相比之下，$Wu+b$更可能具有对称、非稀疏分布，即“更高斯”；使其归一化可能产生具有稳定分布的激活。

&emsp;请注意，由于我们对$Wu+b$进行了归一化，因此偏差b可以忽略，因为其作用将被随后的均值减法抵消（偏差的作用由算法1中的$\beta$所包含）。因此，$z=g\left(Wu+b\right)$被下式所替代：

$$
z=g\left(BN(Wu)\right)
$$

其中BN变换独立地应用于$x=Wu$的每个维度，每个维度具有单独的一对学习参数$\gamma^{(k)}$、$\beta^{(k)}$。

&emsp;对于卷积层，我们还希望归一化遵循卷积属性——这样，同一特征图的不同元素，在不同的位置，以相同的方式归一化。为此，我们在所有位置的一个小批量中共同归一化了所有激活。在算法1中，我们使$\mathcal{B}$为跨越一个小批量的元素和空间位置的特征图中所有值的集合——因此，对于大小为$m$的小批量和大小为$p\times q$的特征图，我们使用大小为$m'=|\mathcal{B}|=m\cdot pq$的有效小批量。我们每个特征图而不是每个激活都学习一对参数$\gamma^{(k)}$和$\beta^{(k)}$。算法2也进行了类似的修改，以便在推断过程中BN变换对给定特征图中的每个激活应用相同的线性变换。

### 3.3 批归一化允许更高的学习率

&emsp;在传统的深度网络中，过高的学习率可能导致梯度的爆炸或消失，以及陷入不好的局部极小值。批归一化有助于解决这些问题。通过对整个网络的激活进行归一化，可以防止对参数的细微变化在梯度上放大为更大的、次优的激活变化；例如，它可以防止训练陷入非线性的饱和状态。

&emsp;批归一化也使训练对参数的范围更有弹性。通常情况下，较大的学习速率会增加层参数的范围，从而在反向传播过程中放大梯度，导致模型爆炸。然而，使用批归一化，通过层的反向传播不受其参数范围的影响。事实上，对于标量$a$，

$$
BN(Wu)=BN((aW)u)
$$

我们可以证明

$$
\frac{\partial BN((aW)u)}{\partial u}=\frac{\partial BN(Wu)}{\partial u}
$$
$$
\frac{\partial BN((aW)u)}{\partial (aW)}=\frac{1}{a}\cdot\frac{\partial BN(Wu)}{\partial W}
$$

尺度不影响层的雅可比矩阵，也不影响梯度传播。此外，权值越大梯度越小，批归一化可以稳定参数的增长。

&emsp;我们进一步推测，批归一化可能导致层的雅可比矩阵的奇异值接近1，这对于训练是有益的。考虑两个具有归一化输入的连续层，以及这些归一化向量之间的转换：$\hat{z}=F(\hat{x})$。假设$\hat{x}$与$\hat{z}$为高斯不相关，且$F(\hat{x})\approx J\hat{x}$为给定模型参数的线性变换，则$\hat{x}$与$\hat{z}$均有单位协方差，且$I=Cov[\hat{z}]=JCov[\hat{x}]J^{T}=JJ^{T}$。因此，$JJ^{T}=I$，所以$J$的所有奇异值都等于1，这使得反向传播期间保留了梯度大小。在实际中，变换不是线性的，归一化的值不能保证是高斯的，也不能保证是独立的，但是我们仍然期望批归一化有助于使梯度传播更好地表现。批归一化对梯度传播的精确影响仍是一个有待进一步研究的领域。

### 3.4 批归一化使模型正则化

&emsp;当使用批归一化进行训练时，可以看到一个训练样本与小批量中的其他样本结合使用，训练网络不再为给定的训练样本生成确定值。在我们的实验中，我们发现这种效应有利于网络的泛化。虽然Dropout通常被用来减少过度拟合，在批归一化网络中，我们发现它可以被删除或大量减少。

## 4. 实验

### 4.1 Activations over time

&emsp;为了验证Internal Covariate Shift对训练的影响以及批归一化减轻（ICS）的能力，我们考虑了MNIST数据集上的预测数字类别的问题。我们使用一个非常简单的网络，一个$28\times 28$二进制图像作为输入，3个完全连接的隐含层，每个层有100个激活。每个隐含层用sigmoid非线性计算$y=g(Wu+b)$，权重$W$初始化为小的随机高斯值。最后一个隐含层后面是一个全连接层，有10个激活（每个类一个）和交叉熵损失。我们对网络进行50000步的训练，每个小批60个样本。我们向网络的每个隐含层添加了批归一化，如第3.1节所示。我们感兴趣的是baseline和批归一化网络之间的比较，而不是在MNIST上实现最先进的性能（所描述的体系结构没有）。

&emsp;图1（a）显示了随着训练的进行，两个网络对保留的测试数据进行正确预测的比例。批归一化网络具有较高的测试精度。为了调查原因，我们在训练过程中研究了原始网络$N$和批归一化网络$N_{BN}^{tr}$（算法2）中sigmoid的输入。在图1（b，c）中，我们展示了每个网络的最后一个隐含层的一个典型激活，它的分布是如何演变的。原始网络中分布的均值和方差随着时间的推移而发生显著变化，这使得后续层的训练变得复杂。相比之下，随着训练的进行，批归一化网络中的分布更加稳定，这有助于训练的进行。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-09-27-BN-Inception(InceptionV2)/figure1.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图1. （a）经过和没有经过批归一化训练的MNIST网络的测试准确度与训练步骤数的关系。批归一化有助于提高网络训练的速度和精度。（b，c）在训练过程中一个典型sigmoid的输入分布的演变，显示为第{15，50，85}个百分位数。批归一化使分布更加稳定，减少了Internal Covariate Shift。</div>
</center>

### 4.2 ImageNet分类

&emsp;我们应用批归一化到Inception网络的一个新的变体上，训练在ImageNet的分类任务上。网络中有大量的卷积和池化层，利用softmax层来预测图像类，有1000种可能性。卷积层使用ReLU作为非线性。与[Going Deeper with Convolutions]中描述的网络（注：也就是GoogLeNet或者说Inception V1）的主要区别是，$5\times 5$卷积层被两个连续的$3\times 3$卷积层替换，最多128个滤波器。该网络包含$13.6\cdot 10^{6}$个参数，并且除了顶部softmax层之外，没有全连接层。更多的详情见附录。在本文的其余部分中，我们将此模型称为Inception。该模型使用动量随机梯度下降的一个版本进行训练，使用的小批量大小为32。训练使用大规模分布式体系结构进行。通过计算验证精度@1来评估所有网络的训练进展情况，即在一组保留的图像上，使用每张图像的single-crop，从1000种可能性中预测正确标签的概率。

&emsp;在我们的实验中，我们评估了使用批归一化的Inception的几种修改。在所有情况下，以卷积的方式对每个非线性的输入采用批归一化，如第3.2节所述，同时保持其余的体系结构不变。

#### 4.2.1 加速BN网络

&emsp;仅仅向网络添加批归一化并不能充分利用我们的方法。为此，我们进一步改变了网络及其训练参数，如下所示：

&emsp;*提高学习率。*&emsp;在批归一化模型中，我们已经能够从更高的学习率中获得训练加速，而没有不良的副作用（第3.3节）。

&emsp;*删除Dropout。*&emsp;如第3.4节所述，批处理规范化实现了与Dropout相同的一些目标。从修改后的BN-Inception中删除Dropout可以加快训练速度，而不会增加过拟合。

&emsp;*减少$L_{2}$权重正则化。*&emsp;在Inception中，模型参数的$L_{2}$损失控制着过拟合，而在修改的BN-Inception中，这种损失的权重减少了5倍。我们发现这提高了在保留的验证数据上的准确性。

&emsp;*加速学习率衰减。*&emsp;在训练Inception时，学习率呈指数衰减。因为我们的网络比Inception训练得更快，所以我们将学习率降低了6倍。

&emsp;*删除局部响应归一化。*&emsp;虽然Inception和其他网络从中受益，但我们发现使用批归一化后它就没有必要了。

&emsp;*更彻底地shuffle训练样本。*&emsp;我们启用了训练数据的分片内改组，从而防止了相同的样本始终出现在一个小批量中。这导致验证准确性提高了约1％，这与批归一化作为正则化器的观点一致（第3.4节）：当每次看到样本都会对样本产生不同的影响时，我们方法固有的随机性应该是最有益的。

&emsp;*减少光度失真。*&emsp;由于批归一化的网络训练速度更快，而且每个训练样本的观察次数更少，所以我们让训练器通过减少对“真实”图像的失真来关注更多的“真实”图像。

#### 4.2.2 单网络分类

&emsp;我们评估了以下网络，均使用LSVRC2012训练数据进行训练，并对验证数据进行测试:

&emsp;*Inception：*&emsp;4.2节开始描述的网络，初始学习率为0.0015。

&emsp;*BN-Baseline：*&emsp;与Inception相同，在每个非线性之前进行批归一化。

&emsp;*BN-x5：*&emsp;批归一化的Inception和第4.2.1节中的修改。初始学习率提高了5倍，达到0.0075。与原始Inception相同的学习率增加导致模型参数达到机器无穷大。

&emsp;*BN-x30：*&emsp;类似于BN-x5，但初始学习率为0.045（初始学习率的30倍）。

&emsp;*BN-x5-Sigmoid：*&emsp;类似于BN-x5，但是用sigmoid非线性$g(t)=\frac{1}{1+exp\left(-x\right)}$而不是ReLU。我们还尝试用sigmoid训练原始的Inception，但该模型仍保持着与偶然性相同的精确性。

&emsp;在图2中，我们展示了网络的验证精度，作为训练步骤数量的函数。经过$31\cdot 10^{6}$个训练步骤，Inception准确率达到72.2%。图3显示了对于每个网络，达到相同的72.2%准确率所需要的训练步骤数，以及网络所达到的最大验证精度和达到该精度所需的步骤数。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-09-27-BN-Inception(InceptionV2)/figure2.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图2. Inception及其批归一化变体的single-crop验证精度与训练步骤数的关系。</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-09-27-BN-Inception(InceptionV2)/figure3.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图3. 对于Inception和批归一化的变体，达到Inception的最大精度（72.2%）和网络所能达到的最大精度所需的训练步骤数。</div>
</center>

&emsp;通过只使用批归一化（BN-Inception），我们在不到一半的训练步骤中匹配Inception的准确性。通过应用第4.2.1节中的修改，我们显著提高了网络的训练速度。BN-x5需要比Inception少14倍的步骤就能达到72.2%的准确率。有趣的是，进一步提高学习率（BN-x30）会使模型最初的训练速度稍微慢一些，但允许它达到更高的最终精度。$6\cdot 10^{6}$步后达到74.8%，即比Inception达到72.2％所需的步数少5倍。

&emsp;我们还验证了Internal Covariate Shift的减少允许将Sigmoid用作非线性时训练具有批归一化的深层网络，尽管训练此类网络存在众所周知的困难。确实，BN-x5-Sigmoid达到了69.8％的准确度。如果没有批归一化，使用sigmoid的Inception的精确度永远不会超过1/1000。

#### 4.2.3 集成分类

&emsp;目前报道的ImageNet大型视觉识别大赛的最佳结果是通过传统模型的Deep Image集成和[Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification]的集成模型得到的。根据ILSVRC服务器的评估，后者报告的top-5错误为4.94%。在这里，我们报告的top-5验证错误率为4.9%，测试错误率为4.82%（根据ILSVRC服务器）。这改进了之前的最佳结果，并超过了人类评分者的估计精度。

&emsp;在我们的集成中，我们使用了6个网络。每一个都是基于BN-x30，作了以下修改：增加卷积层的初始权重；使用Dropout（dropout的概率为5%或10%vs原始Inception的为40%）；以及对模型的最后隐含层使用非卷积的，每个激活的批归一化。每个网络经过$6\cdot 10^{6}$步左右的训练，达到了最大精度。集成预测是基于各组成网络预测的类概率的算术平均。集成和多crop推理的细节类似于\[GoogLeNet\]。

&emsp;我们在图4中证明了批归一化允许我们在ImageNet分类挑战基准上以合理的间隔设置新的最先进技术。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-09-27-BN-Inception(InceptionV2)/figure4.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图4. 在提供的包含50000张图像的验证集上，Batch-Normalized Inception与以前的SOTA进行比较。*根据测试服务器的报告，BN-Inception集成在ImageNet测试集的100000张图像上达到了4.82%的top-5错误率</div>
</center>

## 5. 结论

&emsp;我们提出了一种新的机制，可以显著加快深度网络的训练。基于众所周知的Covariate Shift使机器学习系统的训练复杂化的前提，Covariate Shift也适用于子网和层，从网络的内部激活中将其删除可能有助于训练。我们提出的方法从归一化激活以及将归一化合并到网络体系结构本身中汲取了力量。这可以确保任何用于训练网络的优化方法都能恰当地处理归一化。为了实现深度网络训练中常用的随机优化方法，我们对每个小批量进行归一化，并通过归一化参数对梯度进行反向传播。批归一化对每个激活只添加两个额外的参数，这样做保留了网络的表征能力。提出了一种利用批归一化网络构造、训练和执行推理的算法。得到的网络可以用饱和非线性进行训练，对增加的训练率更有容忍度，而且通常不需要dropout用于正则化。

&emsp;仅仅在最先进的图像分类模型中添加批归一化，就可以大大加快训练速度。通过进一步提高学习率、去除Dropout和应用批归一化所提供的其他修改，我们只需一小部分训练步骤就达到了先前的SOTA，然后在单网络图像分类中击败了SOTA。此外，通过结合使用批归一化训练的多个模型，我们在ImageNet上的表现要比ImageNet上最知名的系统要好。

&emsp;有趣的是，我们的方法与[Knowledge matters: Importance of prior information for optimization]的标准化层有相似之处，尽管这两种方法的目标非常不同，执行的任务也不同。批归一化的目标是在整个训练过程中实现激活值的稳定分布，在我们的实验中，我们将其应用于非线性之前，因为在非线性之前，匹配一阶和二阶矩更有可能得到稳定的分布。相反，另一个方法将标准化层应用于非线性的输出，导致更稀疏的激活。在我们的大规模图像分类实验中，无论有没有批归一化，我们都没有观察到非线性输入是稀疏的。批归一化的另一个显著区别特征包括学习尺度和平移，从而允许BN变换来表示恒等变换（标准化层不需要这个，因为它后面跟着的是学习的线性变换，从概念上讲，它吸收了必要的尺度和平移），卷积层的处理，不依赖于小批量的确定性推理和在网络中批归一化每个卷积层。

&emsp;在这项工作中，我们还没有探索批归一化可能实现的所有可能性。我们未来的工作包括将我们的方法应用于递归神经网络，其中Internal Covariate Shift和梯度消失或爆炸可能特别严重，这将使我们能够更彻底地检验归一化改善梯度传播的假设（第3.3节）。我们计划研究批归一化是否可以帮助领域适应，从传统意义上说，即网络执行的归一化是否可以使其更轻松地推广到新的数据分布，也许只需对总体均值和方差重新计算即可（算法2）。最后,我们相信的进一步理论分析算法将允许更多的改进和应用。

## 附录

### 使用Inception启模型的变体

&emsp;图5记录了与GoogleNet架构相比所做的更改。对于本表的解释，请查阅\[GoogleNet\]。与GoogLeNet模型相比，显著的架构变化包括:

* 将$5\times 5$卷积层替换为两个连续的$3\times 3$卷积层。这将使网络的最大深度增加9个权重层。同时，该方法将参数量提高了25%，计算量提高了30%左右。

* $28\times 28$的Inception模块数量从2增加到3。

* 在模块内部，有时使用平均池化，有时使用最大池化。这在与表中的池化层相对应的条目中指示。

* 在任何两个Inception模块之间没有跨板池化层，但是在模块3c、4e中的滤波器级联之前采用了stride-2卷积/池化层。

&emsp;我们的模型在第一个卷积层上使用了深度乘数为8的可分离卷积。这减少了计算成本，同时增加了训练时的内存消耗。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-09-27-BN-Inception(InceptionV2)/figure5.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图5. Inception架构</div>
</center>

---

## 个人看法

&emsp;这篇文章提出了BN也就是批归一化操作，基本上在现在的CNN中是一个必备操作了。然后对Inception所作的一些修改中，比较有启示作用的还是用了两个$3\times 3$卷积代替了原本的$5\times 5$卷积。
