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

### 3.4 批归一化使模型正规化

&emsp;当使用批归一化进行训练时，可以看到一个训练样本与小批量中的其他样本结合使用，训练网络不再为给定的训练样本生成确定值。在我们的实验中，我们发现这种效应有利于网络的泛化。虽然Dropout通常被用来减少过度拟合，在批归一化网络中，我们发现它可以被删除或大量减少。

## 4. 实验

&emsp;

&emsp;

&emsp;

&emsp;

&emsp;

&emsp;

&emsp;

&emsp;

&emsp;

&emsp;

&emsp;

&emsp;

&emsp;

## 5. 结论

&emsp;

&emsp;

&emsp;

---

## 个人看法

&emsp;
