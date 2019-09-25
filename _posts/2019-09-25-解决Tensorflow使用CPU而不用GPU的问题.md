---
layout:     post
title:      解决Tensorflow使用CPU而不用GPU的问题
subtitle:   Solve the problem that Tensorflow uses CPU instead of GPU
date:       2019-09-25
author:     CJR
header-img: img/2019-09-25-解决Tensorflow使用CPU而不用GPU的问题/post-bg.jpg
catalog: true
mathjax: false
tags:
    - Tensorflow
    - CPU
    - GPU
---

&emsp;之前的文章讲过用Tensorflow的object detection api训练MobileNetV2-SSDLite，然后发现训练的时候没有利用到GPU，反而CPU占用率贼高（可能会有`Could not dlopen library 'libcudart.so.10.0'`之类的警告）。经调查应该是Tensorflow的GPU版本跟服务器所用的cuda及cudnn版本不匹配引起的。知道问题所在之后就好办了。

---

## 检查cuda和cudnn版本

&emsp;首先查看cuda版本：

```
cat /usr/local/cuda/version.txt
```

&emsp;以及cudnn版本：

```
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
```

## 重新安装对应版本Tensorflow

&emsp;根据前面查看得到的cuda和cudnn版本，到[Tensorflow官网](https://tensorflow.google.cn/install/source#tested_build_configurations)查看对应的Tensorflow-GPU版本，然后用`conda install tensorflow-gpu=[version]`重新安装
（把[version]换成对应的版本比如1.12）就OK了。
