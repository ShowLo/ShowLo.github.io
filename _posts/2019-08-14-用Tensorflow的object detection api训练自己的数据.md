---
layout:     post
title:      用Tensorflow的object detection api训练自己的数据
subtitle:   Training your own data with Tensorflow's object detection API
date:       2019-08-14
author:     CJR
header-img: img/2019-07-22-用Tensorflow的object detection api训练自己的数据/post-bg.jpg
catalog: true
mathjax: false
tags:
    - Tensorflow
    - Object Detection
    - Lightweight Network
    - MobileNetV2-SSDLite
---

# Tensorflow的object detection api
&emsp;之前的文章有用caffe去训练一个物体检测网络MobileNetV2-SSDLite，然后这里是用Tensorflow的object detection api训练MobileNetV2-SSDLite。

---

## 数据集准备