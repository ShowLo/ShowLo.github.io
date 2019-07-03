---
layout:     post
title:      MobileNetV2-SSDLite的安装使用
subtitle:   For SSDLite
date:       2019-06-25
author:     CJR
header-img: img/2019-07-03-MobileNetV2-SSDLite的安装使用/post-bg-sci.jpg
catalog: true
mathjax: false
tags:
    - MobileNetV2
    - SSD
    - SSDLite
    - object detection
---

&emsp;前两篇文章已经安装了caffe并切换到ssd分支，同时添加了对ReLU6的支持，接着这里开始安装和使用MobileNetV2-SSDLite。

---

&emsp;首先安装MobileNetV2-SSDLite：

```
git clone https://github.com/chuanqi305/MobileNetv2-SSDLite
```

&emsp;进入到`ssdlite`文件夹之后下载tensorflow模型并解压：

```
wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
tar -zvxf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
```

&emsp;然后将解压的tensorflow模型转为caffe模型，执行下面两个脚本后，会在当前目录下生成deploy.caffemodel这个caffe模型：

```
python dump_tensorflow_weights.py
# 注意将load_caffe_weight.py中caffe_root改为自己的caffe路径
python load_caffe_weight.py
```

&emsp;我自己在执行上面第二个脚本的时候报了`Check failed: status == CUDNN_STATUS_SUCCESS (4 vs. 0)  CUDNN_STATUS_INTERNAL_ERROR`的错误，这是因为GPU的显存不够用，所以需要将`ssdlite`文件夹下的`deploy.prototxt`中所有的`#engine: CAFFE`前面的`#`号去掉。

&emsp;这个时候如果直接跑`demo_caffe.py`同样会因为coco模型需要的显存太大，导致GPU显存不够用，所以还需要将coco模型转换为voc模型，运行：

```
# 同样需要修改caffe_root
python coco2voc.py
```

&emsp;这里还是报了`Check failed: status == CUDNN_STATUS_SUCCESS (4 vs. 0)  CUDNN_STATUS_INTERNAL_ERROR`的错，看了源代码后发现还需要对`ssdlite/voc`文件夹下的`deploy.prototxt`做同样的删除`#engine: CAFFE`前面的`#`号的操作。

&emsp;重新运行`coco2voc.py`后，就可以运行：

```
python demo_caffe_voc.py
```

&emsp;下面是测试的一些结果：

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-07-03-MobileNetV2-SSDLite的安装使用/000001test.jpg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">测试结果1</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-07-03-MobileNetV2-SSDLite的安装使用/000067test.jpg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">测试结果2</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-07-03-MobileNetV2-SSDLite的安装使用/000456test.jpg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">测试结果3</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-07-03-MobileNetV2-SSDLite的安装使用/000542test.jpg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">测试结果4</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-07-03-MobileNetV2-SSDLite的安装使用/001150test.jpg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">测试结果5</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-07-03-MobileNetV2-SSDLite的安装使用/001763test.jpg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">测试结果6</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-07-03-MobileNetV2-SSDLite的安装使用/004545test.jpg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">测试结果7</div>
</center>