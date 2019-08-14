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

## Tensorflow的object detection api

&emsp;之前的文章有用caffe去训练一个物体检测网络MobileNetV2-SSDLite，然后这里是用Tensorflow的object detection api训练MobileNetV2-SSDLite。

---

## 数据集准备

&emsp;数据集的准备可以直接参考[目标检测Tensorflow object detection API之构建自己的模型](https://zhuanlan.zhihu.com/p/35854575)或者自行搜索相应教程，最终应该有两个`tfrecord`文件（分别用于训练和测试）。

## 开发环境

&emsp;这里我的环境是python 3.7+Tensorflow 1.13.1。

## object detection api 下载与安装

&emsp;使用`git clone https://github.com/tensorflow/models.git`下载所需的代码及模型文件。Windows下的话，先在<https://github.com/google/protobuf/releases>下载相应版本的protobuf，解压后将`bin`文件夹中的`protoc.exe`复制到`models`文件夹的`research`目录下，在`research`目录执行下面的代码，将`object_detection/protos`下的`.proto`文件转换成`.py`文件：

```
import os

for each in os.listdir('object_detection/protos'):
    if each.endswith('proto'):
        os.system('protoc object_detection/protos/%s --python_out=.' % each)
```

&emsp;Linux下的话可以参考[linux下安装tensorflow object detection API以及 安装过程问题解决](https://blog.csdn.net/pwtd_huran/article/details/80874791)。

&emsp;接着在`research`和`research/slim`目录下分别执行`python setup.py install`以执行安装，然后就可以把`research`和`research/slim`目录添加到环境变量中去了。

&emsp;为了验证安装成功与否，可以执行`python object_detection/builders/model_builder_test.py`，成功的话会输出`OK`。

## 训练自己的数据

&emsp;为了整理方便，创建一个文件夹目录如下：

```
├── MobileNetV2-SSDLite
  ├── data
  ├── models
  ├── training
```

&emsp;其中`data`用于存放训练及测试数据，`models`就是之前下载的代码及模型，`training`则存放模型文件及之后的训练过程产生的数据。

&emsp;将之前生成的两个`tfrecord`文件移到`data`文件夹下之后，再新建一个`my_label_map.pbtxt`文件，根据实际情况写入（这里因为我的数据只有一个类别所以就只有一个item）：

```
item {
    name : "tfeature"
    id : 1
}
```

&emsp;接着在`object_detection`文件夹中的`samples/config`路径下，找到配置文件`ssdlite_mobilenet_v2_coco.config`，复制到`training`文件夹中，并做如下修改：

* num_classes修改为1（当然根据你自己的数据作修改）
* batch_size根据电脑/服务器配置，可以适当调高或者调低
* 两个input_path分别设置成`data/train.tfrecord`和`data/test.tfrecord`
* 两个label_map_path均设置成`data/my_label_map.pbtxt`
* 第158、159行的

```
    fine_tune_checkpoint: "ssd_mobilenet_v1_coco_11_06_2017/model.ckpt"   
    from_detection_checkpoint: true
```

&emsp;&emsp;这2行需要删除或注释掉

&emsp;到这就可以开始训练了，但为了输出loss信息，先在`model_main.py`文件的`import`区域之后添加`tf.logging.set_verbosity(tf.logging.INFO)`，接着在`MobileNetV2-SSDLite`目录下执行以下命令：

```
python model_main.py \
    --pipeline_config_path=training/ssdlite_mobilenet_v2_coco.config \
    --model_dir=training \
    --num_train_steps=400000 \
    --num_eval_steps=20 \
    --alsologtostderr
```

&emsp;接着就开始训练了，正常的话可以应该有如下的输出：

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-07-22-用Tensorflow的object detection api训练自己的数据/train.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">训练过程</div>
</center>

&emsp;此外也可以用`tensorboard --logdir=training`可视化训练过程：

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-07-22-用Tensorflow的object detection api训练自己的数据/mAP.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">可视化训练过程</div>
</center>
