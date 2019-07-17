---
layout:     post
title:      用自己的数据训练MobileNetV2-SSDLite
subtitle:   Training MobileNetV2-SSDLite using your own dataset
date:       2019-07-16
author:     CJR
header-img: img/2019-07-16-用自己的数据训练MobileNetV2-SSDLite/post-bg.jpg
catalog: true
mathjax: false
tags:
    - MobileNetV2-SSDLite
    - SSDLite
    - MobileNetV2
    - object detection
---

&emsp;前面的文章介绍了如何安装caffe并切换到ssd分支，如何添加对ReLU6的支持，以及如何安装和使用MobileNetV2-SSDLite。这篇文章开始介绍如何利用自己的数据集训练MobileNetV2-SSDLite。这篇文章主要参考了[caffe-SSD配置及用caffe-MobileNet-SSD训练自己的数据集](https://blog.csdn.net/Chris_zhangrx/article/details/80458515)，并作了一些修改。

---

## 数据集准备

&emsp;首先我们需要有图片以及与之对应的xml文件，并存放成VOC2007文件夹格式，可以采用labelImg工具进行标注，标注完成之后还需要切分训练集、验证集与测试集。这部分可以参考[此篇博客](https://blog.csdn.net/u011574296/article/details/78953681)。最终可以得到如以下所示的文件夹：

```
├── VOC2007
  ├── Annotations
      ├── xxxxxx.xml
      ├──        ...
  ├── ImageSets
      ├── Main
          ├── test.txt
          ├── train.txt
          ├── trainval.txt
          ├── val.txt
  ├── JPEGImages
      ├── xxxxxx.jpg
      ├──        ...
```

&emsp;接着需要将数据转换为训练用的lmdb文件，我们需要在之前安装了caffe的目录下找到`data/VOC0712`文件夹，把其下的`create_list.sh`和`create_data.sh`给复制到一个临时文件夹中去，并修改其下的`label_voc.prototxt`（根据数据集中的类别修改，注意保留第0类的背景类，也就是说如果你的数据集有一个类别，那么修改后的文件就会有两个类别），然后将修改后的`label_voc.prototxt`文件保存到之前的`VOC2007`文件夹下面。

&emsp;接着修改`create_list.sh`，将`root_dir`改为存放`VOC2007`的文件夹路径，并将第41行的`$bash_dir/../../build/tools/get_image_size`改为自己caffe路径下的`build/tools/get_image_size`（这是因为前面将`create_list.sh`这个文件从caffe目录下给复制了出来再作的修改，如果是在caffe目录下直接执行的话就不用改这一行了）。然后因为我是在Windows下标注的数据，而Windows和Linux下换行符的不同会使得执行时报类似`io.cpp:187] Could not open or find file .../VOC2007/Annotations/000611`的错，所以还需要修改第25行的`"s/$/.jpg/g"`为`"s/\r$/.jpg/g"`、第30行的`"s/$/.xml/g"`为`"s/\r$/.xml/g"`。输入`bash create_list.sh`执行可以得到以下输出：

```
Create list for VOC2007 trainval...
Create list for VOC2007 test...
I0716 15:05:41.717645 30716 get_image_size.cpp:61] A total of 1571 images.
I0716 15:06:58.354274 30716 get_image_size.cpp:100] Processed 1000 files.
I0716 15:07:34.616385 30716 get_image_size.cpp:105] Processed 1571 files.
Create list for VOC2007 train...
Create list for VOC2007 val...
```

&emsp;并且可以在临时文件夹下看到`test_name_size.txt`、`test.txt`、`train.txt`、`trainval.txt`、`val.txt`这几个文件。

&emsp;下面是我修改之后的`create_list.sh`以供参考：

```
#!/bin/bash

root_dir=/media/data2/chenjiarong/Parking   #修改1
sub_dir=ImageSets/Main
bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for dataset in trainval test train val      #修改2，也可不改
do
  dst_file=$bash_dir/$dataset.txt
  if [ -f $dst_file ]
  then
    rm -f $dst_file
  fi
  for name in VOC2007
  do
    if [[ $dataset == "test" && $name == "VOC2012" ]]
    then
      continue
    fi
    echo "Create list for $name $dataset..."
    dataset_file=$root_dir/$name/$sub_dir/$dataset.txt

    img_file=$bash_dir/$dataset"_img.txt"
    cp $dataset_file $img_file
    sed -i "s/^/$name\/JPEGImages\//g" $img_file
    sed -i "s/\r$/.jpg/g" $img_file     #修改3

    label_file=$bash_dir/$dataset"_label.txt"
    cp $dataset_file $label_file
    sed -i "s/^/$name\/Annotations\//g" $label_file
    sed -i "s/\r$/.xml/g" $label_file   #修改4

    paste -d' ' $img_file $label_file >> $dst_file

    rm -f $label_file
    rm -f $img_file
  done

  # Generate image name and size infomation.
  if [ $dataset == "test" ]
  then
    $bash_dir/../anaconda3/envs/ssdlite/caffe/build/tools/get_image_size $root_dir $dst_file $bash_dir/$dataset"_name_size.txt" #修改5
  fi

  # Shuffle trainval file.
  if [ $dataset == "trainval" ]
  then
    rand_file=$dst_file.random
    cat $dst_file | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > $rand_file
    mv $rand_file $dst_file
  fi
done
```

&emsp;接着修改`create_data.sh`，具体见下面：

```
cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
root_dir=~/anaconda3/envs/ssdlite/caffe     #修改为自己的caffe目录

cd $root_dir

redo=1
data_root_dir=".../Parking"    #修改为存放VOC2007文件夹的目录
dataset_name="VOC0712"
mapfile=".../VOC2007/labelmap_voc.prototxt" #修改为之前放置labelmap_voc文件的绝对路径
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in test trainval
do
  python $root_dir/scripts/create_annoset.py \
  	--anno-type=$anno_type \
  	--label-map-file=$mapfile \
  	--min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height \
  	--check-label $extra_cmd $data_root_dir \
  	~/temp/$subset.txt \    #前一步生成的test.txt、trainval.txt的路径
  	$data_root_dir/$dataset_name/$db/$dataset_name"_"$subset"_"$db \    #生成lmdb文件的路径及名称
  	$root_dir/examples/$dataset_name    #修改为caffe路径
done
```

&emsp;接着需要激活caffe环境--`conda activate ssdlite`，再执行`create_data.sh`。成功后就可以在我们设置的路径下面看到一个`lmdb`文件夹了，其内有`VOC0712_trainval_lmdb`和`VOC0712_test_lmdb`两个之后要用到的文件。

## 配置训练文件

&emsp;这里参考了[MobileNetv2-SSDlite训练自己的数据集（二）——训练MSSD](https://blog.csdn.net/qq_43150911/article/details/85107261)与[MobileNetv2-SSDLite训练自己的数据集](https://blog.csdn.net/u010986080/article/details/84983310))。首先，将前面修改之后的`labelmap_voc.txt`文件复制到`MobileNetv2-SSDLite/ssdlite/voc`文件夹下。然后，通过`ssdlite`目录下的`gen_model.py`生成`train.prototxt`、`test.prototxt`、`deploy.prototxt`（需要修改`gen_model.py`中caffe路径）：

```
cd ~/MobileNetv2-SSDLite/ssdlite
python gen_model.py -s train -c CLASS_NUM --tfpad --relu6 >train.prototxt
python gen_model.py -s test -c CLASS_NUM --tfpad --relu6 >test.prototxt
python gen_model.py -s deploy -c CLASS_NUM --tfpad --relu6 >deploy.prototxt
mv train.prototxt voc
mv test.prototxt voc
mv deploy.prototxt voc
cd voc
```

&emsp;其中的`CLASS_NUM`需要根据实际数据进行修改，其值为数据中类别数+1（加上背景类）；`tfpad`是为了消除识别时可能出现`bounding box`的偏差；`relu6`是为了将`ReLU`替换为`ReLU6`。

&emsp;然后需要修改`train.prototxt`中的参数，主要是修改`data_param`中`source`的值为之前生成的`VOC0712_trainval_lmdb`的路径，以及修改`annotated_data_param`中`label_map_file`的值为修改后的`labelmap_voc.prototxt`的地址。对于`test.prototxt`的修改是类似的。此外，如果GPU显存太少，还需要将`train.prototxt`的`data_param`中`batch_size`的值改小。

&emsp;对于`solver_train.prototxt`和`solver_test.prototxt`的修改则根据需要自行修改，比如几次迭代保存一次模型需要修改`snapshot`，模型保存地址修改`snapshot_prefix`等。

&emsp;最后还需要修改`train.sh`和`test.sh`，主要修改`snapshot`为前面设置的模型保存地址，修改`../../build/tools/caffe`为自己的caffe地址。如果要使用预训练模型，将`-weights`指定为预训练模型的位置，否则注释掉。

&emsp;到这里就可以进行训练了：

```
conda activate ssdlite
CUDA_VISIBLE_DEVICES=1 sh train.sh
```

&emsp;但是我自己训练的时候又报了`status == CUDNN_STATUS_SUCCESS (4 vs. 0) CUDNN_STATUS_INTERNAL_ERROR`的错，具体原因可以看[这篇博客](https://blog.csdn.net/goodxin_ie/article/details/89057095)，这里的做法就是将用到`depthwise`（深度卷积）的地方都用CPU跑（加上一句`engine: CAFFE`），所以需要对三个`xxx.prototxt`文件都做多处的类似修改（在编辑器下搜索`group`可以快速定位到需要修改的地方）：

```
layer {
  name: "conv/depthwise"
  type: "Convolution"
  bottom: "Conv"
  top: "conv/depthwise"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    num_output: 32
    bias_term: false
    pad: 1
    kernel_size: 3
    group: 32
    engine: CAFFE
    weight_filler {
      type: "msra"
    }
  }
}
```

&emsp;修改之后再重新训练就没问题了，下图所示是训练的过程：

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-07-16-用自己的数据训练MobileNetV2-SSDLite/train.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">训练过程</div>
</center>