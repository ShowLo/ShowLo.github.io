---
layout:     post
title:      Linux&Windows下用C++跑pytorch模型
subtitle:   Running pytorch model with C++ under Linux & Windows
date:       2019-07-15
author:     CJR
header-img: img/2019-07-15-Linux&Windows下用C++跑pytorch模型/post-bg.jpg
catalog: true
mathjax: false
tags:
    - pytorch
    - Linux
    - Windows
    - C++
---

&emsp;前一篇博客讲了怎么用pytorch实现一个简单的图像分类器，这一篇的主要目的则是将前面训练好的模型用C++跑起来。

---

## 模型转换

&emsp;第一步需要python环境下训练得到的pytorch模型转换为C++可读的模型，更具体的可以参考[官方教程](https://pytorch.org/tutorials/advanced/cpp_export.html)，这里用了最简单的通过Tracing的方法来进行模型转换：
```
import torch
from SimpleNet import SimpleNet

if __name__ == "__main__":
    
    # An instance of your model.
    best_model_path = 'epoch_462.pth'
    model = SimpleNet()
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # An example input you would normally provide to the forward() method
    example = torch.rand(1, 1, 90, 90)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)

    # Save model
    traced_script_module.save("model.pt")
```

&emsp;跑完以上代码应该会得到一个`model.pt`文件，这个文件就是接下来将用于C++下的模型文件。

## C++程序代码

&emsp;这里将整个预测过程写成了一个函数，输入为一个代表了一张灰度图片的二维数组，而在`main`函数里对此进行测试。代码如下：

```
#include <torch/script.h>

#include <iostream>
#include <memory>

// The function to predict the class of an image
int predict(void* img)
{
	std::string model_path = "model.pt";
	// load the model
	std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(model_path);
	// Change the Image into Tensor for prediction
	torch::Tensor tensor_image = torch::from_blob(img, { 1, 1, 90, 90 }, torch::kInt);
	tensor_image = tensor_image.toType(torch::kFloat);
	// distribution between 0 and 1
	tensor_image = tensor_image.div(255);
	// normalize, value between -1 and 1
	tensor_image = tensor_image.sub(0.5);
	tensor_image = tensor_image.div(0.5);
	// predict
	torch::Tensor result = module->forward({ tensor_image }).toTensor();
	// get the class index
	auto max_result = result.max(1, true);
    auto max_index = std::get<1>(max_result).item<float>();

	return int(max_index);
}

int main()
{
	// A test image whose class is bubble
	//int img[90][90] = ...a 2-d array here...;
	int pred = predict(img);
	std::cout << "The predicted result is : " << pred << std::endl;
	while (1);
	return 0;
}
```

&emsp;与python代码对应的，这里同样需要对输入图像做包括归一化操作在内的预处理，然后才能调用模型进行预测。上面的代码保存为一个`example-app.cpp`文件，保存在新建的一个`SimpleNet`文件夹下。

## 编译之前的准备

&emsp;接着在`SimpleNet`文件夹下新建一个`CMakeLists.txt`文件并写入以下内容：

```
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(SimpleNet)

find_package(Torch REQUIRED)

add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 11)
```


## Linux下用C++调用pytorch模型

&emsp;首先从[官网](https://pytorch.org/get-started/locally/)下载Linux版本的libtorch并解压：

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-07-15-Linux&Windows下用C++跑pytorch模型/libtorch-linux.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Linux版本的libtorch</div>
</center>

&emsp;然后进入`SimpleNet`文件夹，依次执行以下命令：

```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=~/cmake/libtorch ..	(这里libtorch路径需要根据实际解压路径作修改)
make
```

&emsp;然后把`model.pt`文件复制到`build`文件夹里面去，便可以用以下命令执行了：

```
./example-app
```

## Windows下用C++调用pytorch模型

&emsp;首先从[官网](https://pytorch.org/get-started/locally/)下载Windows版本的libtorch并解压：

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/ShowLo/ShowLo.github.io/master/img/2019-07-15-Linux&Windows下用C++跑pytorch模型/libtorch-windows.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Windows版本的libtorch</div>
</center>


&emsp;参考[这篇博客](https://oldpan.me/archives/pytorch-windows-libtorch)，在`SimpleNet`文件夹中新建`build`文件夹，进入`build`文件夹，在命令行下输入：
```
cmake -DCMAKE_PREFIX_PATH=path\libtorch -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 15 Win64" ..
```

&emsp;其中`DCMAKE_PREFIX_PATH`修改为解压后的libtorch路径，`Visual Studio 15`代表的是VS2017所用的vc版本为15，如果用的VS2015则修改15为14，注意最后有`..`这两个点。

&emsp;接着将`model.pt`复制到`build`文件夹中，进入`build`文件夹，用VS2017打开`example-app.vcxproj`文件，把`example-app`设置为启动项目，执行`example-app.cpp`，这时会报找不到`xxx.dll`的错，把`libtorch/lib`下的`c10.dll`、`caffe2.dll`、`libiomp5md.dll`和`torch.dll`复制到`Release/Debug`文件夹中去再重新执行即可（可能需要在`debug`模式下执行一次，再在`release`模式下执行一次）。