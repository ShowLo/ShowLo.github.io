---
layout:     post
title:      实现Xshell退出（断开SSH连接）而远程服务器上跑的程序不退出
subtitle:   Xshell exit (disconnect SSH connection) and the program running on the remote server does not exit
date:       2019-10-17
author:     CJR
header-img: img/2019-10-17-实现Xshell退出（断开SSH连接）而远程服务器上跑的程序不退出/post-bg.jpg
catalog: true
mathjax: false
tags:
    - Xshell
    - SSH
    - nohup
    - tail
---

&emsp;最近用Xshell连接服务器跑代码的时候，好几次因为停电/断网导致服务器上跑的程序因SSH连接退出而中断，很烦，每次都得重连服务器后重新跑。所以下面讲一下怎么利用`nohup`命令和`tail`命令解决这个问题。

---

## 不挂断地运行

&emsp;用`nohup`运行命令可以永久执行此命令直至其运行完毕，和用户终端没有关系，也就是断开SSH连接也不会影响其运行，所以就可以用比如`nohup python xxx.py`来保证代码在执行过程中不会因为用户终端的退出等原因而中断。

## 实时查看输出

&emsp;但我们执行代码有时需要实时查看输出，而用了`nohup`之后程序的输出被重定向到了`nohup.out`文件中，这时我们可以使用`tail -f nohup.out`来实时查看输出。
