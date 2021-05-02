# cornell_dataset

## 1.简介(Introduction)

追随着大佬的脚步，从头复现一下大佬的论文程序，大佬的代码项目地址为：https://github.com/dougsm/ggcnn

Re-implementation of excellent ggcnn code,the original repositiry address is :https://github.com/dougsm/ggcnn

## 2.文件说明(Instruction for files)

- cornell:存放原始数据集解压后数据(注意，项目文件夹中并不包含这个文件，太大了传不上来，只需要自己在官网下载解压，然后使用gg-cnn里面的程序根据点云生成tiff图片就行了,程序在大佬的项目中，路径为：utils->dataset_processing->generate_cornell_depth.py)
- 1.load_data 数据的载入部分
- 2.net_data_stream 整体训练数据流的构建
- 3.data_augmentation（not exactly） 数据增强部分（误）
- 4.train 训练部分
- 5.validate 验证部分
- 6.optimize 优化部分（实际上这里才是数据增强）
- 7.ggcnn2 就是用了下ggcnn2模型
- 8.jacquard_code 使用jacquard数据集进行训练
- 9.acceleration 尝试使用DALI库进行加速,还没完成,估计后面也不会完成了,觉得成不了
- images:存放一些笔记中用到的图片
- support_files:存放一些积累答疑文件
- problems:遇到的一些问题



> **后缀为ipynb的文件为相应程序的原始建立过程，后缀为py的文件为与ipynb对应的建立好的完整程序**



## 3.与ggcnn中定义的不同之处(Some differences with original GGCNN)
- 1.width和length命名是反过来的
- ==~~2.角点的x,y坐标好像也是反过来的~~==
- ==~~3.在我的定义里面，抓取框对象的center属性，center[0]是x坐标，减去二分之一output size后对应left坐标，center[1]是y坐标，所以crop的时候，我的是（left,top），ggcnn是（top,left）~~==

## 4.一些注意事项(Notes)

- 1.刚发现如果文件在jupyter中处于打开状态的话，git这边无法push，记录一下以后注意。
- 2.截至到data_augmentation的程序在hp上测试无误，不晓得为什么在ubuntu上就不行，报了一个错，明天好好检查一下，已经解决，详见support_files/functions 第5点
- 3.调试过程中遇到的重大bug1：调节output_size后网络的输出与标注shape不匹配导致的一系列问题，问题原因：输入维度发生改变后，网络结构不变，输出的shape本就会发生变化，想要控制具体的输出尺寸就必须对网络参数进行仔细设计调整
- 4.调试过程中遇到的重大bug2：Gaussian滤波后的十字问题，问题原因：所使用的Gaussian滤波函数不支持一次输入多个样本，val过程中batch_size的设定应当为1
- 5.调试过程中遇到的重大bug3：这个应该不能说是重大bug，因为，这是一个失误，之前其实已经发现了，就是，可视化绘制抓取框的时候，不要用rectangles函数，这函数只能根据你给的两个角点画出标准的，竖直的矩形，画这种斜的矩形要用line函数

## 5.特别说明(Attention)
由于我之前复现写的时候,有一些自己的想法,所以对原始的代码做了一些修改,也就是上面的第三部分,与`ggcnn`的不同之处,正是这些自以为是的修改让我最后的程序性能不如原始的`ggcnn`,然后后面花了一个多月(十一月初至十二月上旬)的时间来查找排除问题,最终定位到自己修改的那几个部分,目前的代码定义啥得已经全部改会和`ggcnn`的**顺序**是一样的了,只是一些函数的定义有点区别,比如后处理部分,这个无伤大雅,就不动了,要说明的是那几个复现过程的 `ipynb` 文件还是按着原来的思路来的,暂时没有时间修改校正,如果我这项目有人看的话,哪怕就一个人看,我也要说明一下这个问题,笔记的基本思路和99%以上的内容都是对的,只是一小部分和`ggcnn`的定义不同而已,不影响理解程序,同时所有的`py`文件都是没有问题的,可以放心使用.