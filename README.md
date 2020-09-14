# cornell_dataset

cornell grasp dataset analyses and process

追随着大佬的脚步，从头复现一下大佬的论文程序，大佬的代码项目地址为：https://github.com/dougsm/ggcnn

Follow the big guy's step to re-implement his excellent ggcnn code,his repositiry address is :https://github.com/dougsm/ggcnn

文件夹：

|-cornell:存放原始数据集解压后数据(注意，项目文件夹中并不包含这个文件，太大了传不上来，只需要自己在官网下载解压，然后使用gg-cnn里面的程序根据点云生成tiff图片就行了,程序在大佬的项目中，路径为：utils->dataset_processing->generate_cornell_depth.py)

|-images:存放一些笔记中用到的图片

|-support_files:存放一些积累答疑文件

|-net_data_stream:数据流建立相关文件

|-data_augmentation:数据增强处理相关文件

|-train:训练调试过程相关文件

|-validate:模型性能验证程序调试过程相关文件

|-trained_models:训练好的模型

文件说明：

后缀为ipynb的文件为相应程序的原始建立过程

后缀为py的文件为与ipynb对应的建立好的完整程序

trian_model:一个已经初步训练好的模型
查看顺序为：

1.load_data.ipynb

2.net_data_stream.ipynb

3.data_augmentation.ipynb

4.train.ipynb

5.validate.ipynb


与ggcnn中定义的不同之处：

1.width和length是反过来的

2.角点的x,y坐标好像也是反过来的

3.在我的定义里面，抓取框对象的center属性，center[0]是x坐标，减去二分之一output size后对应left坐标，center[1]是y坐标，所以crop的时候，我的是（left,top），ggcnn是（top,left）

note:

1.刚发现如果文件在jupyter中处于打开状态的话，git这边无法push，记录一下以后注意。

2.截至到data_augmentation的程序在hp上测试无误，不晓得为什么在ubuntu上就不行，报了一个错，明天好好检查一下，已经解决，详见support_files/functions 第5点

3.调试过程中遇到的重大bug1：调节output_size后网络的输出与标注shape不匹配导致的一系列问题，问题原因：输入维度发生改变后，网络结构不变，输出的shape本就会发生变化，想要控制具体的输出尺寸就必须对网络参数进行仔细设计调整

4.调试过程中遇到的重大bug2：Gaussian滤波后的十字问题，问题原因：所使用的Gaussian滤波函数不支持一次输入多个样本，val过程中batch_size的设定应当为1

5.调试过程中遇到的重大bug3：这个应该不能说是重大bug，因为，这是一个失误，之前其实已经发现了，就是，可视化绘制抓取框的时候，不要用rectangles函数，这函数只能根据你给的两个角点画出标准的，竖直的矩形，画这种斜的矩形要用line函数


