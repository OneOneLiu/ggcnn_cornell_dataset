# cornell_dataset

cornell grasp dataset analyses and process

追随着大佬的脚步，从头复现一下大佬的论文程序，大佬的代码项目地址为：https://github.com/dougsm/ggcnn

Follow the big guy's step to re-implement his excellent ggcnn code,his repositiry address is :https://github.com/dougsm/ggcnn

文件夹：
|-cornell:存放原始数据集解压后数据

|-images:存放一些笔记中用到的图片

|-support_files:存放一些积累答疑文件


文件说明：

后缀为ipynb的文件为相应程序的原始建立过程

后缀为py的文件为建立好的完整程序


查看顺序为：

1.load_data.ipynb

2.net_data_stream.ipynb

3.data_augmentation.ipynb


与ggcnn中定义的不同之处：

1.width和length是反过来的
2.角点的x,y坐标好像也是反过来的

note:

1.刚发现如果文件在jupyter中处于打开状态的话，git这边无法push，记录一下以后注意。
2.截至到data_augmentation的程序在hp上测试无误，不晓得为什么在ubuntu上就不行，报了一个错，明天好好检查一下
