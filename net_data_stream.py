# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 09:27:49 2020

@author: LiuDahui
"""


import torch

from net_data_stream.cornell import Cornell
from net_data_stream.ggcnn import GGCNN

#准备数据集
cornell_data = Cornell('cornell')
dataset = torch.utils.data.DataLoader(cornell_data,batch_size = 1)

#从数据集中读取一个样本
for x,y in dataset:
    xc = x
    yc = y
    break

#实例化一个网络
net = GGCNN(4)

#将输入传递到网络并计算输出
pos,cos,sin,width = net.forward(xc)

#待解决：目前的输入还未经过裁剪处理，结合当前的网络参数还不能获得理想的输出，所以暂时还不能进行训练