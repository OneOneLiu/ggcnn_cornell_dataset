# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 09:27:49 2020

@author: LiuDahui
"""


import torch
import matplotlib.pyplot as plt
import numpy as np

from cornell import Cornell
from ggcnn import GGCNN

#准备数据集
cornell_data = Cornell('../cornell')
dataset = torch.utils.data.DataLoader(cornell_data,batch_size = 1)

#从数据集中读取一个样本
for x,y in dataset:
    xc = x
    yc = y
    break

#简单可视化它一下
depth_img = xc[0][0].data.numpy()
rgb_img = xc[0][1:4].data.numpy()

rgb_img = np.moveaxis(rgb_img,0,2)*255

pos_img = yc[0][0][0].data.numpy()
cos_img = yc[1][0][0].data.numpy()
sin_img = yc[2][0][0].data.numpy()
width_img = yc[3][0][0].data.numpy()

plt.subplot(231)
plt.title('depth_input')
plt.imshow(depth_img)
plt.subplot(232)
plt.title('rgb_input')
plt.imshow(rgb_img)
plt.subplot(233)
plt.title('pos_input')
plt.imshow(pos_img)
plt.subplot(234)
plt.title('cos_input')
plt.imshow(cos_img)
plt.subplot(235)
plt.title('sin_input')
plt.imshow(sin_img)
plt.subplot(236)
plt.title('width_input')
plt.imshow(width_img)
plt.show()


#实例化一个网络
net = GGCNN(4)

#将输入传递到网络并计算输出
pos,cos,sin,width = net.forward(xc)

#待解决：目前的输入还未经过裁剪处理，结合当前的网络参数还不能获得理想的输出，所以暂时还不能进行训练