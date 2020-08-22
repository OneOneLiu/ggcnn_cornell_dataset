# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 09:27:49 2020

@author: LiuDahui
"""


import torch
import matplotlib.pyplot as plt
import numpy as np

from data_augmentation.cornell_pro import Cornell
from data_augmentation.ggcnn import GGCNN

#准备数据集
cornell_data = Cornell('cornell')
dataset = torch.utils.data.DataLoader(cornell_data,batch_size = 32)

#想查看batch里面的第几个i就等于几-1
i = 30

#从数据集中读取一个样本
for x,y in dataset:
    xc = x
    yc = y
    break

#简单可视化输入一下
depth_img = xc[i][0].data.numpy()
rgb_img = xc[i][1:4].data.numpy()

rgb_img = np.moveaxis(rgb_img,0,2)*255

pos_img = yc[0][i][0].data.numpy()
cos_img = yc[1][i][0].data.numpy()
sin_img = yc[2][i][0].data.numpy()
width_img = yc[3][i][0].data.numpy()

plt.figure()

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

#可视化一下输出
plt.figure()

plt.subplot(141)
plt.title('pos_out')
plt.imshow(pos[i][0].data.numpy())
plt.subplot(142)
plt.title('cos_out')
plt.imshow(cos[i][0].data.numpy())
plt.subplot(143)
plt.title('sin_out')
plt.imshow(sin[i][0].data.numpy())
plt.subplot(144)
plt.title('width_out')
plt.imshow(width[i][0].data.numpy())
plt.show()
