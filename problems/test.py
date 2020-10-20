#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 11:03:09 2020

@author: ldh
"""

import cv2


import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from cornell_pro import Cornell
from ggcnn import GGCNN

batch_size = 16

#准备数据集
cornell_data = Cornell('../cornell',output_size = 300)
dataset = torch.utils.data.DataLoader(cornell_data,batch_size = batch_size)

#从数据集中读取一个样本
for x,y,_ in dataset:
    xc = x
    yc = y
    break

#实例化一个网络
net = GGCNN(4)

#定义一个优化器
optimizer = optim.Adam(net.parameters())

#设置GPU设备
device = torch.device("cuda:0")

net = net.to(device)

x = xc.to(device)
y = [yy.to(device) for yy in yc]

#动态显示每次优化过后的预测结果
plt.ion()
plt.show()

#想要查看的结果编号num<batch_size
num = 10


for i in range(200):
    losses = net.compute_loss(x,y)
    loss = losses['loss']
    #反向传播优化
    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()
    
    if i % 5 == 0:
        plt.cla()
        pos,cos,sin,width = net.forward(x)
        pos = pos.cpu()
        cos = cos.cpu()
        sin = sin.cpu()
        width = width.cpu()
        plt.subplot(141)
        plt.title('pos_out')
        plt.imshow(pos[num][0].data.numpy(),cmap=plt.cm.gray)
        plt.subplot(142)
        plt.title('cos_out')
        plt.imshow(cos[num][0].data.numpy(),cmap=plt.cm.gray)
        plt.subplot(143)
        plt.title('sin_out')
        plt.imshow(sin[num][0].data.numpy(),cmap=plt.cm.gray)
        plt.subplot(144)
        plt.title('width_out')
        plt.imshow(width[num][0].data.numpy(),cmap=plt.cm.gray)
        plt.pause(0.01)


#将输入传递到网络并计算输出
pos,cos,sin,width = net.forward(xc.to(device))

q_img = pos.cpu().data.numpy().squeeze()
ang_img = (torch.atan2(sin, cos) / 2.0).cpu().data.numpy().squeeze()
width_img = width.cpu().data.numpy().squeeze()



img_guassian = cv2.blur(np.abs(q_img),(3,3),0)

plt.subplot(311)
plt.title('before_abs')
plt.imshow(q_img[0],cmap=plt.cm.gray)
plt.subplot(312)
plt.title('before_gaussian')
plt.imshow(np.abs(q_img[0]),cmap=plt.cm.gray)
plt.subplot(313)
plt.title('after_guassian')
plt.imshow(img_guassian[0],cmap=plt.cm.gray)
plt.show()