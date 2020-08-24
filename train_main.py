#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 15:32:02 2020

@author: ldh
"""

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from train.cornell_pro import Cornell
from train.ggcnn import GGCNN

batch_size = 64

#准备数据集
cornell_data = Cornell('cornell',output_size = 250)
dataset = torch.utils.data.DataLoader(cornell_data,batch_size = batch_size)

#从数据集中读取一个样本
for x,y in dataset:
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

print(x.shape)
#动态显示每次优化过后的预测结果
plt.ion()
plt.show()

#想要查看的结果编号num<batch_size
num = 60


for i in range(200):
    losses = net.compute_loss(x,y)
    loss = losses['loss']
    print(i)
    print(loss)
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