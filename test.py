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
cornell_data = Cornell('cornell')
dataset = torch.utils.data.DataLoader(cornell_data,batch_size = batch_size)
    

#实例化一个网络
net = GGCNN(4)

#定义一个优化器
optimizer = optim.Adam(net.parameters())

#设置GPU设备
device = torch.device("cuda:0")

net = net.to(device)

#想要查看的结果编号num<batch_size
num = 60

for x,y in dataset:
    xc = x.to(device)
    yc = [yy.to(device) for yy in y]
    #动态显示每次优化过后的预测结果
    for i in range(200):
        losses = net.compute_loss(xc,yc)
        loss = losses['loss']
        print(i)
        print(loss)
        #反向传播优化
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()