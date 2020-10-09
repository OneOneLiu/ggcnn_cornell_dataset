#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 15:32:02 2020

#这个程序是在一个batch上测试的程序，只在一个样本上训练的，并没有遍历整个数据集，train_main2.py是遍历了整个数据集的

@author: ldh
"""

import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from train.cornell_pro import Cornell
from train.ggcnn import GGCNN

batch_size = 32

#准备数据集
cornell_data = Cornell('cornell',output_size = 300)
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
fig = plt.figure()
plt.ion()
plt.show()

#想要查看的结果编号num<batch_size
num = 10

loss_results = []
max_results = []
width_results = []
for i in range(1000):
    losses = net.compute_loss(x,y)
    loss = losses['loss']
    #反向传播优化
    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()
    pos,cos,sin,width = net.forward(x)
    loss_results.append(loss)
    max_results.append(pos.cpu().data.numpy().max())
    width_results.append(width.cpu().data.numpy().max())
    if i % 2 == 0:
        print(loss)
        plt.cla()
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
        
        #拿它做个预测试试
        fig.suptitle('epoch:{0}\n loss:{1} \n max:{2}'.format(i,loss, pos.cpu().data.numpy().max()))

fig2 = plt.figure()
fig2.suptitle('loss and q_img & width_img max value')
plt.plot(loss_results,label = 'loss')
plt.plot(max_results,label = 'q_img_max')
plt.plot(width_results,label = 'width_img_max')
plt.legend()
plt.show()
torch.save(net,'trained_models/model1')