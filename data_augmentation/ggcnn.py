# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 09:27:01 2020

@author: LiuDahui
"""


import torch.nn as nn
import torch.nn.functional as F

#网络参数定义
filter_sizes = [32, 16, 8, 8, 16, 32]
kernel_sizes = [9, 5, 3, 3, 5, 9]
strides = [3, 2, 2, 2, 2, 3]

class GGCNN(nn.Module):
    #定义抓取预测模型的结构、前向传递过程以及损失计算
    
    def __init__(self,input_channels = 1):
        '''
        :功能                  :类初始化函数
        :参数 input_channels   :int,输入数据的通道数，1或3或4
        :返回                  :None
        '''
        super(GGCNN,self).__init__()
        
        #网络结构定义，直接照搬GGCNN 三层卷积三层反卷积
        self.conv1 = nn.Conv2d(input_channels,filter_sizes[0],kernel_sizes[0],stride=strides[0],padding=3)
        self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1],kernel_sizes[1], stride=strides[1], padding=2)
        self.conv3 = nn.Conv2d(filter_sizes[1], filter_sizes[2],kernel_sizes[2], stride=strides[2], padding=1)
        
        self.convt1 = nn.ConvTranspose2d(filter_sizes[2], filter_sizes[3], kernel_sizes[3], stride=strides[3], padding=1, output_padding=1)
        self.convt2 = nn.ConvTranspose2d(filter_sizes[3], filter_sizes[4], kernel_sizes[4], stride=strides[4], padding=2, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(filter_sizes[4], filter_sizes[5], kernel_sizes[5], stride=strides[5], padding=3, output_padding=1)
        
        self.pos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.cos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.sin_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.width_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        
        #使用Glorot初始化法初始化权重
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)
        
    def forward(self, x):
        '''
        :功能     :前向传播函数
        :参数 x   :tensors,一次网络输入
        :返回     :tensors，各参数的预测结果
        '''
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))
        x = F.relu(self.convt3(x))

        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

    def compute_loss(self, xc, yc):
        '''
        :功能      :损失计算函数
        :参数 xc   :tensors,一次网络输入
        :参数 yc   :tensors,网络输入对应真实标注信息
        :返回      :dict，各损失和预测结果
        '''
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }