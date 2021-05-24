# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 09:27:01 2020

@author: LiuDahui
"""
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import Conv2DTranspose

#网络参数定义
filter_sizes = [32, 16, 8, 8, 16, 32]#这个是滤波器的个数，决定了输出维度
kernel_sizes = [9, 5, 3, 3, 5, 9]#这个是滤波器大小，也即滑窗的大小
strides = [3, 2, 2, 2, 2, 3]

class GGCNN(fluid.dygraph.Layer):
    # 定义抓取预测模型的结构、前向传递过程以及损失计算
    
    def __init__(self,input_channels = 1):
        '''
        :功能                  :类初始化函数
        :参数 input_channels   :int,输入数据的通道数，1或3或4
        :返回                  :None
        '''
        super(GGCNN,self).__init__()
        
        #网络结构定义，直接照搬GGCNN 三层卷积三层反卷积
        self.conv1 = Conv2D(input_channels,filter_sizes[0],kernel_sizes[0],stride=strides[0],padding=3,act = 'relu',param_attr=fluid.initializer.Xavier(uniform=False))
        self.conv2 = Conv2D(filter_sizes[0], filter_sizes[1],kernel_sizes[1], stride=strides[1], padding=2,act = 'relu',param_attr=fluid.initializer.Xavier(uniform=False))
        self.conv3 = Conv2D(filter_sizes[1], filter_sizes[2],kernel_sizes[2], stride=strides[2], padding=1,act = 'relu',param_attr=fluid.initializer.Xavier(uniform=False))
        
        self.convt1 = Conv2DTranspose(filter_sizes[2], filter_sizes[3], kernel_sizes[3], stride=strides[3], act = 'relu',output_size = 50,padding=1,param_attr=fluid.initializer.Xavier(uniform=False))#这里本来有个output_padding参数，但paddle没有，故删了
        self.convt2 = Conv2DTranspose(filter_sizes[3], filter_sizes[4], kernel_sizes[4], stride=strides[4], act = 'relu',output_size = 100,padding=2,param_attr=fluid.initializer.Xavier(uniform=False))
        self.convt3 = Conv2DTranspose(filter_sizes[4], filter_sizes[5], kernel_sizes[5], stride=strides[5], act = 'relu',output_size = 301,padding=3,param_attr=fluid.initializer.Xavier(uniform=False))
        
        self.pos_output = Conv2D(filter_sizes[5], 1, filter_size=2,param_attr=fluid.initializer.Xavier(uniform=False))
        self.cos_output = Conv2D(filter_sizes[5], 1, filter_size=2,param_attr=fluid.initializer.Xavier(uniform=False))
        self.sin_output = Conv2D(filter_sizes[5], 1, filter_size=2,param_attr=fluid.initializer.Xavier(uniform=False))
        self.width_output = Conv2D(filter_sizes[5], 1, filter_size=2,param_attr=fluid.initializer.Xavier(uniform=False))
        
    def forward(self, x):
        '''
        :功能     :前向传播函数
        :参数 x   :tensors,一次网络输入
        :返回     :tensors，各参数的预测结果
        '''
        #print('raw_input:{}'.format(x.shape))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.convt1(x)
        #print('trans1:{}'.format(x.shape))
        x = self.convt2(x)
        #print('trans2:{}'.format(x.shape))
        x = self.convt3(x)
        #print('trans3:{}'.format(x.shape))

        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)
        #print('output:{}'.format(width_output.shape))

        return pos_output, cos_output, sin_output, width_output

    def compute_loss(self, xc, yc):
        '''
        :功能      :损失计算函数
        :参数 xc   :tensors,一次网络输入
        :参数 yc   :tensors,网络输入对应真实标注信息
        :返回      :dict，各损失和预测结果
        '''
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self.forward(xc)
        p_loss = fluid.layers.mse_loss(pos_pred, y_pos)
        cos_loss = fluid.layers.mse_loss(cos_pred, y_cos)
        sin_loss = fluid.layers.mse_loss(sin_pred, y_sin)
        width_loss = fluid.layers.mse_loss(width_pred, y_width)

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