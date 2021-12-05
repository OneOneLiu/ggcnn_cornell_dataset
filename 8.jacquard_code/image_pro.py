# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:17:44 2020

@author: LiuDahui
"""
from imageio import imread
import numpy as np
from skimage.transform import resize,rotate
import random

class Image:
    '''定义一个图像类，主要功能是将原始的图像输入转化为适合网络训练的格式并根据图像处理需求完成一些其他功能'''
    def __init__(self,img):
        '''
        :功能 :类初始化函数`
        :参数 :ndarray,原始图像
        '''
        self.img = img
    
    @classmethod
    def from_file(cls,file_path):
        '''
        :功能           : 从原始图片的路径对其进行载入
        :参数 file_path : str,原始图像所在的路径
        :返回 class     : 由指定路径的原始图片实例化的Image类
        :备注           : 这里用到的cls方法要学习一下
        '''
        return cls(imread(file_path))
    
    def img_format(self):
        '''
        :功能 :将原始图像转换为指定格式
        '''
        pass
    
    def normalize(self):
        '''
        :功能 :将图像像素值标准化至[0,1]范围
        :功能 :将图像像素值标准化至[0,1]范围
        '''
        self.img = self.img.astype('float32')/255.0
        self.img = self.img-self.img.mean()
        
    def crop(self,top_left, bottom_right):
        '''
        :功能              :按照给定参数对图像进行裁剪操作
        :参数 top_left     :ndarray,要裁剪区域的左上角点坐标
        :参数 bottom_right :ndarray,要裁剪区域的右下角点坐标
        '''
        self.img = self.img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    
    def resize(self,shape):
        '''
        :功能           :将图片resize成指定的shape
        :参数 shape     :ndarray,要裁剪区域的左上角点坐标
        '''
        if self.img.shape == shape:
            return
        self.img = resize(self.img, shape, preserve_range=True).astype(self.img.dtype)
    
    def rotate(self,angle,center = None):
        '''
        :功能           :将图片绕指定中心旋转指定角度
        :参数 angle     :要旋转的角度（弧度制）
        :参数 center    :旋转中心像素坐标，如不指定则默认为图像中心像素坐标
        '''
        if center is not None:
            center = (int(center[1]),int(center[0]))#不管你原来什么数据类型，这里都变成tuple,而且这边不转整型的话后面旋转就会出错，所以转换了一下
        self.img = rotate(self.img,angle/np.pi*180,center = center,mode = 'symmetric',preserve_range = True).astype(self.img.dtype)
    
    def zoom(self,factor):
        '''
        :功能        ：通过裁剪和resize来实现缩放操作，注意，缩放并不是直接一个函数一步实现的
        :参数 factor ：缩放比例因子，比如设置为0.5就是裁剪原图50%的区域
        '''
        sr = int(self.img.shape[0] * (1 - factor)) // 2
        sc = int(self.img.shape[1] * (1 - factor)) // 2
        
        orig_shape = self.img.shape
        self.img = self.img[sr:self.img.shape[0] - sr, sc: self.img.shape[1] - sc].copy()
        self.img = resize(self.img, orig_shape, mode='symmetric', preserve_range=True).astype(self.img.dtype)

class DepthImage(Image):
    '''深度图像类，读取，载入，正则等预处理都是一样的，后面可能会添加一些针对深度图的专属处理功能'''
    def __init__(self,img):
        super().__init__(img)

    def normalize(self):
        """
        通过减去均值并修剪至[-1,1]的范围的方式进行正则化,与RGB的处理不同
        """
        self.img = np.clip((self.img - self.img.mean()), -1, 1)
        self.img = self.img + random.randint(-200,200)/1000.0
        self.img = self.img + np.random.normal(size = (1024,1024))/200.0
        self.img = self.img + get_gradation_2d(np.random.randint(0,20),np.random.randint(0,20),1024,1024,np.random.randint(0,2))/100.0
    
    @classmethod
    def from_tiff(cls, fname):
        return cls(imread(fname))

def get_gradation_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T