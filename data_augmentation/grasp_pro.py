# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 09:23:54 2020

@author: LiuDahui
"""


import numpy as np
from skimage.draw import polygon


def str2num(point):
    '''
    :功能  :将字符串类型存储的抓取框脚点坐标取整并以元组形式返回
    
    :参数  :point,字符串，以字符串形式存储的一个点的坐标
    :返回值 :列表，包含int型抓取点数据的列表[x,y]
    '''
    x,y = point.split()
    x,y = int(round(float(x))),int(round(float(y)))
    
    return np.array([x,y])



class Grasp:
    '''定义一个抓取框处理类，主要功能是将由四个角点坐标定义的原始的抓取框提取转化为训练所定义的表征信息，如中心位置，面积角度等，并根据图像处理需求完成一些相应的其他功能'''
    def __init__(self,points):
        '''
        :功能        : 类初始化函数
        :参数 points : 2darry,定义一个抓取框的四个角点坐标信息[[x1,y1],[x2,y2],[x3,y3],[x4,x4]]
        '''
        self.points = points
       
    @property#类装饰器，可以让一个类的方法以属性的方式被调用
    def center(self):
        '''
        :功能          : 计算本类中所包含的抓取框的中心点
        :返回 1darray  : 本类所包含抓取框的中心点array[x,y]
        '''
        center = np.mean(self.points,axis = 0).astype(np.int)
        return center
    
    @property
    def width(self):
        '''
        :功能          : 计算本类中所包含的抓取框手指张开宽度width
        :返回 1darray  : 本类所包含抓取框的长度[width]
        '''
        #第二个点和第三个点之间的间距长度
        dx = self.points[0][0] - self.points[1][0]
        dy = self.points[0][1] - self.points[1][1]
        
        return np.sqrt(dx**2+dy**2)
    
    @property
    def length(self):
        '''
        :功能          : 计算本类中所包含的抓取框长度(手指张开宽度width的邻边)
        :返回 1darray  : 本类所包含抓取框的长度[length]
        '''
        #第二个点和第三个点之间的间距长度
        dx = self.points[1][0] - self.points[2][0]
        dy = self.points[1][1] - self.points[2][1]
        
        return np.sqrt(dx**2+dy**2)
    
    @property
    def angle(self):
        '''
        :功能          : 计算本类中所包含的抓取框相对于x轴正方向的偏转角度
        :返回 1darray  : 本类所包含抓取框的旋转角度（弧度值）
        ''' 
        
        dx = self.points[0][0] - self.points[1][0]
        dy = self.points[0][1] - self.points[1][1]
        
        return (np.arctan2(-dy,dx) + np.pi/2) % np.pi - np.pi/2
    
    def compact_polygon_coords(self,shape):
        '''
        :功能          : 计算并返回本抓取矩形内部点的坐标
        :参数 shape    : tuple, optional.Image shape which is used to determine the maximum extent of output pixel coordinates.
        :返回 ndarray  : rr,cc 本抓取框内部点的行列坐标
        ''' 
        return polygon(self.points[:,0],self.points[:,1],shape)

class Grasps:
    '''定义一个多抓取框处理类，主要功能是从原始的标注文件中读出多个抓取框并将其构建成多个单一的抓取框Grasp类，同时能够对这些属于同一对象的多个抓取框对象进行一些数据的统一集成处理'''
    def __init__(self,grs = None):
        '''
        :功能     : 多抓取框类初始化函数，功能是将属于一个对象的多个单独的抓取框集成到一个类里面来。
        :参数 grs : list,包含一个对象中多个抓取框类的列表
        '''
        if grs:
            self.grs = grs
        else:
            self.grs = []
    
    @classmethod
    def load_from_cornell_files(cls,cornell_grasp_files):
        '''
        :功能     : 从一个graspf文件中读取载入多个抓取框并构建成为这个类（其实就是从之前的那个get_rectangles改的）
        :参数 grs : list,包含一个对象中多个抓取框类的列表
        '''
        grasp_rectangles = []
        with open(cornell_grasp_files,'r') as f:
            while True:
                grasp_rectangle = []
                point0 = f.readline().strip()
                if not point0:
                    break
                point1,point2,point3 = f.readline().strip(),f.readline().strip(),f.readline().strip()
                if point0[0] == 'N':#后面发现有些坐标点坐标是NaN，会报错，这里处理一下，暂时还不晓得gg-cnn里面怎么处理的
                    break
                grasp_rectangle = np.array([str2num(point0),
                               str2num(point1),
                               str2num(point2),
                               str2num(point3)])
                grasp_rectangles.append(Grasp(grasp_rectangle))#找出各个框后就直接用它构造Grasp对象了

            return cls(grasp_rectangles)#返回实例化的类
        
    def generate_img(self,pos = True,angle = True,width = True,shape = (480,640)):
        '''
        :功能       :将本对象的多个的抓取框信息融合并生成指定的映射图，以这种方式返回定义一个抓取的多个参数，包括中心点，角度，宽度
        :参数 pos   :bool,是否生成返回位置映射图
        :参数 angle :bool,是否生成返回角度映射图
        :参数 width :bool,是否生成返回夹爪宽度映射图
        :参数 shape :tuple
        :返回       :融合本对象的多个抓取框信息的映射图
        '''
        
        if pos:
            pos_out = np.zeros(shape)
        else:
            pos_out = None
        if angle:
            angle_out = np.zeros(shape)
        else:
            angle_out = None
        if width:
            width_out = np.zeros(shape)
        else:
            width_out = None
        
        for gr in self.grs:
            rr,cc = gr.compact_polygon_coords(shape)#shape的指定还是很重要的，可以考虑图像边界
            
            if pos:
                pos_out[cc,rr] = 1.0
            if angle:
                angle_out[cc,rr] = gr.angle
            if width:
                width_out[cc,rr] = gr.width

        return pos_out,angle_out,width_out
    @property
    def center(self):
        '''
        :功能       :计算本类中所包含的多个抓取框共同的中心
        :返回       :ndarray，中心坐标
        '''
        centers = []
        for gr in self.grs:
            centers.append(gr.center)
        center = np.mean(np.array(centers),axis = 0).astype(np.uint)
        return center