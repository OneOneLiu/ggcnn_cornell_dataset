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
        center = np.mean(self.points,axis = 0).astype(np.uint32)
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
    
    def polygon_coords(self, shape=None):
        """
        :功能          : 计算并返回给定抓取矩形内部点的坐标
        :参数 shape    : tuple, optional.Image shape which is used to determine the maximum extent of output pixel coordinates.
        :返回 1darray  : rr,cc 本抓取框内部点的行列坐标
        """
        return polygon(self.points[:, 0], self.points[:, 1], shape)
    
    def compact_polygon_coords(self,shape):
        '''
        :功能          : 计算并返回本抓取矩形内部点的坐标
        :参数 shape    : tuple, optional.Image shape which is used to determine the maximum extent of output pixel coordinates.
        :返回 ndarray  : rr,cc 本抓取框内部点的行列坐标
        ''' 
        return Grasp_cpaw(self.center, self.angle, self.length, self.width/3).as_gr.polygon_coords(shape)
    
    def offset(self, offset):
        """
        :参数       : 偏移类中所包含抓取框的坐标
        :参数 offset: array [y, x] 要偏移的距离
        """
        self.points += np.array(offset).reshape((1, 2))#这个操作可以保证所有的x坐标都加上x的偏移量，所有的y都加上y的偏移量
    
    def rotate(self,angle,center):
        '''
        :功能        ：将抓取标注矩形按照给定的旋转角度和旋转中心进行逆时针旋转
        :参数 angle  ：要旋转的角度（弧度制）
        :参数 center ：旋转中心
        '''
        #定义旋转矩阵
        R = np.array(
            [
                [np.cos(angle), np.sin(angle)],
                [-1 * np.sin(angle), np.cos(angle)],
            ]
        )
        #处理旋转中心
        c = np.array(center).reshape((1, 2))
        #执行旋转运算
        self.points = ((np.dot(R, (self.points - c).T)).T + c).astype(np.int)
    
    def zoom(self,factor,center):
        '''
        :功能         :按照指定的缩放因子(factor)和中心点来缩放抓取矩形
        :参数：factor :缩放因子
        :参数：center :缩放中心
        '''
        #缩放矩阵定义
        T = np.array(
            [
                [1/factor,0],
                [0,1/factor]
            ]
        )
        c = np.array(center).reshape((1,2))
        
        self.points = ((np.dot(T,(self.points - c).T)).T+c).astype(np.int)
    
    def scale(self,factor):
        '''
        :功能         :按照指定的缩放因子(factor)来缩放抓取矩形
        :参数：factor :缩放因子
        :参数：center :缩放中心
        '''
        if factor == 1.0:
            return
        self.points *= factor


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
        
    def __getattr__(self, attr):
        """
        当用户调用某一个Grasps类中没有的属性时，查找Grasp类中有没有这个函数，有的话就对Grasps类中的每个Grasp对象调用它。
        这里是直接从ggcnn里面搬运过来的，高端操作，，，学到了
        """
        # Fuck yeah python.
        if hasattr(Grasp, attr) and callable(getattr(Grasp, attr)):
            return lambda *args, **kwargs: list(map(lambda gr: getattr(gr, attr)(*args, **kwargs), self.grs))
        else:
            raise AttributeError("在BoundingBoxes or BoundingBox中找不到函数 %s " % attr)
    
    @classmethod
    def load_from_cornell_files(cls,cornell_grasp_files):
        '''
        :功能                     : 从一个graspf文件中读取载入多个抓取框并构建成为这个类（其实就是从之前的那个get_rectangles改的）
        :参数 cornell_grasp_files : str,目标文件路径
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
    
    @classmethod
    def load_from_jacquard_files(cls,jacquard_grasp_files,scale = 1.0):
        '''
        :功能                      : 从一个graspf文件中读取载入多个抓取框并构建成为这个类
        :参数 jacquard_grasp_files : str,目标文件路径
        :参数 scale                : flote,抓取框缩放比例因子，因为每幅图像都是resize了的，所以抓取框也必须进行同样的缩放处理才能保证匹配
        '''
        grasp_rectangles = []
        with open(jacquard_grasp_files) as f:
            for line in f:
                x, y, theta, w, h = [float(v) for v in line[:-1].split(';')]
                grasp_rectangles.append(Grasp_cpaw(np.array([x,y]),-theta/180.0*np.pi,h,w).as_gr)#我这边读取的顺序跟GGCNN中的有些不同
        grasp_rectangles = cls(grasp_rectangles)
        grasp_rectangles.scale(scale)
        return grasp_rectangles#返回实例化的类

    def generate_img(self,pos = True,angle = True,width = True,shape = (300,300)):
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
    def points(self):
        '''
        :功能       :返回本类中所包含的多个抓取框的角点
        :返回       :ndarray，抓取框角点坐标
        '''
        points = []
        for gr in self.grs:
            points.append(gr.points)
        return points

    @property
    def center(self):
        '''
        :功能       :计算本类中所包含的多个抓取框共同的中心
        :返回       :ndarray，中心坐标
        '''
        centers = []
        for gr in self.grs:
            centers.append(gr.center)
        center = np.mean(np.array(centers),axis = 0).astype(np.uint32)
        return center


class Grasp_cpaw:
    '''
    前面的抓取类都是由四个角点坐标信息定义的，如果想对框进行什么操作的话，不太方便，
    这里使用其中提取出的中心点坐标，角度，以及长宽来定义一个矩形，对于这个矩形的整体处
    理比较方方便（比如将矩形的宽度缩小三倍），但是最终的绘制还是要通过角点坐标来实现，所以，里面还要有一个能够根据
    这几个参数反求角点坐标的函数
    '''
    def __init__(self,center, angle, length=60, width=30):
        '''
        :功能       :类初始化函数，进行参数传递
        :参数       :这些参数是啥很明显了吧，就不再赘述了
        '''
        self.center = center
        self.angle = angle   # 正角度表示沿水平方向逆时针旋转
        self.length = length
        self.width = width
        
    @property
    def as_gr(self):
        '''
        :功能       :通过这几个参数反求所定义的坐标角点，并由其建立返回Grasp对象
        :返回       :由反求出的角点所定义的Grasp对象
        '''
        xo = np.cos(self.angle)
        yo = np.sin(self.angle)
        
        y1 = self.center[0] - self.width / 2 * xo
        x1 = self.center[1] + self.width / 2 * yo
        y2 = self.center[0] + self.width / 2 * xo
        x2 = self.center[1] - self.width / 2 * yo
        
        return Grasp(np.array(
            [
             [y1 - self.length/2 * yo, x1 - self.length/2 * xo],
             [y2 - self.length/2 * yo, x2 - self.length/2 * xo],
             [y2 + self.length/2 * yo, x2 + self.length/2 * xo],
             [y1 + self.length/2 * yo, x1 + self.length/2 * xo],
             ]
        ).astype(np.float))#搞成整数后返回，后面可视化的时候比较好处理