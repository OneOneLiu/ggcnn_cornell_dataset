# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 23:53:20 2020

@author: LiuDahui
"""

#载入第三方库
import torch
import glob
import os

#载入自定义的库
from image_pro import Image,DepthImage
from grasp_pro import Grasps


class Jacquard(torch.utils.data.Dataset):
    #载入Jacquard数据集的类
    def __init__(self,file_dir,include_depth = True,include_rgb = True,start = 0.0,end = 1.0,random_rotate = False,random_zoom = False,output_size = 300):
        '''
        参数
        ----------
        file_dir : str
            存放Jacquard数据集的路径.
        include_depth : bool, optional
            描述. 是否包含深度图像数据，The default is True.
        include_rgb : bool, optional
            描述. 是否包含RGB图像数据The default is True.
        start : float, optional
            描述. 数据集划分起点参数，The default is 0.0.
        end : float, optional
            描述. 数据集划分终点参数，The default is 1.0.
        Returns
        -------
        None.
        '''
        super(Jacquard,self).__init__()
        
        #一些参数的传递
        self.include_depth = include_depth
        self.include_rgb = include_rgb
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.output_size = output_size
        #去指定路径载入数据集数据
        graspf = glob.glob(os.path.join(file_dir,'*','*','*_grasps.txt'))
        graspf.sort()
        
        
        l = len(graspf)
        if l == 0:
            raise FileNotFoundError('没有查找到数据集，请检查路径{}'.format(file_dir))
        
        rgbf = [filename.replace('grasps.txt','RGB.png') for filename in graspf]
        depthf = [filename.replace('grasps.txt','perfect_depth.tiff') for filename in graspf]
        
        #按照设定的边界参数对数据进行划分并指定为类的属性
        self.graspf = graspf[int(l*start):int(l*end)]
        self.rgbf = rgbf[int(l*start):int(l*end)]
        self.depthf = depthf[int(l*start):int(l*end)]
        
    
    def get_rgb(self,idx,rot = 0,zoom = 1.0,normalise = True):
        '''
        :功能     :读取返回指定id的rgb图像
        :参数 idx :int,要读取的数据id
        :返回     :ndarray,处理好后的rgb图像
        '''
        rgb_img = Image.from_file(self.rgbf[idx])
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        #注意，这里暂且不论是否zoom，图像是直接由（1024X1024）resize成为了(300X300)的，相应的抓取框标注也必须进行同样程度的“resize”或者说缩放（scale）才行
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            #这里还有一句transpose没写，先不管
        return rgb_img.img
        
    def get_depth(self,idx,rot = 0,zoom = 1.0):
        '''
        :功能     :读取返回指定id的depth图像
        :参数 idx :int,要读取的数据id
        :返回     :ndarray,处理好后的depth图像
        '''
        depth_img = DepthImage.from_file(self.depthf[idx])
        depth_img.rotate(rot)
        depth_img.normalize()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size,self.output_size))
        
        return depth_img.img
        
    def get_grasp(self,idx,rot,zoom):
        grs = Grasps.
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        