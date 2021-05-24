# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 16:32:03 2020

@author: LiuDahui
"""


import glob
import os
import numpy as np
import random

import paddle
from paddle.io import Dataset

from grasp_pro import Grasps
from image_pro import Image,DepthImage

#不管中间用的什么方法和类，只要保证我的输入输出不变就好了
class Cornell(Dataset):
    def __init__(self,file_dir,include_depth=True,include_rgb=True,start = 0.0,end = 1.0,random_rotate = False,random_zoom = False,output_size = 300):
        #一些参数的传递
        self.include_depth = include_depth
        self.include_rgb = include_rgb
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.output_size = output_size
        #去指定路径载入数据集数据
        graspf = glob.glob(os.path.join(file_dir,'*','pcd*cpos.txt'))

        graspf.sort()
        l = len(graspf)
        if l == 0:
            raise FileNotFoundError('没有查找到数据集，请检查路径{}'.format(file_dir))
        
        rgbf = [filename.replace('cpos.txt','r.png') for filename in graspf]
        depthf = [filename.replace('cpos.txt','d.tiff') for filename in graspf]
        
        #按照设定的边界参数对数据进行划分并指定为类的属性
        self.graspf = graspf[int(l*start):int(l*end)]
        self.rgbf = rgbf[int(l*start):int(l*end)]
        self.depthf = depthf[int(l*start):int(l*end)]
    
    @staticmethod
    def numpy_to_paddle(s):
        if len(s.shape) == 2:
            return np.expand_dims(s,0).astype(np.float32)
        else:
            return s.astype(np.float32)

    def _get_crop_attrs(self,idx):
        grasp_rectangles = Grasps.load_from_cornell_files(self.graspf[idx]) 
        center = grasp_rectangles.center

        left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))

        return center, left, top
    
    def get_rgb(self,idx,rot=0, zoom=1.0,normalize = True):
        rgb_img = Image.from_file(self.rgbf[idx])
        center,left,top = self._get_crop_attrs(idx)
        #先旋转后裁剪再缩放最后resize
        rgb_img.rotate(rot,center)
        rgb_img.crop((top,left),(min(480,top+self.output_size),min(640,left+self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalize:
            rgb_img.normalize()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))

        return rgb_img.img
    
    def get_depth(self,idx,rot=0, zoom=1.0):
        depth_img = DepthImage.from_file(self.depthf[idx])
        center,left,top = self._get_crop_attrs(idx)
        #先旋转后裁剪再缩放最后resize
        depth_img.rotate(rot,center)
        depth_img.crop((top,left),(min(480,top+self.output_size),min(640,left+self.output_size)))
        depth_img.normalize()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))

        return depth_img.img

    def get_grasp(self,idx,rot=0, zoom=1.0):
        grs = Grasps.load_from_cornell_files(self.graspf[idx])
        center, left, top = self._get_crop_attrs(idx)
        #先旋转再偏移再缩放
        grs.rotate(rot,center)
        grs.offset((-top,-left))
        grs.zoom(zoom,(self.output_size//2,self.output_size//2))
        pos_img,angle_img,width_img = grs.generate_img(shape = (self.output_size,self.output_size))
        
        return pos_img,angle_img,width_img
    
    def get_raw_grasps(self,idx,rot=0, zoom=1.0):
        raw_grasps = Grasps.load_from_cornell_files(self.graspf[idx])
        center, left, top = self._get_crop_attrs(idx)
        # 这是paddlepaddel这里报错而专门加的一句，rot.sequeeze()
        raw_grasps.rotate(rot.squeeze(),center)
        raw_grasps.offset((-top,-left))
        raw_grasps.zoom(zoom,(self.output_size//2,self.output_size//2))

        return raw_grasps
    
    def __getitem__(self,idx):
        if self.random_rotate:
            rotations = [0, np.pi/2, 2*np.pi/2, 3*np.pi/2]
            rot = random.choice(rotations)
        else:
            rot = 0.0
        #随机缩放的因子大小果然有限制，太大或者太小都会导致一些问题
        if self.random_zoom:
            zoom_factor = np.random.uniform(0.5, 1.0)
        else:
            zoom_factor = 1.0
        #载入深度图像
        if self.include_depth:
            depth_img = self.get_depth(idx,rot = rot,zoom = zoom_factor)
            x = self.numpy_to_paddle(depth_img)
        #载入rgb图像
        if self.include_rgb:
            rgb_img = self.get_rgb(idx,rot = rot,zoom = zoom_factor)
            #paddle是要求channel-first的，检测一下，如果读进来的图片是channel-last就调整一下，ggcnn中目前我没看到在哪调整的，但肯定是做了的
            if rgb_img.shape[2] == 3:
                rgb_img = np.moveaxis(rgb_img,2,0)
            x = self.numpy_to_paddle(rgb_img)
        #如果灰度信息和rgb信息都要的话，就把他们堆到一起构成一个四通道的输入，
        if self.include_depth and self.include_rgb:
            x = self.numpy_to_paddle(
                np.concatenate(
                    (np.expand_dims(depth_img,0),rgb_img),0
                )
            )
        # 载入抓取标注参数
        pos_img,angle_img,width_img = self.get_grasp(idx,rot = rot,zoom = zoom_factor)

        # 处理一下角度信息，因为这个角度值区间比较大，不怎么好处理，所以用两个三角函数把它映射一下：
        cos_img = self.numpy_to_paddle(np.cos(2*angle_img))
        sin_img = self.numpy_to_paddle(np.sin(2*angle_img))
        
        pos_img = self.numpy_to_paddle(pos_img)
        
        # 限定抓取宽度范围并将其映射到[0,1]
        width_img = np.clip(width_img, 0.0, 150.0)/150.0
        width_img = self.numpy_to_paddle(width_img)

        return x,np.array([pos_img,cos_img,sin_img,width_img]),idx,rot,zoom_factor
    
    def __len__(self):
        return len(self.graspf)