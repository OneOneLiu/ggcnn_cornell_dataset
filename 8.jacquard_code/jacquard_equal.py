# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 23:53:20 2020

@author: LiuDahui
"""

#载入第三方库
import torch
import glob
import os
import numpy as np
import random

#载入自定义的库
from image_pro import Image,DepthImage
from grasp_pro import Grasps


class Jacquard(torch.utils.data.Dataset):
    #载入Jacquard数据集的类
    def __init__(self,file_dir,include_depth = True,include_rgb = True,start = 0.0,end = 1.0,ds_rotate = 0,random_rotate = False,random_zoom = False,output_size = 300):
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
        if ds_rotate:
            graspf = graspf[int(l*ds_rotate):] + graspf[:int(l*ds_rotate)]
        depthf = [filename.replace('grasps.txt','perfect_depth.tiff') for filename in graspf]
        rgbf = [filename.replace('perfect_depth.tiff','RGB.png') for filename in depthf]
        
        
        #按照设定的边界参数对数据进行划分并指定为类的属性
        self.graspf = graspf[int(l*start):int(l*end)]
        self.rgbf = rgbf[int(l*start):int(l*end)]
        self.depthf = depthf[int(l*start):int(l*end)]
    
    @staticmethod
    def numpy_to_torch(s):
        '''
        :功能     :将输入的numpy数组转化为torch张量，并指定数据类型，如果数据没有channel维度，就给它加上这个维度
        :参数 s   :numpy ndarray,要转换的数组
        :返回     :tensor,转换后的torch张量
        '''
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))
        
    
    def get_rgb(self,idx,rot = 0,zoom = 1.0,normalize = True):
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
        if normalize:
            rgb_img.normalize()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
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
        
    def get_grasp(self,idx,rot = 0,zoom=1.0):
        grs = Grasps.load_from_jacquard_files(self.graspf[idx],scale = self.output_size/1024.0)#因为图像每个都resize了，所以这里每个抓取框都要缩放
        c = self.output_size//2
        grs.rotate(rot,(c,c))
        grs.zoom(zoom,(c,c))
        
        pos_img,angle_img,width_img = grs.generate_img(self.output_size,self.output_size)
        
        return pos_img,angle_img,width_img
        
    def get_raw_grasps(self,idx,rot = 0,zoom = 1.0):
        '''
        :功能       :读取返回指定id的抓取框信息斌进行一系列预处理(裁剪，缩放等)后以Grasps对象的形式返回
        :参数 idx   :int,要读取的数据id
        :返回       :Grasps，此id中包含的抓取
        '''
        raw_grasps = Grasps.load_from_jacquard_files(self.graspf[idx],scale = self.output_size/1024.0)
        c = self.output_size//2
        raw_grasps.rotate(rot,(c,c))
        raw_grasps.zoom(zoom,(c,c))
        return raw_grasps
    
    def __getitem__(self,idx):
        #一些参数的设置
        #随机旋转的角度在下面四个里面选，并不是完全随机
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
        # 载入深度图像
        #print(idx)
        if self.include_depth:
            depth_img = self.get_depth(idx,rot = rot,zoom = zoom_factor)
            
        # 载入rgb图像
        if self.include_rgb:
            rgb_img = self.get_rgb(idx,rot = rot,zoom = zoom_factor)

        if self.include_depth and self.include_rgb:#如果灰度信息和rgb信息都要的话，就把他们堆到一起构成一个四通道的输入，
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img,0),rgb_img),0
                )
            )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)
        # 载入抓取标注参数
        pos_img,angle_img,width_img = self.get_grasp(idx,rot = rot,zoom = zoom_factor)
        
        # 处理一下角度信息，因为这个角度值区间比较大，不怎么好处理，所以用两个三角函数把它映射一下：
        cos_img = self.numpy_to_torch(np.cos(2*angle_img))
        sin_img = self.numpy_to_torch(np.sin(2*angle_img))
        
        pos_img = self.numpy_to_torch(pos_img)
        
        # 限定抓取宽度范围并将其映射到[0,1]
        width_img = np.clip(width_img, 0.0, 150.0)/150.0
        width_img = self.numpy_to_torch(width_img)
        
        return x,(pos_img,cos_img,sin_img,width_img),idx,rot,zoom_factor#这里多返回idx,rot和zomm_facor参数，方便后面索引查找真实标注，并对真实的标注做同样处理以保证验证的准确性
    def get_jname(self, idx):
        return '_'.join(self.grasp_files[idx].split(os.sep)[-1].split('_')[:-1])
    #映射类型的数据集，别忘了定义这个函数
    def __len__(self):
        return len(self.graspf)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        