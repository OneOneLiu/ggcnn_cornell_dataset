# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 21:52:23 2020
validate部分用到的一些函数汇总
@author: LiuDahui
"""
import torch
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.draw import polygon
import numpy as np

from validate.grasp_pro import Grasp_cpaw

def post_process(pos_img,cos_img,sin_img,width_img):
    '''
    :功能           :对原始的网络输出进行预处理，包括求解角度数据和高斯滤波
    :参数 pos_img   :cuda tensor,原始输出的抓取位置映射图
    :参数 cos_img   :cuda tensor,原始输出的抓取角度cos值映射图
    :参数 sin_img   :cuda tensor,原始输出的抓取角度sin值映射图
    :参数 wid_img   :cuda tensor,原始输出的抓取宽度值映射图
    :返回           :3个ndarray，分别是抓取质量（位置）映射图，抓取角度映射图以及抓取宽度映射图
    '''
    q_img = pos_img.cpu().data.numpy().squeeze()
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().data.numpy().squeeze()
    width_img = width_img.cpu().data.numpy().squeeze()
    
    #一定注意，此处Guassian滤波时，batch_size一定要是1才行，这个滤波函数不支持一次输入多个样本
    q_img_g = gaussian(q_img, 2.0, preserve_range=True)
    ang_img_g = gaussian(ang_img, 2.0, preserve_range=True)
    width_img_g = gaussian(width_img, 1.0, preserve_range=True)
    
    return q_img_g, ang_img_g, width_img_g

def detect_grasps(q_out,ang_out,wid_out = None,no_grasp = 1):
    '''
    :功能          :从抓取预测处理所得到的位置，角度，宽度映射图中提取no_grasp个最有效的抓取
    :参数 q_out    :int,抓取质量（位置）映射图
    :参数 ang_out  :int,抓取角度映射图
    :参数 wid_out  :int,抓取宽度映射图
    :参数 no_grasp :int,想要提取的有效抓取个数
    :返回          :list,包含多个grasp_cpaw对象的列表
    '''
    grasps_pre = []
    local_max = peak_local_max(q_out, min_distance=20, threshold_abs=0.1,num_peaks = 1)
    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)
        grasp_angle = ang_out[grasp_point]
        grasp_width = wid_out[grasp_point]
    g = Grasp_cpaw(grasp_point,grasp_angle,grasp_width)
    if wid_out is not None:
        g.width = wid_out[grasp_point]*150
        g.length = g.width/2
    grasps_pre.append(g)
    
    return grasps_pre

def max_iou(grasp_pre,grasps_true):
    '''
    :功能 :对于一个给定的预测抓取框，首先将其转化为Grasp对象，然后遍历计算其与各个真是标注的iou，返回最大的iou
    :参数 : grasp_pre  :Grasp对象，单个预测结果中反求出的抓取框
    :参数 : grasps_true:Grasps对象，该对象所有的真实标注抓取框
    :返回 : 最大的iou
    '''
    grasp_pre = grasp_pre.as_gr
    max_iou = 0
    for grasp_true in grasps_true.grs:
        if iou(grasp_pre,grasp_true) > max_iou:
            max_iou = iou(grasp_pre,grasp_true)
    return max_iou

def iou(grasp_pre,grasp_true):
    '''
    :功能 :计算两个给定框的iou
    :参数 : grasp_pre :Grasp对象，单个预测结果中反求出的抓取框
    :参数 : grasp_true:Grasp对象，单个真实标注抓取框
    :返回 : 两者的iou
    '''
    #先提取出两个框的所覆盖区域
    rr1, cc1 = grasp_pre.polygon_coords()#现在是中心点和角度定义的抓取，要转换成四个角点定义的抓取才方便操作
    rr2, cc2 = polygon(grasp_true.points[:, 0], grasp_true.points[:, 1])
    
    #读取两个框的极限位置
    r_max = max(rr1.max(),rr2.max())+1
    c_max = max(cc1.max(),cc2.max())+1
    
    #根据最大的边界来确定蒙版画布大小
    canvas = np.zeros((r_max,c_max))
    canvas[rr1,cc1] += 1
    canvas[rr2,cc2] += 1
    
    union = np.sum(canvas > 0)
    
    if union == 0:
        return 0
    
    intersection = np.sum(canvas == 2)
    print(intersection/union)
    return intersection/union
    