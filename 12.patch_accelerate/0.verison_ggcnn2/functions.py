# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 21:52:23 2020
validate部分用到的一些函数汇总
@author: LiuDahui
"""
import torch
import torchvision.transforms.functional as VisionF

import cv2
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.draw import polygon
import numpy as np


from grasp_pro import Grasp_cpaw,Grasps
from image_pro import Image
import matplotlib.pyplot as plt

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
    width_img = width_img.cpu().data.numpy().squeeze() * 150.0
    
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
    local_max = peak_local_max(q_out, min_distance=20, threshold_abs=0.2,num_peaks = no_grasp)
    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)
        grasp_angle = ang_out[grasp_point]

        g = Grasp_cpaw(grasp_point,grasp_angle)
        if wid_out is not None:
            g.width = wid_out[grasp_point]
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
        Iou = iou(grasp_pre,grasp_true)
        max_iou = max(max_iou,Iou)
    return max_iou

def iou(grasp_pre,grasp_true,angle_threshold = np.pi/6):
    '''
    :功能 :计算两个给定框的iou
    :参数 : grasp_pre      :Grasp对象，单个预测结果中反求出的抓取框
    :参数 : grasp_true     :Grasp对象，单个真实标注抓取框
    :参数 : angle_threshold:角度阈值，超过这个角度就认为两者不符
    :返回 : 两者的iou
    '''
    #超过这个角度阈值就认为这两者不符，下面的计算复杂是为了消除角度方向的影响
    if abs((grasp_pre.angle - grasp_true.angle + np.pi/2) % np.pi - np.pi/2) > angle_threshold:
        return 0
    #先提取出两个框的所覆盖区域
    rr1, cc1 = grasp_pre.polygon_coords()
    rr2, cc2 = polygon(grasp_true.points[:, 0], grasp_true.points[:, 1])
    try:#有时候这边返回的rr2是空的，再运行下面的就会报错，在这加个故障处理确保正常运行
        r_max = max(rr1.max(), rr2.max()) + 1
        c_max = max(cc1.max(), cc2.max()) + 1
    except:
        return 0

    #根据最大的边界来确定蒙版画布大小
    canvas = np.zeros((r_max,c_max))
    canvas[rr1,cc1] += 1
    canvas[rr2,cc2] += 1

    union = np.sum(canvas > 0)
    
    if union == 0:
        return 0

    intersection = np.sum(canvas == 2)
    #print(intersection/union)
    return intersection/union
    
# 下面的是修正参数所需要的函数
import copy

def show_grasp(img,grasp,color = (0,255,0)):
    # 给定一张图及一个抓取(Grasps和Grasp类型均可),在图上绘制出抓取框
    if not isinstance(grasp,Grasps):
        grasp = Grasps([grasp])
    for gr in grasp.grs:
        # 有时候预测出负值会报错
        try:
            for i in range(3):
                cv2.line(img,tuple(gr.points.astype(np.uint32)[i][::-1]),tuple(gr.points.astype(np.uint32)[i+1][::-1]),color,1)
            cv2.line(img,tuple(gr.points.astype(np.uint32)[3][::-1]),tuple(gr.points.astype(np.uint32)[0][::-1]),color,1)
        except:
            pass
    return img

def get_edge(resized_img):
    # 中心高度提取
    img_middle = resized_img[:,20:80]
    # 在这计算height_m,主要就是取较高的值
    hm = np.max(img_middle,axis = 0)
    mean_height = img_middle.mean()
    new_hm = hm[np.where(hm >= mean_height)]
    mean_hm = new_hm.mean()
    height_m = new_hm[np.where(new_hm >= mean_hm)].mean()
    if height_m == 0: # 如果中间没有,那么就放大框,暂时也没别的办法
        return 0,0,0,0,0

    # 边缘宽度检测
    # 下面进行碰撞检测边缘提取
    h1 = np.max(resized_img,axis = 0)
    threshold = height_m - 10
    if height_m < 20:
        threshold = height_m // 2
    # 获取各个边缘的宽度
    # 1.左边缘
    edge1 = np.where(h1[0:40] > threshold)
    if len(edge1[0]) == 0:
        edge_left = 40
    else:
        edge_left = edge1[0][0]
    # 2.右边缘
    edge2 = np.where(h1[60:100] > threshold)
    if len(edge2[0]) == 0:
        edge_right = 40
    else:
        edge_right = 40-edge2[0][-1]-1

    h2 = np.max(resized_img,axis = 1)
    # 3.上边缘 调整位置用
    edge3 = np.where(h2[0:20] > threshold)
    if len(edge3[0]) == 0:
        edge_top = 20
    else:
        edge_top = edge3[0][0]
    # 4.下边缘 调整位置用
    edge4 = np.where(h2[30:50] > threshold)
    if len(edge4[0]) == 0:
        edge_bottom = 20
    else:
        edge_bottom = 20-edge4[0][-1]-1
    edge = min(edge_left,edge_right)
    
    return edge,edge_left,edge_right,edge_top,edge_bottom

def collision_validate(gr,mask_height):
    y = gr.center[0]
    x = gr.center[1]
    angle = gr.angle
    width = gr.width
    length = gr.length

    top = int(y - length / 2)
    left = int(x - width / 2)
    rt_angle = -float((angle / np.pi *180))

    rectified_img = VisionF.rotate(img = mask_height.view(1,1,300,300),angle = rt_angle,center = (x,y))

    crop_img = VisionF.crop(rectified_img,top,left,int(length),int(width))

    resized_img = VisionF.resize(crop_img,[50,100]).squeeze().cpu().data.numpy()
    
    # plt.imshow(resized_img)
    # plt.show()
    # 获取图像各边缘宽度
    edge,_,_,_,_= get_edge(resized_img)

    return edge
