# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 21:29:59 2020

验证函数一次预测是否正确的函数，这里仅仅是示例，一个测试而已，还没有将它集成到一个完整的训练过程中去，集成到训练过程中去的程序见第二部分

@author: LiuDahui
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
#自定义包
from validate.cornell_pro import Cornell
from validate.functions import post_process,detect_grasps,max_iou

#要验证网络性能，就必须要有一个训练完的网络，这里不从头训练，直接载入一个训练好的模型
net = torch.load('trained_models/model')

#设置训练设备
device = torch.device('cuda:0')

#构建数据集并获取一组数据
dataset = Cornell('cornell')
val_data = torch.utils.data.DataLoader(dataset,shuffle = True,batch_size = 1)

#读出一组数据及其编号
for x,y,id_x in val_data:
    xc = x.to(device)
    yc = [yy.to(device) for yy in y]
    idx = id_x
    print(idx)
    break
    
#输入网络计算预测结果
net = net.to(device)
pos_img,cos_img,sin_img,width_img = net(xc)

#原始输出的预处理
q_out, angle_out, width_out = post_process(pos_img,cos_img,sin_img,width_img)

#结果中检测有效的抓取,注意，这边返回来的是个列表，里面每个对象都是Grasp_cpaw
grasps_pre = detect_grasps(q_out,angle_out,width_out,no_grasp = 1)

#按照idx读入真实标注,注意，这边返回来的是Grasps对象，也就是由多个角点定义的一个对象的所有正确抓取
grasps_true = val_data.dataset.get_raw_grasps(idx)

for grasp_pre in grasps_pre:
    if max_iou(grasp_pre,grasps_true) > 0.25:
        print('true')


#调试，可视化看一下预测出的结果和真实的标注结果，单看指标可不能说明一个预测的抓取是否真的有效
from validate.image_pro import Image

img = Image.from_file(val_data.dataset.rgbf[idx])
left = val_data.dataset._get_crop_attrs(idx)[1]
top = val_data.dataset._get_crop_attrs(idx)[2]
img.crop((left,top),(left+300,top+300))

a = img.img
a_points = grasps_pre[0].as_gr.points.astype(np.uint8)#预测出的抓取
b_points = grasps_true.points

color1 = (255,255,0)
color2 = (255,0,0)
#可以不注释这一句看看直接用rectangle会出来多么不好的结果，这个失误耽误了我4个小时的时间
#a = cv2.rectangle(a,tuple(a_points[0]),tuple(a_points[2]),2)
for i in range(3):
    img = cv2.line(a,tuple(a_points[i]),tuple(a_points[i+1]),color1 if i % 2 == 0 else color2,1)
img = cv2.line(a,tuple(a_points[3]),tuple(a_points[0]),color2,1)
plt.subplot(121)
plt.imshow(a)

color1 = (0,0,0)
color2 = (0,255,0)
for b_point in b_points:
    for i in range(3):
        img = cv2.line(a,tuple(b_point[i]),tuple(b_point[i+1]),color1 if i % 2 == 0 else color2,1)
    img = cv2.line(a,tuple(b_point[3]),tuple(b_point[0]),color2,1)
plt.subplot(122)
plt.imshow(a)
plt.show()