# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 21:29:59 2020

验证函数一次预测是否正确的函数

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
net = torch.load('trained_models/model1')

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


#调试   
a = np.zeros([300,300])
a_points = grasps_pre[0].as_gr.points.astype(np.uint8)#预测出的抓取
b_points = grasps_true.points[3]#真实的抓取，可以先测试一下跟哪个标注框的重合度较高，然后拿好的那个来计算

a = cv2.rectangle(a,tuple(a_points[0]),tuple(a_points[2]),2)
plt.subplot(211)
plt.imshow(a)

cv2.rectangle(a,tuple(b_points[0]),tuple(b_points[2]),2)

plt.subplot(212)
plt.imshow(a)
plt.show()
plt.imsave('a.png',a)