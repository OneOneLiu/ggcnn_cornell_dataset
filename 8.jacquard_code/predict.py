# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:18:10 2020

@author: LiuDahui
"""


import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

from cornell_pro import Cornell
from jacquard import Jacquard
from functions import post_process,detect_grasps,max_iou

batch_size = 1

#建立数据集
# dataset = Jacquard('../jacquard',random_rotate = True)
dataset = Cornell('../cornell',random_rotate = True,random_zoom = True)
pre_dataset = torch.utils.data.DataLoader(dataset,batch_size = batch_size,shuffle = True)

for x,y,idx,rot,zoom in pre_dataset:
    pre_x = x
    pre_y = y
    break

#载入训练好的网络
device = torch.device("cuda:0")
pre_x = pre_x.to(device)
net = torch.load('trained_models/201108_0954/model0.489_epoch90_batch_2')
pos_img,cos_img,sin_img,width_img= net(pre_x)

q_out,ang_out,wid_out = post_process(pos_img,cos_img,sin_img,width_img)

grasps_pre = detect_grasps(q_out,ang_out,wid_out,no_grasp = 1)
grasps_true = pre_dataset.dataset.get_raw_grasps(idx,rot,zoom)

result = 0
for grasp_pre in grasps_pre:
    if max_iou(grasp_pre,grasps_true) > 0.25:
        result = 1
        break
    
img = dataset.get_rgb(int(idx[0]),float(rot),float(zoom),normalize = False)

# for gr in grasps_true.grs:
#     for i in range(3):
#         cv2.line(img,tuple(gr.points.astype(np.uint32)[i]),tuple(gr.points.astype(np.uint32)[i+1]),(255,255,0),3)
#     cv2.line(img,tuple(gr.points.astype(np.uint32)[3]),tuple(gr.points.astype(np.uint32)[0]),(255,255,0),3)

gr = grasps_pre[0].as_gr
for i in range(3):
    cv2.line(img,tuple(gr.points.astype(np.uint32)[i]),tuple(gr.points.astype(np.uint32)[i+1]),(255,0,0),1)
cv2.line(img,tuple(gr.points.astype(np.uint32)[3]),tuple(gr.points.astype(np.uint32)[0]),(255,0,0),1)
plt.imshow(img)
plt.show()
