#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 16:40:50 2020

@author: ldh
"""

#导入第三方包
import torch
import torch.optim as optim
from torchsummary import summary
import numpy as np
import cv2
import time
import datetime
import os
import sys
import logging
#导入自定义包
from ggcnn2_new import GGCNN2
from cornell_pro import Cornell
from functions import post_process,detect_grasps,max_iou
from image_pro import Image

#一些训练参数的设定
batch_size = 16
batches_per_epoch = 1000
epochs = 20
lr = 0.001
num_workers = 6
val_batches = 250

split = 0.9
use_rgb = True
use_depth = False
random_rotate = True
random_zoom = True

logging.basicConfig(level=logging.INFO)

def train(epoch,net,device,train_data,optimizer,batches_per_epoch):
    """
    :功能: 执行一个训练epoch
    :参数: epoch             : 当前所在的   epoch数
    :参数: net               : 要使用的网络模型
    :参数: device            : 训练使用的设备
    :参数: train_data        : 训练所用数据集
    :参数: optimizer         : 优化器
    :参数: batches_per_epoch : 每个epoch训练中所用的数据批数

    :返回: 本epoch的平均损失
    """
    #结果字典，最后返回用
    results = { 
        'loss': 0,
        'losses': {
        }
    }
    
    #训练模式，所有的层和参数都会考虑进来，eval模式下比如dropout这种层会不使能
    net.train()
    
    batch_idx = 0
    
    #开始样本训练迭代
    while batch_idx < batches_per_epoch:
        #print('唉，又来重新载入数据了')
        #print('time1:',time.time())
        for x, y, _,_,_ in train_data:#这边就已经读了len(dataset)/batch_size个batch出来了，所以最终一个epoch里面训练过的batch数量是len(dataset)/batch_size*batch_per_epoch个，不，你错了，有个batch_idx来控制的，一个epoch中参与训练的batch就是batch_per_epoch个
            batch_idx += 1
            #print('time2:',time.time())
            if batch_idx >= batches_per_epoch:
                break
            #将数据传到GPU
            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            
            lossdict = net.compute_loss(xc,yc)
            
            #获取当前损失
            loss = lossdict['loss']
            
            #打印一下训练过程
            if batch_idx % 100 == 0:
                logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))
            
            #记录总共的损失
            results['loss'] += loss.item()
            #单独记录各项损失，pos,cos,sin,width
            for ln, l in lossdict['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()
            
            #反向传播优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('time3:',time.time())

    #计算总共的平均损失
    results['loss'] /= batch_idx
        
    #计算各项的平均损失
    for l in results['losses']:
        results['losses'][l] /= batch_idx
    return results

def validate(net,device,val_data,batches_per_epoch,vis = False):
    """
    :功能: 在给定的验证集上验证给定模型的是被准确率并返回
    :参数: net              : torch.model,要验证的网络模型
    :参数: device           : 验证使用的设备
    :参数: val_data         : dataloader,验证所用数据集
    :参数: batches_per_epoch: int,一次验证中用多少个batch的数据，也就是说，要不要每次validate都在整个数据集上迭代，还是只在指定数量的batch上迭代
    :参数: vis              : bool,validate时是否可视化输出
    
    :返回: 模型在给定验证集上的正确率
    """
    val_result = {
        'correct':0,
        'failed':0,
        'loss':0,
        'losses':{}
        }
    #设置网络进入验证模式
    net.eval()
    
    length = len(val_data)
    
    with torch.no_grad():
        batch_idx = 0
        while batch_idx < (batches_per_epoch-1):
            for x,y,idx,rot,zoom_factor in val_data:
                batch_idx += 1
                
                xc = x.to(device)
                yc = [yy.to(device) for yy in y]
                
                lossdict = net.compute_loss(xc, yc)
                 
                loss = lossdict['loss']
                
                val_result['loss'] += loss.item()/length
                
                q_out,ang_out,width_out = post_process(lossdict['pred']['pos'], lossdict['pred']['cos'], 
                                                       lossdict['pred']['sin'], lossdict['pred']['width'])
                grasps_pre = detect_grasps(q_out,ang_out,width_out,no_grasp = 1)
                grasps_true = val_data.dataset.get_raw_grasps(idx,rot,zoom_factor)
                
                result = 0
                for grasp_pre in grasps_pre:
                    if max_iou(grasp_pre,grasps_true) > 0.25:
                        result = 1
                        break
                
                if result:
                    val_result['correct'] += 1
                else:
                    val_result['failed'] += 1
        
        if vis:
            #前面几个迭代过程没有有效的grasp_pre提取出来，所以，len是0，所以，不会有可视化结果显示出来
            if len(grasps_pre)>0:
                visualization(val_data,idx,grasps_pre,grasps_true)
            
        #print('acc:{}'.format(val_result['correct']/(batches_per_epoch*batch_size)))绝对的计算方法不清楚总数是多少，那就用相对的方法吧
        acc = val_result['correct']/(val_result['correct']+val_result['failed'])
        logging.info('acc:{}'.format(acc))
    return(acc)

def visualization(val_data,idx,grasps_pre,grasps_true):
    #最开始的几个迭代过程中不会有q_img值足够大的预测出现，所以，此时提出不出有效的抓取，进而是不会有visuaization出现的
    img = Image.from_file(val_data.dataset.rgbf[idx])
    left = val_data.dataset._get_crop_attrs(idx)[1]
    top = val_data.dataset._get_crop_attrs(idx)[2]
    img.crop((left,top),(left+300,top+300))
    
    a = img.img
    
    a_points = grasps_pre[0].as_gr.points.astype(np.uint8)#预测出的抓取
    b_points = grasps_true.points
    color1 = (255,255,0)
    color2 = (255,0,0)
    for i in range(3):
        img = cv2.line(a,tuple(a_points[i]),tuple(a_points[i+1]),color1 if i % 2 == 0 else color2,1)
    img = cv2.line(a,tuple(a_points[3]),tuple(a_points[0]),color2,1)
    
    color1 = (0,0,0)
    color2 = (0,255,0)
    
    for b_point in b_points:
        for i in range(3):
            img = cv2.line(a,tuple(b_point[i]),tuple(b_point[i+1]),color1 if i % 2 == 0 else color2,1)
        img = cv2.line(a,tuple(b_point[3]),tuple(b_point[0]),color2,1)
    #cv2.imshow('img',a)
    cv2.imwrite('img.png',a)
    #cv2.waitKey(1000)

def run():
    #设置输出文件夹
    out_dir = 'trained_models/'
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    
    save_folder = os.path.join(out_dir, dt)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    #获取设备
    device = torch.device("cuda:0")
    max_acc = 0.1

    #实例化一个网络
    input_channels = 1*use_depth+3*use_rgb
    net = GGCNN2(input_channels)
    net = net.to(device)
    
    #保存网络和训练参数信息
    summary(net,(input_channels,300,300))
    f = open(os.path.join(save_folder,'arch.txt'),'w')
    sys.stdout = f
    summary(net,(input_channels,300,300))
    sys.stdout = sys.__stdout__
    f.close()
    sys.stdout.write('start run...\n')
    with open(os.path.join(save_folder,'params.txt'),'w') as f:
        f.write('batch_size:{}\nbatches_per_epoch:{}\nepochs:{}\nlr:{}'.format(batch_size,batches_per_epoch,epochs,lr))
    #准备数据集
    #训练集
    train_data = Cornell('./cornell',include_rgb = use_rgb,include_depth = use_depth,start = 0.0,end = split,random_rotate = random_rotate,random_zoom = random_zoom,output_size=300)
    train_dataset = torch.utils.data.DataLoader(train_data,batch_size = batch_size,shuffle = True,num_workers = num_workers)
    #验证集
    val_data = Cornell('./cornell',include_rgb = use_rgb,include_depth = use_depth,start = split,end = 1.0,random_rotate = random_rotate,random_zoom = random_zoom,output_size = 300)
    val_dataset = torch.utils.data.DataLoader(val_data,batch_size = 1,shuffle = True,num_workers = num_workers)

    #设置优化器
    optimizer = optim.Adam(net.parameters(),lr = lr)
    
    #开始主循环
    for epoch in range(epochs):
        train_results = train(epoch, net, device, train_dataset, optimizer, batches_per_epoch = batches_per_epoch)
        logging.info('validating...')
        validate_results = validate(net,device,val_dataset,val_batches,vis = True)
        #logging.info('{0}/model{1}_epoch{2}_batch_{3}'.format(save_folder,str(validate_results)[0:5],epoch,batch_size))
        if validate_results > max_acc:
            max_acc = validate_results
            torch.save(net.state_dict(),'{0}/model{1}_epoch{2}_batch_{3}.pth'.format(save_folder,str(validate_results)[0:5],epoch,batch_size))
    return train_results,validate_results

if __name__ == '__main__':
    run()