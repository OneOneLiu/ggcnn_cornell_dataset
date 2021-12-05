import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
from skimage.filters import gaussian

from utils.functions import post_process,detect_grasps,max_iou
from utils.grasp_pro import Grasps,Grasp

def show_grasp(img,grasp,color = (0,255,0),title = 'Show Grasp',dis = True):
    # 给定一张图及一个抓取(Grasps和Grasp类型均可),在图上绘制出抓取框
    if not isinstance(grasp,Grasps):
        grasp = Grasps([grasp])
    for gr in grasp.grs:
        # 有时候预测出负值会报错
        try:
            for i in range(3):
                cv2.line(img,tuple(gr.points.astype(np.uint32)[i][::-1]),tuple(gr.points.astype(np.uint32)[i+1][::-1]),color,1)
            cv2.line(img,tuple(gr.points.astype(np.uint32)[3][::-1]),tuple(gr.points.astype(np.uint32)[0][::-1]),color,1)
            cv2.circle(img, (gr.center[1],gr.center[0]), 2, color, 1)
        except:
            pass
    if dis:
        plt.figure(figsize=(5,5))
        plt.title(title)
        plt.imshow(img)
        plt.show()

    return img

def show_img(dataset,idx,rot = 0.0,zoom = 1.0):
    # display the original img
    plt.figure(figsize=(10,10))
    rgb_img = dataset.get_rgb(idx,rot,zoom,normalize = False)
    depth_img = dataset.get_depth(idx,rot,zoom)
    plt.subplot(121)
    plt.imshow(rgb_img)
    plt.subplot(122)
    plt.imshow(depth_img)
    plt.show()
    
def show_label_img(dataset,idx,rot,zoom):
    # 展示指定编号的标注映射图
    pos_img,ang_img,width_img = dataset.get_grasp(idx,rot,zoom)
    plt.figure(figsize = (10,10))
    plt.subplot(131)
    plt.title('Positon Image')
    plt.imshow(pos_img,cmap = 'gray')
    plt.subplot(132)
    plt.title('Angle Image')
    plt.imshow(ang_img,cmap = 'gray')
    plt.subplot(133)
    plt.title('Width Image')
    plt.imshow(width_img,cmap = 'gray')
    plt.show()
    
def show_input(dataset,idx,rot,zoom,dis = True):
    # display the original img
    plt.figure(figsize=(10,10))
    color = (0,255,0)
    rgb_img = dataset.get_rgb(idx,rot,zoom,normalize = False)
    depth_img = dataset.get_depth(idx,rot,zoom)
    grasp_true = dataset.get_raw_grasps(idx,rot,zoom)
    plt.subplot(131)
    plt.title('RGB Image')
    plt.imshow(rgb_img)
    plt.subplot(132)
    plt.title('Depth Image')
    plt.imshow(depth_img)
    plt.subplot(133)
    plt.title('Labels')
    for gr in grasp_true.grs:
        try:
            for i in range(3):
                cv2.line(rgb_img,tuple(gr.points.astype(np.uint32)[i][::-1]),tuple(gr.points.astype(np.uint32)[i+1][::-1]),color,1)
            cv2.line(rgb_img,tuple(gr.points.astype(np.uint32)[3][::-1]),tuple(gr.points.astype(np.uint32)[0][::-1]),color,1)
        except:
            #缩放略大,有些原始框超出边界了,无法画出
            pass
    if dis:
    	plt.imshow(rgb_img)
    	plt.show()
def show_result(dataset,idx,rot,zoom,grasp_pre,save = False,name = None,dis = True,title_p = 'Predicted Grasp'):
    # 给定数据idx和增强参数,比较预测结果和真实结果,并可视化
    img = dataset.get_rgb(idx,float(rot),float(zoom),normalize = False)
    grasp_true = dataset.get_raw_grasps(idx,rot,zoom)
    
    # 检查结果是否正确
    result = check_result(grasp_pre,grasp_true)
    if dis:
        if result:
            print('The prediction is Positive')
        else:
            print('The prediction is Negative')
    show_input(dataset,idx,rot,zoom,dis = dis)
    if len(grasp_pre):
        for grasp in grasp_pre:
            img = show_grasp(img,grasp.as_gr,color = (255,0,0),title = title_p,dis = dis)
    else:
        print('未检测到合适抓取')

    if save:
    	cv2.imwrite(name+'.png',img[::,::,::-1])
def show_output_img(net,x):
    pos_img,cos_img,sin_img,width_img= net(x)
    q_img = pos_img.cpu().data.numpy().squeeze()
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().data.numpy().squeeze()
    width_img = width_img.cpu().data.numpy().squeeze() * 150.0

    q_img_g = gaussian(q_img, 2.0, preserve_range=True)
    ang_img_g = gaussian(ang_img, 2.0, preserve_range=True)
    width_img_g = gaussian(width_img, 1.0, preserve_range=True)
    plt.figure(figsize = (15,15))
    plt.subplot(131)
    plt.imshow(q_img_g,cmap = 'gray')
    plt.subplot(132)
    plt.imshow(ang_img_g,cmap = 'gray')
    plt.subplot(133)
    plt.imshow(width_img_g,cmap = 'gray')
    plt.show()
def show_gen_img(pos_img,ang_img,width_img):
    # 可视化展示生成的三个映射图
    plt.figure(figsize = (10,10))
    plt.subplot(131)
    plt.title('Positon Image')
    plt.imshow(pos_img,cmap = 'gray')
    plt.subplot(132)
    plt.title('Angle Image')
    plt.imshow(ang_img,cmap = 'gray')
    plt.subplot(133)
    plt.title('Width Image')
    plt.imshow(width_img,cmap = 'gray')
    plt.show()
def visual():
    plt.clf()
    files = os.listdir()
    idxs = [name.split('_')[0] for name in files]
    idxs.sort()

    idx_set = set(idxs)
    idx_array = []

    for i in range(89):
        if str(i) in list(idx_set):
            idx_array.append(idxs.count(str(i)))
        else:
            idx_array.append(0)
    x = np.linspace(0,89,89)
    plt.bar(x,idx_array,width=0.4)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig('0_visual.png',dpi=600)
    #plt.show()

def train(epoch,net,device,train_data,optimizer,batches_per_epoch,weighted_loss):
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
            if batch_idx % 10 == 0:
                logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))
            
            #记录总共的损失
            results['loss'] += loss.item()
            #单独记录各项损失，pos,cos,sin,width
            for ln, l in lossdict['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()
            
            # 为不同loss加权
            p_loss     = lossdict['losses']['p_loss']
            cos_loss   = lossdict['losses']['cos_loss']
            sin_loss   = lossdict['losses']['sin_loss']
            width_loss = lossdict['losses']['width_loss']
            
            loss = weighted_loss[0]*p_loss+weighted_loss[1]*cos_loss+weighted_loss[2]*sin_loss+weighted_loss[3]*width_loss
            
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
        'losses':{},
        'failed_info':{}
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
                # 记录各项的单独损失
                for ln, l in lossdict['losses'].items():
                    if ln not in val_result['losses']:
                        val_result['losses'][ln] = 0
                    val_result['losses'][ln] += l.item()/length

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
                    val_result['failed_info'][str(batch_idx)] = [idx.numpy()[0],rot.numpy()[0],zoom_factor.numpy()[0]] 
        
        #print('acc:{}'.format(val_result['correct']/(batches_per_epoch*batch_size)))绝对的计算方法不清楚总数是多少，那就用相对的方法吧
        acc = val_result['correct']/(val_result['correct']+val_result['failed'])
        logging.info('{}/{} acc:{}'.format(val_result['correct'],batches_per_epoch,acc))
        val_result['acc'] = acc
    return(val_result)
