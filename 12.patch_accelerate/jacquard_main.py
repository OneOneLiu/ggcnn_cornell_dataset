# -*- coding: utf-8 -*-
"""
Created on Thur July 15 14:55:01 2021


@author: LiuDaohui
"""

# 导入第三方包
import torch
import torch.optim as optim
from torchsummary import summary
import torchvision.transforms.functional as VisionF
from torch.utils.tensorboard import SummaryWriter
from imageio import imsave

import time
import datetime
import os
# summary输出保存之后print函数失灵，但还可以用这个logging来打印输出
import logging
import sys
import numpy as np
# 导入自定义包
from jacquard import Jacquard
from ggcnn2 import GGCNN2
from functions import post_process, detect_grasps, max_iou, show_grasp, collision_check
import matplotlib.pyplot as plt
# 一些训练参数的设定
batch_size = 8
epochs = 30
c_epochs = 40
batches_per_epoch = 100
val_batches = 250
lr = 0.001

use_depth = True
use_rgb = False
r_rotate = True
r_zoom = True

split = 0.9
num_workers = 6

pretrain = True

dataset = 'ADJ'

pretrain_net_path = 'Prob/210902_0903/model0.939_epoch69_batch_8.pth'

fronzen = True

def train(epoch, net, device, train_data, optimizer, batches_per_epoch):
    """
    :功能: 执行一个训练epoch
    :参数: epoch             : 当前所在的epoch数
    :参数: net               : 要使用的网络模型
    :参数: device            : 训练使用的设备
    :参数: train_data        : 训练所用数据集
    :参数: optimizer         : 优化器
    :参数: batches_per_epoch : 每个epoch训练中所用的数据批数

    :返回: 本epoch的平均损失
    """
    # 结果字典，最后返回用
    results = {
        'loss': 0,
        'losses': {
        }
    }

    # 训练模式，所有的层和参数都会考虑进来，eval模式下比如dropout这种层会不使能
    net.train()

    batch_idx = 0

    # 开始样本训练迭代
    while batch_idx < batches_per_epoch:
        # 这边就已经读了len(dataset)/batch_size个batch出来了，所以最终一个epoch里面训练过的batch数量是len(dataset)/batch_size*batch_per_epoch个，不，你错了，有个batch_idx来控制的，一个epoch中参与训练的batch就是batch_per_epoch个
        for x, y, _, _, _, target in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break

            # 将数据传到GPU
            xc = x.to(device)
            yc = [yy.to(device) for yy in y]

            batch_inputs = (xc, yc, target)

            lossdict = net.compute_loss(batch_inputs)

            # 获取当前损失
            loss = lossdict['loss']

            # 打印一下训练过程
            if batch_idx % 10 == 0:
                logging.info(time.ctime())
                logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(
                    epoch, batch_idx, loss.item()))

            # 记录总共的损失
            results['loss'] += loss.item()
            # 单独记录各项损失，pos,cos,sin,width
            for ln, l in lossdict['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()
            # 反向传播优化模型

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 计算总共的平均损失
    results['loss'] /= batch_idx

    # 计算各项的平均损失
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results


def validate(net, device, val_data, batches_per_epoch):
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
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {},
        'acc': 0.0,
        'positive': 0,
        'negative':0
    }
    # 设置网络进入验证模式
    net.eval()

    length = len(val_data)

    with torch.no_grad():
        batch_idx = 0
        while batch_idx < (batches_per_epoch):
            for x, y, idx, rot, zoom_factor, target in val_data:
                batch_idx += 1
                if batch_idx >= batches_per_epoch:
                    break
                xc = x.to(device)
                yc = [yy.to(device) for yy in y]

                batch_inputs = (xc, yc, target)

                lossdict = net.compute_loss(batch_inputs)

                loss = lossdict['loss']

                val_result['loss'] += loss.item()/length

                # 记录各项的单独损失
                for ln, l in lossdict['losses'].items():
                    if ln not in val_result['losses']:
                        val_result['losses'][ln] = 0
                    val_result['losses'][ln] += l.item()/length

                q_out, ang_out, width_out = post_process(lossdict['pred']['pos'], lossdict['pred']['cos'],lossdict['pred']['sin'], lossdict['pred']['width'])
                grasp_pres = detect_grasps(q_out, ang_out, width_out)
                grasps_true = val_data.dataset.get_raw_grasps(idx, rot, zoom_factor)

                result = 0
                for grasp_pre in grasp_pres:
                    if max_iou(grasp_pre, grasps_true) > 0.25:
                        result = 1
                        break

                if result:
                    val_result['correct'] += 1
                else:
                    val_result['failed'] += 1

                # 进行碰撞检查来判定是否真的是成功的
                rgb = val_data.dataset.get_rgb(idx, rot, zoom_factor, normalize = False)
                img_name = '11.add_POTO/output/patch/f_{0}_{1}_{2}.png'.format(idx.cpu().data.numpy()[0],rot.cpu().data.numpy()[0],zoom_factor.cpu().data.numpy()[0])
                flag = collision_validate(grasp_pres,batch_inputs[2][3],rgb,img_name)
                if flag:
                    val_result['negative'] += 1
                else:
                    val_result['positive'] += 1
        logging.info(time.ctime())
        acc = val_result['correct'] / (val_result['correct']+val_result['failed'])
        logging.info('iou_acc:{}'.format(acc))
        val_result['acc'] = acc
        true_acc = val_result['positive'] / (val_result['positive']+val_result['negative'])
        logging.info('true acc:{}'.format(true_acc))
        val_result['true acc'] = true_acc

    return(val_result)
def collision_validate(grasp_pres,mask_height,rgb,img_name):
    # 1代表有碰撞
    flag = 1
    for grasp_pre in grasp_pres:
        gr = grasp_pre.as_gr
        img = show_grasp(rgb,gr,255)
        # 如果从高度图上看这里的高度为0的话,就跳过了,肯定不行的.
        flag = collision_check(mask_height,(gr.center,gr.angle,gr.width,gr.length))
        imsave(img_name.replace('f',str(flag)), img)
        if not flag:
            break
    return flag
def run():
    # 设置输出文件夹
    home_dir = '11.add_POTO/trained_models'
    out_dir = '11.add_POTO/trained_models/Patch'
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_tb'.format(dt)  

    save_folder = os.path.join(out_dir, dt)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    logging.basicConfig(filename = os.path.join(save_folder,'logger.log'),level=logging.INFO)

    logging.info('\nVersion: Train Prob\nModel: GGCNN2 + filter\nValidate: IOU\nQuality map: position img\nPretrain: {}\nInfo: This version start to train the filter layer by add a prob loss. And you can choose whether to use or not to use the return of "get ground truth" function "prob" as the validate qulity map.And this version of code is tranfered from the model trained on the original ggcnn.\nNOTE:prob is supervised under the file **prob.png'.format(str(pretrain)))
    
    logging.info('\nbatch_size:{0}\nlr:{1}\nuse_depth:{2}\nuse_rgb:{3}\nr_rotate:{4}\nr_zoom:{5}\ndataset:{6}\npretrain_net:{7}'.format(batch_size,lr,use_depth,use_rgb,r_rotate,r_zoom,dataset,pretrain_net_path))

    device = torch.device("cuda:0")

    # 实例化一个网络
    input_channels = 1 * use_depth + 3 * use_rgb
    # net = GGCNN2(input_channels)
    # net = net.to(device)

    net = GGCNN2(input_channels)
    # net.load_state_dict(torch.load(os.path.join(out_dir,'210814_1917/model0.959_epoch95_batch_8.pth')))
    # 
    # net.load_state_dict(torch.load(os.path.join(out_dir,'210819_0935/model0.955_epoch73_batch_8.pth')))
    # 这个是加了prob的
    net.load_state_dict(torch.load(os.path.join(home_dir,pretrain_net_path)))
    # 这个是加了poto的
    # net.load_state_dict(torch.load(os.path.join(out_dir,'210825_1745/model0.939_epoch1_batch_8.pth')))
    net = net.to(device)
    # 准备数据集
    # 训练集
    if dataset == 'ADJ':
        train_data = Jacquard('./jacquard', include_rgb=use_rgb, include_depth=use_depth,start=0.0,end=split, random_rotate=r_rotate, random_zoom=r_zoom, output_size=300,load_from_npy = True, npy_path = 'train_ADJ.npy')
        train_dataset = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        # 验证集
        val_data = Jacquard('./jacquard', include_rgb=use_rgb, include_depth=use_depth, start=split,end=1.0, random_rotate=r_rotate, random_zoom=r_zoom, output_size=300,load_from_npy = True, npy_path = 'test_ADJ.npy')
        val_dataset = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=num_workers)
    else:
        train_data = Jacquard('./jacquard', include_rgb=use_rgb, include_depth=use_depth,start=0.0,end=split, random_rotate=r_rotate, random_zoom=r_zoom, output_size=300)
        train_dataset = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        # 验证集
        val_data = Jacquard('./jacquard', include_rgb=use_rgb, include_depth=use_depth, start=split,end=1.0, random_rotate=r_rotate, random_zoom=r_zoom, output_size=300)
        val_dataset = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=num_workers)

    # 设置优化器
    optimizer = optim.Adam(net.parameters())
    logging.info('Start training')
    # 载入模型并训练一定的epoch
    for i in range(5):
        validate_results = validate(net, device, val_dataset, batches_per_epoch=val_batches)
        
    for epoch in range(100):
        # 先在没加patch loss的网络上验证5次
        train_results = train(epoch, net, device,train_dataset, optimizer, batches_per_epoch)

        logging.info('Validating....')
        validate_results = validate(
            net, device, val_dataset, batches_per_epoch=val_batches)
        # print('正确正确')
        # print(validate_results['true_positive']/validate_results['correct'])
        # print('正确错误')
        # print(validate_results['true_negative']/validate_results['failed'])
        # print('正确精度')
        # print((validate_results['true_positive']+validate_results['false_negative'])/(validate_results['failed']+validate_results['correct']))
        # if validate_results['acc'] > max_acc:
        #     max_acc = validate_results['acc']
        torch.save(net.state_dict(), '{0}/model{1}_epoch{2}_batch_{3}.pth'.format(
            save_folder, str(validate_results['acc'])[0:5], epoch, batch_size))
        # logging.info('checking...')
        # check_false_positive(net, device, val_dataset, batches_per_epoch=val_batches)

if __name__ == '__main__':
    run()
