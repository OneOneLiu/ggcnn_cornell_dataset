# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 21:29:59 2020


@author: LiuDaohui
"""

# 导入第三方包
import torch
import torch.optim as optim
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import datetime
import os
# summary输出保存之后print函数失灵，但还可以用这个logging来打印输出
import logging
import sys
# 导入自定义包
from jacquard import Jacquard
from eenet import EENET

# 一些训练参数的设定
batch_size = 8
epochs = 300
batches_per_epoch = 100
val_batches = 250
lr = 0.001

use_rgb = False
use_depth = True
r_rotate = True
r_zoom = True

split = 0.9
num_workers = 6

logging.basicConfig(level=logging.INFO)
# 这部分是直接copy的train_main2.py


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
        for x, _, idx, _, _, y in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break
            
            
            # 将数据传到GPU
            xc = x.to(device)
            yc = y.to(device)

            lossdict = net.compute_loss(xc, yc)
            if batch_idx == 1:
                for i in range(8):
                    pred = lossdict['pred'][i].cpu().data.numpy()[0]
                    s = x[i].numpy()[0]
                    e = y[i].numpy()[0]
                    plt.figure(figsize=(15,5))
                    plt.suptitle(str(idx[i].data.numpy()))
                    plt.subplot(131)
                    plt.imshow(s)
                    plt.subplot(132)
                    plt.imshow(e)
                    plt.subplot(133)
                    plt.imshow(pred)
                    f = plt.gcf()
                    f.savefig('edges_pre/epoch{}_batch{}_pred.png'.format(epoch,i))
                    f.clear()
                    plt.close()
            # 获取当前损失
            loss = lossdict['loss']

            # 打印一下训练过程
            if batch_idx % 10 == 0:
                logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(
                    epoch, batch_idx, loss.item()))

            # 记录总共的损失
            results['loss'] += loss.item()
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

def train_typical(epoch, net, device, train_data, optimizer, batches_per_epoch):
    """
    :功能: 执行一个训练epoch,注意batch_size = 1,程序内会对typial目标进行判断,仅训练简单样本
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
        for x, _, idx, rot, zoom, y in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break
            
            percent, _ = train_data.dataset.check_typical(idx,rot,zoom)
            if percent < 0.70:
                break
            # 将数据传到GPU
            xc = x.to(device)
            yc = y.to(device)

            lossdict = net.compute_loss(xc, yc)
            for i in range(1):
                pred = lossdict['pred'][i].cpu().data.numpy()[0]
                s = x[i].numpy()[0]
                e = y[i].numpy()[0]
                plt.figure(figsize=(15,5))
                plt.suptitle(str(idx[i].data.numpy()))
                plt.subplot(131)
                plt.imshow(s)
                plt.subplot(132)
                plt.imshow(e)
                plt.subplot(133)
                plt.imshow(pred)
                f = plt.gcf()
                f.savefig('edges_pre_t/epoch{}_batch{}_pred.png'.format(epoch,batch_idx))
                f.clear()
                plt.close()
            # 获取当前损失
            loss = lossdict['loss']

            # 打印一下训练过程
            if batch_idx % 10 == 0:
                logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(
                    epoch, batch_idx, loss.item()))

            # 记录总共的损失
            results['loss'] += loss.item()
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

def run():
    # 设置输出文件夹
    out_dir = 'trained_models/'
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_tb'.format(dt)

    save_folder = os.path.join(out_dir, dt)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 获取设备
    device = torch.device("cuda:0")

    net = EENET(1)
    net = net.to(device)

    # 保存网络和训练参数信息
    summary(net, (1, 300, 300))
    f = open(os.path.join(save_folder, 'arch.txt'), 'w')
    sys.stdout = f
    summary(net, (1, 300, 300))
    sys.stdout = sys.__stdout__
    f.close()
    with open(os.path.join(save_folder, 'params.txt'), 'w') as f:
        f.write('batch_size:{}\nbatches_per_epoch:{}\nepochs:{}\nlr:{}'.format(
            batch_size, batches_per_epoch, epochs, lr))
    # 准备数据集
    # 训练集
    train_data = Jacquard('./jacquard', include_rgb=use_rgb, include_depth=use_depth,start=0.0,
                          end=split, random_rotate=r_rotate, random_zoom=r_zoom, output_size=300)
    # train_data = Jacquard('./jacquard', include_rgb=use_rgb, include_depth=use_depth,start=0.0,
    #                       end=split, random_rotate=r_rotate, random_zoom=r_zoom, output_size=300,load_from_txt = True,txt_path = 'no_typical_0.7.npy')
    train_dataset = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # 设置优化器
    optimizer = optim.Adam(net.parameters())
    # 设置tensorboardX
    # 开始主循环
    for epoch in range(100):
        train_results = train(epoch, net, device,
                              train_dataset, optimizer, batches_per_epoch)
    # # 典型训练集,batch_size = 1
    # train_data = Jacquard('./jacquard', include_rgb=use_rgb, include_depth=use_depth,start=0.0,
    #                       end=split, random_rotate=r_rotate, random_zoom=r_zoom, output_size=300)

    # train_dataset = torch.utils.data.DataLoader(
    #     train_data, batch_size=1, shuffle=True, num_workers=num_workers)
    # for epoch in range(50):
    #     train_results = train_typical(epoch, net, device,
    #                           train_dataset, optimizer, batches_per_epoch)
    torch.save(net.state_dict(), 'model.pth')
    return train_results


if __name__ == '__main__':
    run()