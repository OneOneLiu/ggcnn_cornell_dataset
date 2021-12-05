# -*- coding: utf-8 -*-
"""
Created on Thur July 15 14:55:01 2021


@author: LiuDaohui
"""

# 导入第三方包
import torch
import torch.optim as optim
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from imageio import imsave

import time
import datetime
import os
# summary输出保存之后print函数失灵，但还可以用这个logging来打印输出
import logging
import sys
# 导入自定义包
from jacquard import Jacquard
from ggcnn2 import GGCNN2
from functions import post_process, detect_grasps, max_iou, collision_validate
net_path = '/home/ldh/github_repositorys/ggcnn_cornell_dataset/12.patch_accelerate/trained_models/filter'
# 一些训练参数的设定
batch_size = 8
epochs = 100
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

# 设置输出文件夹
out_dir = '12.patch_accelerate/trained_models/prob'
dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
net_desc = '{}_tb'.format(dt)
save_folder = os.path.join(out_dir, dt)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
fh = logging.FileHandler(filename = os.path.join(save_folder,'logger.log'), mode='w', encoding='utf-8')  # 不拆分日志文件，a指追加模式,w为覆盖模式
fh.setLevel(logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(ch)
logger.addHandler(fh)

model_path = '210911_1608/model0.943_epoch48_batch_8.pth'
logger.info('ADJ')
logger.info(model_path)

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
        for x, y, _, _, _ in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break

            # 将数据传到GPU
            xc = x.to(device)
            yc = [yy.to(device) for yy in y]

            lossdict = net.compute_loss(xc, yc)

            # 获取当前损失
            loss = lossdict['loss']

            # 打印一下训练过程
            if batch_idx % 10 == 0:
                logger.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(
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
            for x, y, idx, rot, zoom_factor in val_data:
                batch_idx += 1
                if batch_idx >= batches_per_epoch:
                    break
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

                q_out, ang_out, width_out = post_process(lossdict['pred']['pos'], lossdict['pred']['cos'],
                                                         lossdict['pred']['sin'], lossdict['pred']['width'])
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
                # 真实精度检测
                if len(grasp_pres) > 0:
                    edge = collision_validate(grasp_pre.as_gr,yc[5])
                    if edge > 5:
                        val_result['positive'] += 1
                    else:
                        val_result['negative'] += 1
                else:
                    val_result['negative'] += 1
        logger.info(time.ctime())
        acc = val_result['correct'] / \
            (val_result['correct']+val_result['failed'])
        logger.info('iou_acc:{}'.format(acc))
        val_result['acc'] = acc
        true_acc = val_result['positive'] / (val_result['positive']+val_result['negative'])
        logger.info('true_acc:{}'.format(true_acc))
    return(val_result)

def run():
    device = torch.device("cuda:0")

    # 实例化一个网络
    input_channels = 1*use_depth+3*use_rgb
    net = GGCNN2(input_channels)
    net.load_state_dict(torch.load(os.path.join(net_path,'210910_1711/model0.923_epoch92_batch_8.pth')))
    net = net.to(device)
    # 保存网络和训练参数信息
    summary(net, (1*use_depth+3*use_rgb, 300, 300))
    f = open(os.path.join(save_folder, 'arch.txt'), 'w')
    sys.stdout = f
    summary(net, (input_channels, 300, 300))
    sys.stdout = sys.__stdout__
    f.close()
    with open(os.path.join(save_folder, 'params.txt'), 'w') as f:
        f.write('batch_size:{}\nbatches_per_epoch:{}\nepochs:{}\nlr:{}'.format(
            batch_size, batches_per_epoch, epochs, lr))
    # 准备数据集
    # 训练集
    train_data = Jacquard('./jacquard', include_rgb=use_rgb, include_depth=use_depth,start=0.0,end=split, random_rotate=r_rotate, random_zoom=r_zoom, output_size=300,load_from_npy = True, npy_path = 'train_ADJ.npy')
    train_dataset = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # 验证集
    val_data = Jacquard('./jacquard', include_rgb=use_rgb, include_depth=use_depth, start=split,end=1.0, random_rotate=r_rotate, random_zoom=r_zoom, output_size=300,load_from_npy = True, npy_path = 'test_ADJ.npy')
    val_dataset = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=num_workers)

    # 设置优化器
    optimizer = optim.Adam(net.parameters())
    # 设置tensorboardX
    tb = SummaryWriter(log_dir=os.path.join(save_folder, net_desc))
    # 开始主循环
    # 添加模型图
    test_img = torch.randn((batch_size, input_channels, 300, 300))
    tb.add_graph(net, test_img.to(device))
    for i in range(10):
        logger.info('validating...')
        validate_results = validate(net, device, val_dataset, batches_per_epoch=val_batches)
    # 载入模型并训练一定的epoch
    for epoch in range(epochs):
        train_results = train(epoch, net, device,
                              train_dataset, optimizer, batches_per_epoch)

        # 添加总的loss到tb
        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        # 添加各项的单独loss到tb
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)

        logger.info('validating...')
        validate_results = validate(
            net, device, val_dataset, batches_per_epoch=val_batches)

        # 添加IOU到tb
        tb.add_scalar('loss/IOU', validate_results['correct'] / (
            validate_results['correct'] + validate_results['failed']), epoch)
        # 添加各项的单独loss到tb
        tb.add_scalar('loss/val_loss', validate_results['loss'], epoch)
        for n, l in validate_results['losses'].items():
            tb.add_scalar('val_loss/' + n, l, epoch)

        max_acc = validate_results['acc']
        torch.save(net.state_dict(), '{0}/model{1}_epoch{2}_batch_{3}.pth'.format(
            save_folder, str(validate_results['acc'])[0:5], epoch, batch_size))

if __name__ == '__main__':
    run()
