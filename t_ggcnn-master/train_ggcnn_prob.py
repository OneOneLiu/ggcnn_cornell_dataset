import datetime
import os
import sys
import argparse
import logging

import cv2
# import time

import torch
import torch.utils.data
import torch.optim as optim

from torchsummary import summary

import tensorboardX

from utils.visualisation.gridshow import gridshow

from utils.dataset_processing import evaluation_new
from utils.dataset_processing.evaluation_new import collision_validate
from utils.data import get_dataset
from models import get_network
from models.common import post_process_output

# 设置输出路径
dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
net_desc = '{}_{}'.format(dt, '_'.join(''))

save_folder = os.path.join('output/2.prob/', net_desc)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 配置logging
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
fh = logging.FileHandler(filename = os.path.join(save_folder,'logger.log'), mode='w', encoding='utf-8')  # 不拆分日志文件，a指追加模式,w为覆盖模式
fh.setLevel(logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(ch)
logger.addHandler(fh)

# 是否载入预训练模型
pretrain = False

model_path = 'output/models/210928_1951_/epoch_42_iou_0.00_statedict.pt'

logger.info('Pretrain:' + str(pretrain))

if pretrain:
    logger.info('load model from :' + model_path)

map_list = ['pos','prob']

def parse_args():
    parser = argparse.ArgumentParser(description='Train GG-CNN')

    # Network
    parser.add_argument('--network', type=str, default='ggcnn2_prob', help='Network Name in .models')

    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, default = 'jacquard', help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str, default = './jacquard', help='Path to dataset')
    parser.add_argument('--ADJ', default = 0,type=int, help='Whether to use ADJ dataset')
    parser.add_argument('--ADJV', default = 0,type=int, help='Whether to use ADJ dataset for Validation')
    parser.add_argument('--ADJtrain-path', default = 'train_ADJ.npy',type=str, help='Path to dataset')
    parser.add_argument('--ADJtest-path', default = 'test_ADJ.npy',type=str, help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for training (0/1)')
    parser.add_argument('--split', type=float, default=0.95, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split for cross validation.')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=300, help='Batches per Epoch')
    parser.add_argument('--val-batches', type=int, default=250, help='Validation Batches')

    # Logging etc.
    parser.add_argument('--description', type=str, default='', help='Training description')
    parser.add_argument('--outdir', type=str, default='output/models/', help='Training Output Directory')
    parser.add_argument('--logdir', type=str, default='tensorboard/', help='Log directory')
    parser.add_argument('--vis', action='store_true', help='Visualise the training process')

    args = parser.parse_args()
    return args


def validate(net, device, val_data, batches_per_epoch):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param batches_per_epoch: Number of batches to run
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {
        # 基于原始的四个映射图
        'pos':
            {'correct': 0,
            'failed': 0,
            'loss': 0,
            'positive':0,
            'negative':0,
            'losses': {},
            'perfect':0,
            'not_perfect':0},
        # 基于原始的prob+后面三个映射图
        'prob':
            {'correct': 0,
            'failed': 0,
            'positive':0,
            'negative':0,
            'perfect':0,
            'not_perfect':0},
    }

    for i in range(20):
        results['pos']['positive_'+str(i)] = 0
        results['pos']['negative_'+str(i)] = 0
    for i in range(20):
        results['prob']['positive_'+str(i)] = 0
        results['prob']['negative_'+str(i)] = 0

    ld = len(val_data)

    with torch.no_grad():
        batch_idx = 0
        while batch_idx < batches_per_epoch:
            for x, y, didx, rot, zoom_factor in val_data:
                batch_idx += 1
                if batches_per_epoch is not None and batch_idx >= batches_per_epoch:
                    break

                xc = x.to(device)
                yc = [yy.to(device) for yy in y]
                lossd = net.compute_loss(xc, yc)

                loss = lossd['loss']

                results['pos']['loss'] += loss.item()/ld
                for ln, l in lossd['losses'].items():
                    if ln not in results['pos']['losses']:
                        results['pos']['losses'][ln] = 0
                    results['pos']['losses'][ln] += l.item()/ld
                
                for pos in map_list:

                    q_out, ang_out, w_out = post_process_output(lossd['pred'][pos], lossd['pred']['cos'],
                                                                lossd['pred']['sin'], lossd['pred']['width'])

                    gs,s = evaluation_new.calculate_iou_match(q_out, ang_out,
                                                    val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                                                    no_grasps=1,
                                                    grasp_width=w_out,
                                                    )
                    if s:
                        results[pos]['correct'] += 1
                    else:
                        results[pos]['failed'] += 1
                    
                    # 真实精度检测
                    if len(gs) > 0:
                        edge, lr_congruous, tb_congruous = collision_validate(gs[0].as_gr,yc[5])
                        for edge_threshold in range(0,20):
                            if edge > edge_threshold:
                                results[pos]['positive_'+str(edge_threshold)] += 1
                            else:
                                results[pos]['negative_'+str(edge_threshold)] += 1
                        if edge > 4 and lr_congruous < 0.85 and tb_congruous < 0.85:
                            results[pos]['perfect'] += 1
                        else:
                            results[pos]['not_perfect'] += 1
                    else:
                        for edge_threshold in range(0,20):
                            results[pos]['negative_'+str(edge_threshold)] += 1
                        results[pos]['not_perfect'] += 1
    return results

def train(epoch, net, device, train_data, optimizer, batches_per_epoch):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'losses': {
        }
    }

    net.train()

    batch_idx = 0
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    while batch_idx < batches_per_epoch:
        for x, y, _, _, _ in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break

            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(xc, yc)

            loss = lossd['loss']

            if batch_idx % 50 == 0:
                logger.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))

            results['loss'] += loss.item()
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results

def print_result(val_results):
    edge_num = 20 # 打印多少个edge的结果
    name_list = ['iou_acc','iou_prob_acc']
    perfect_name_list = ['perfect_acc','perfect_prob_acc']
    for pos, name, perfect_name in zip(map_list,name_list,perfect_name_list):
        test_results = val_results[pos]
        # 输出信息到屏幕
        logger.info(name + ':%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                    test_results['correct']/(test_results['correct']+test_results['failed'])))
        for i in range(edge_num):
            logger.info('edge_%d:%d/%d = %f' % (i,test_results['positive_'+str(i)], test_results['positive_'+str(i)] + test_results['negative_'+str(i)],test_results['positive_'+str(i)]/(test_results['positive_'+str(i)]+test_results['negative_'+str(i)])))

        logger.info(perfect_name + ':%d/%d = %f' % (test_results['perfect'], test_results['perfect'] + test_results['not_perfect'],test_results['perfect']/(test_results['perfect']+test_results['not_perfect'])))

def run(i):
    args = parse_args()

    # Vis window
    if args.vis:
        cv2.namedWindow('Display', cv2.WINDOW_NORMAL)

    tb = tensorboardX.SummaryWriter(os.path.join(args.logdir, net_desc))

    # Load Dataset
    logger.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)

    train_dataset = Dataset(args.dataset_path, start=0.0, end=args.split, ds_rotate=args.ds_rotate,
                            random_rotate=True, random_zoom=True,
                            include_depth=args.use_depth, include_rgb=args.use_rgb)
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_dataset = Dataset(args.dataset_path, start=args.split, end=1.0, ds_rotate=args.ds_rotate,
                          random_rotate=True, random_zoom=True,
                          include_depth=args.use_depth, include_rgb=args.use_rgb)
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    logger.info('Done')

    # Load the network
    logger.info('Loading Network...')
    input_channels = 1*args.use_depth + 3*args.use_rgb
    ggcnn = get_network(args.network)

    net = ggcnn(input_channels=input_channels)
    device = torch.device("cuda:0")
    net = net.to(device)
    optimizer = optim.Adam(net.parameters())
    logger.info('Done')

    # Print model architecture.
    summary(net, (input_channels, 300, 300))
    f = open(os.path.join(save_folder, 'arch.txt'), 'w')
    sys.stdout = f
    summary(net, (input_channels, 300, 300))
    sys.stdout = sys.__stdout__
    f.close()

    for epoch in range(args.epochs):
        logger.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, net, device, train_data, optimizer, args.batches_per_epoch)

        # Log training losses to tensorboard
        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)

        # Run Validation
        logger.info('Validating...')
        val_results = validate(net, device, val_data, args.val_batches)
        test_results = val_results['pos']
        print_result(val_results)

        # Log validation results to tensorbaord
        tb.add_scalar('loss/IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']), epoch)
        tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        for n, l in test_results['losses'].items():
            tb.add_scalar('val_loss/' + n, l, epoch)

        # Save best performing network
        iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        # if iou > best_iou or epoch == 0 or (epoch % 10) == 0:
        # torch.save(net, os.path.join(save_folder, '%02d_epoch_%02d_iou_%0.2f' % (i,epoch, iou)))
        torch.save(net.state_dict(), os.path.join(save_folder, '%02d_epoch_%02d_iou_%0.2f_statedict.pt' % (i,epoch, iou)))
            # best_iou = iou

if __name__ == '__main__':
    for i in range(10):
        run(i)