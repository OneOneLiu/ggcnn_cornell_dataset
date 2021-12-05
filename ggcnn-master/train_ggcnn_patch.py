import datetime
import os
import sys
import argparse
import logging

import cv2

import torch
import torch.utils.data
import torch.optim as optim

from torchsummary import summary

import tensorboardX

from utils.visualisation.gridshow import gridshow

from utils.dataset_processing import evaluation
from utils.dataset_processing.evaluation import collision_validate
from utils.data import get_dataset
from models import get_network
from models.common import post_process_output

# Set-up output directories
dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
net_desc = '{}_{}'.format(dt, '_'.join(''))

save_folder = os.path.join('output/models/', net_desc)
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

pretrain = False
load_part = False
model_path = 'output/models/211126_2004_/epoch_46_iou_0.87_statedict.pt'

logger.info('Pretrain:' + str(pretrain))
if pretrain:
    logger.info('load model from :' + model_path)

# patch_size = input("patch_size")
# logger.info('patch_size:' + str([patch_size]))

def parse_args():
    parser = argparse.ArgumentParser(description='Train GG-CNN')

    # Network
    parser.add_argument('--network', type=str, default='ggcnn2_patch_v4', help='Network Name in .models')

    # Dataset & Data & Training5
    parser.add_argument('--dataset', default='jacquard_t',type=str, help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', default = './jacquard',type=str, help='Path to dataset')
    parser.add_argument('--ADJ', default = 0,type=int, help='Whether to use ADJ dataset')
    parser.add_argument('--ADJtrain-path', default =
     'train_ADJ.npy',type=str, help='Path to dataset')
    parser.add_argument('--ADJtest-path', default = 'test_ADJ.npy',type=str, help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for training (0/1)')
    parser.add_argument('--split', type=float, default=0.95, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset t1 use a different test/train split for cross validation.')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=300, help='Batches per Epoch')
    parser.add_argument('--val-batches', type=int, default=250, help='Validation Batches')

    # Validating
    parser.add_argument('--position', type=str, default='pos', help='position img used whilt validating')
    parser.add_argument('--ang_cos', type=str, default='cos', help='angle cos img used whilt validating')
    parser.add_argument('--ang_sin', type=str, default='sin', help='angle_sin img used whilt validating')
    parser.add_argument('--width', type=str, default='width', help='width img used whilt validating')
    
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
        # 基于prob+后面三个过滤后的映射图
        'patch':
            {'correct': 0,
            'failed': 0,
            'positive':0,
            'negative':0,
            'perfect':0,
            'not_perfect':0}
    }
    for i in range(20):
        results['pos']['positive_'+str(i)] = 0
        results['pos']['negative_'+str(i)] = 0
    for i in range(20):
        results['prob']['positive_'+str(i)] = 0
        results['prob']['negative_'+str(i)] = 0
    for i in range(20):
        results['patch']['positive_'+str(i)] = 0
        results['patch']['negative_'+str(i)] = 0
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
                
                # 基于两个映射图进行不同的预测
                for pos in ['pos','prob','patch']:
                    if pos == 'patch':
                        q_out, ang_out, w_out = post_process_output(lossd['pred']['prob'], lossd['pred']['filtered_cos'],
                                                                lossd['pred']['filtered_sin'], lossd['pred']['filtered_width'])
                    else:
                        q_out, ang_out, w_out = post_process_output(lossd['pred'][pos], lossd['pred']['cos'],
                                                                lossd['pred']['sin'], lossd['pred']['width'])
                    gs,s = evaluation.calculate_iou_match(q_out, ang_out,
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
                        if lr_congruous < 0.85 and tb_congruous < 0.85:
                            results[pos]['perfect'] += 1
                        else:
                            results[pos]['not_perfect'] += 1
                    else:
                        for edge_threshold in range(0,20):
                            results[pos]['negative_'+str(edge_threshold)] += 1
                        results[pos]['not_perfect'] += 1
    return results

def train(epoch, net, device, train_data, optimizer, batches_per_epoch, vis=False):
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
            include_patch = 0
            if epoch > 50:
                include_patch = 1
            lossd = net.compute_loss(xc, yc, include_patch)

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

            # Display the images
            if vis:
                imgs = []
                n_img = min(4, x.shape[0])
                for idx in range(n_img):
                    imgs.extend([x[idx,].numpy().squeeze()] + [yi[idx,].numpy().squeeze() for yi in y] + [
                        x[idx,].numpy().squeeze()] + [pc[idx,].detach().cpu().numpy().squeeze() for pc in lossd['pred'].values()])
                gridshow('Display', imgs,
                         [(xc.min().item(), xc.max().item()), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (0.0, 1.0)] * 2 * n_img,
                         [cv2.COLORMAP_BONE] * 10 * n_img, 10)
                cv2.waitKey(2)

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results


def run():
    args = parse_args()
    logger.info(args)
    # Vis window
    print(os.getcwd())
    if args.vis:
        cv2.namedWindow('Display', cv2.WINDOW_NORMAL)

    tb = tensorboardX.SummaryWriter(os.path.join(args.logdir, net_desc))

    # Load Dataset
    logger.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)

    train_dataset = Dataset(args.dataset_path, start=0.0, end=args.split, ds_rotate=args.ds_rotate,
                            random_rotate=True, random_zoom=True,ADJ = args.ADJ,npy_path = args.ADJtrain_path,
                            include_depth=args.use_depth, include_rgb=args.use_rgb)
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_dataset = Dataset(args.dataset_path, start=args.split, end=1.0, ds_rotate=args.ds_rotate,
                          random_rotate=True, random_zoom=True,ADJ = 0, npy_path = args.ADJtest_path,
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
    if load_part:
        checkpoint= torch.load(model_path)
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict} 
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
    elif pretrain:
        net.load_state_dict(torch.load(model_path))
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

    # 先validate10次
    if pretrain:
        for i in range(10):
            logger.info('Validating...')
            val_results = validate(net, device, val_data, args.val_batches)
            test_results = val_results['pos']
            test_results_prob = val_results['prob']
            test_results_patch = val_results['patch']

            # 输出信息到屏幕
            logger.info('iou_acc:%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                        test_results['correct']/(test_results['correct']+test_results['failed'])))
            for i in range(20):
                logger.info('edge_%d:%d/%d = %f' % (i,test_results['positive_'+str(i)], test_results['positive_'+str(i)] + test_results['negative_'+str(i)],test_results['positive_'+str(i)]/(test_results['positive_'+str(i)]+test_results['negative_'+str(i)])))

            logger.info('perfect_acc:%d/%d = %f' % (test_results['perfect'], test_results['perfect'] + test_results['not_perfect'],test_results['perfect']/(test_results['perfect']+test_results['not_perfect'])))

            # 输出第二个结果
            logger.info('iou_prob_acc:%d/%d = %f' % (test_results_prob['correct'], test_results_prob['correct'] + test_results_prob['failed'],
                                        test_results_prob['correct']/(test_results_prob['correct']+test_results_prob['failed'])))
            for i in range(20):
                logger.info('edge_prob_%d:%d/%d = %f' % (i,test_results_prob['positive_'+str(i)], test_results_prob['positive_'+str(i)] + test_results_prob['negative_'+str(i)],test_results_prob['positive_'+str(i)]/(test_results_prob['positive_'+str(i)]+test_results_prob['negative_'+str(i)])))

            logger.info('perfect_prob_acc:%d/%d = %f' % (test_results_prob['perfect'], test_results_prob['perfect'] + test_results_prob['not_perfect'],test_results_prob['perfect']/(test_results_prob['perfect']+test_results_prob['not_perfect'])))
            # 输出第三个结果
            logger.info('iou_patch_acc:%d/%d = %f' % (test_results_patch['correct'], test_results_patch['correct'] + test_results_patch['failed'],test_results_patch['correct']/(test_results_patch['correct']+test_results_patch['failed'])))
            for i in range(20):
                logger.info('edge_patch_%d:%d/%d = %f' % (i,test_results_patch['positive_'+str(i)], test_results_patch['positive_'+str(i)] + test_results_patch['negative_'+str(i)],test_results_patch['positive_'+str(i)]/(test_results_patch['positive_'+str(i)]+test_results_patch['negative_'+str(i)])))

            logger.info('perfect_patch_acc:%d/%d = %f' % (test_results_patch['perfect'], test_results_patch['perfect'] + test_results_patch['not_perfect'],test_results_patch['perfect']/(test_results_patch['perfect']+test_results_patch['not_perfect'])))
    for epoch in range(args.epochs):
        logger.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, net, device, train_data, optimizer, args.batches_per_epoch, vis=args.vis)

        # Log training losses to tensorboard
        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)

        # Run Validation
        logger.info('Validating...')
        val_results = validate(net, device, val_data, args.val_batches)
        test_results = val_results['pos']
        test_results_prob = val_results['prob']
        test_results_patch = val_results['patch']

        # 输出信息到屏幕
        logger.info('iou_acc:%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct']/(test_results['correct']+test_results['failed'])))
        for i in range(20):
            logger.info('edge_%d:%d/%d = %f' % (i,test_results['positive_'+str(i)], test_results['positive_'+str(i)] + test_results['negative_'+str(i)],test_results['positive_'+str(i)]/(test_results['positive_'+str(i)]+test_results['negative_'+str(i)])))

        logger.info('perfect_acc:%d/%d = %f' % (test_results['perfect'], test_results['perfect'] + test_results['not_perfect'],test_results['perfect']/(test_results['perfect']+test_results['not_perfect'])))

        # 输出第二个结果
        logger.info('iou_prob_acc:%d/%d = %f' % (test_results_prob['correct'], test_results_prob['correct'] + test_results_prob['failed'],
                                     test_results_prob['correct']/(test_results_prob['correct']+test_results_prob['failed'])))
        for i in range(20):
            logger.info('edge_prob_%d:%d/%d = %f' % (i,test_results_prob['positive_'+str(i)], test_results_prob['positive_'+str(i)] + test_results_prob['negative_'+str(i)],test_results_prob['positive_'+str(i)]/(test_results_prob['positive_'+str(i)]+test_results_prob['negative_'+str(i)])))

        logger.info('perfect_prob_acc:%d/%d = %f' % (test_results_prob['perfect'], test_results_prob['perfect'] + test_results_prob['not_perfect'],test_results_prob['perfect']/(test_results_prob['perfect']+test_results_prob['not_perfect'])))
        # 输出第三个结果
        logger.info('iou_patch_acc:%d/%d = %f' % (test_results_patch['correct'], test_results_patch['correct'] + test_results_patch['failed'],test_results_patch['correct']/(test_results_patch['correct']+test_results_patch['failed'])))
        for i in range(20):
            logger.info('edge_patch_%d:%d/%d = %f' % (i,test_results_patch['positive_'+str(i)], test_results_patch['positive_'+str(i)] + test_results_patch['negative_'+str(i)],test_results_patch['positive_'+str(i)]/(test_results_patch['positive_'+str(i)]+test_results_patch['negative_'+str(i)])))

        logger.info('perfect_patch_acc:%d/%d = %f' % (test_results_patch['perfect'], test_results_patch['perfect'] + test_results_patch['not_perfect'],test_results_patch['perfect']/(test_results_patch['perfect']+test_results_patch['not_perfect'])))


        # Log validation results to tensorbaord
        tb.add_scalar('loss/IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']), epoch)
        tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        for n, l in test_results['losses'].items():
            tb.add_scalar('val_loss/' + n, l, epoch)

        # Save best performing network
        iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        # torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, iou)))
        torch.save(net.state_dict(), os.path.join(save_folder, 'epoch_%02d_iou_%0.2f_statedict.pt' % (epoch, iou)))

if __name__ == '__main__':
    for i in range(10):
        run()