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
model_path = 'output/models/210916_1553_/epoch_09_iou_0.90_statedict.pt'
logger.info(str(pretrain))
if pretrain:
    logger.info('load model from :' + model_path)
def parse_args():
    parser = argparse.ArgumentParser(description='Train GG-CNN')

    # Network
    parser.add_argument('--network', type=str, default='ggcnn2_f', help='Network Name in .models')

    # Dataset & Data & Training
    parser.add_argument('--dataset', default='jacquard',type=str, help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', default = './jacquard',type=str, help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for training (0/1)')
    parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split for cross validation.')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
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
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'positive':0,
        'negative':0,
        'losses': {
        }
    }
    for i in range(20):
        results['positive_'+str(i)] = 0
        results['negative_'+str(i)] = 0

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

                results['loss'] += loss.item()/ld
                for ln, l in lossd['losses'].items():
                    if ln not in results['losses']:
                        results['losses'][ln] = 0
                    results['losses'][ln] += l.item()/ld

                q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                            lossd['pred']['sin'], lossd['pred']['width'])

                gs,s = evaluation.calculate_iou_match(q_out, ang_out,
                                                   val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                                                   no_grasps=1,
                                                   grasp_width=w_out,
                                                   )

                if s:
                    results['correct'] += 1
                else:
                    results['failed'] += 1
                
                # 真实精度检测
                if len(gs) > 0:
                    edge = collision_validate(gs[0].as_gr,yc[5])
                    for edge_threshold in range(0,20):
                        if edge > edge_threshold:
                            results['positive_'+str(edge_threshold)] += 1
                        else:
                            results['negative_'+str(edge_threshold)] += 1
                else:
                    for edge_threshold in range(0,20):
                        results['negative_'+str(edge_threshold)] += 1
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
    if pretrain:
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

    for epoch in range(args.epochs):
        logger.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, net, device, train_data, optimizer, args.batches_per_epoch, vis=args.vis)

        # Log training losses to tensorboard
        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)

        # Run Validation
        logger.info('Validating...')
        test_results = validate(net, device, val_data, args.val_batches)
        logger.info('iou_acc:%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct']/(test_results['correct']+test_results['failed'])))
        for i in range(20):
            logger.info('edge_%d:%d/%d = %f' % (i,test_results['positive_'+str(i)], test_results['positive_'+str(i)] + test_results['negative_'+str(i)],test_results['positive_'+str(i)]/(test_results['positive_'+str(i)]+test_results['negative_'+str(i)])))

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
    run()