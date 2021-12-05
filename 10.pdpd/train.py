# 训练函数定义
import paddle
import paddle.fluid as fluid
import numpy as np
from paddle.io import Dataset, DataLoader
import time

from cornell_pro import Cornell
from ggcnn import GGCNN
from functions import post_process,detect_grasps,max_iou

lr = 0.001

batch_size = 2
epoch_nums = 20
split = 0.9
include_depth = True
include_rgb = True
random_rotate = True
random_zoom = True

train_batches = 100
val_batches = 25

num_workers = 0


#准备数据集
cornell_path = 'cornell'

train_data = Cornell(cornell_path,start=0.0,end=split,include_depth=include_depth,include_rgb = include_rgb,random_rotate = random_rotate,random_zoom = random_zoom)

eval_data = Cornell(cornell_path,start=split,end=1.0,include_depth=include_depth,include_rgb = include_rgb,random_rotate = random_rotate,random_zoom = random_zoom)

train_loader = DataLoader(train_data,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=num_workers)
eval_loader = DataLoader(eval_data,
                    batch_size=1,
                    shuffle=True,
                    drop_last=True,
                    num_workers=num_workers)

def train(net,epoch_num,train_batches,params_dir = None):
    # 加载之前训练的模型继续训练
    optim = paddle.optimizer.Adam(parameters=net.parameters())
    print("Training...")
    # 模型训练
    net.train()
    batch_id = 0
    while batch_id < train_batches:
        for x,y,_,_,_ in train_loader():
            batch_id += 1
            if batch_id > train_batches:
                break

            # 读入构建训练数据
            xc = x
            yc = [y[0][i] for i in range(4)]

            # 计算loss
            losses = net.compute_loss(xc,yc)
            loss = losses['loss']

            #打印一下训练过程
            if batch_id % 100 == 0:
                print('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch_num, batch_id, loss.numpy().item()))
            # 反向传播
            loss.backward()
            # 优化参数
            optim.step()
            optim.clear_grad()
    train_result = {'loss':loss.numpy().item()}
    return train_result

def eval_net(net,val_batches):
    net.eval()

    val_result = {
    'correct':0,
    'failed':0,
    'loss':0,
    'losses':{}
    }

    batch_id = 0
    while(batch_id < val_batches):
        for x,y,idx,rot,zoom_factor in eval_loader():
            batch_id += 1
            if batch_id > val_batches:
                break
            # 读入构建验证数据
            with paddle.no_grad():
                xc = x
                yc = [y[0][i] for i in range(4)]

                # 计算loss
                losses = net.compute_loss(xc,yc)
                val_result['loss'] += losses['loss']/val_batches

                # 记录各项的单独损失
                for ln, l in losses['losses'].items():
                    if ln not in val_result['losses']:
                        val_result['losses'][ln] = 0
                    val_result['losses'][ln] += l/val_batches
                q_out,ang_out,width_out = post_process(losses['pred']['pos'], losses['pred']['cos'], 
                                                    losses['pred']['sin'], losses['pred']['width'])
                
                grasps_pre = detect_grasps(q_out,ang_out,width_out,no_grasp = 1)
                grasps_true = eval_data.get_raw_grasps(idx,rot,zoom_factor)

                result = 0
                for grasp_pre in grasps_pre:
                    if max_iou(grasp_pre,grasps_true) > 0.25:
                        result = 1
                        break
            
            if result:
                val_result['correct'] += 1
            else:
                val_result['failed'] += 1
    acc = val_result['correct']/(val_result['correct']+val_result['failed'])
    val_result['acc'] = acc
    print(time.ctime())
    print('Correct: {}/{}, val_loss: {:0.4f}, acc: {:0.4f}'.format(val_result['correct'], val_result['correct']+val_result['failed'], val_result['loss'].numpy()[0], acc))
    return val_result
if __name__ == '__main__':
    use_gpu = True
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
    net = GGCNN(include_depth+include_rgb*3)
    paddle.summary(net, (1, 4, 300, 300))
    for epoch_num in range(epoch_nums):
        train_result = train(net,epoch_num,train_batches)
        print('validating...')
        val_result = eval_net(net,val_batches)
        fluid.dygraph.save_dygraph(net.state_dict(), "save_dir/params")

