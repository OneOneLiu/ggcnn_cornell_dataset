from warnings import simplefilter 
simplefilter(action='ignore', category=DeprecationWarning)

# 训练函数定义
import paddle
import paddle.fluid as fluid
import numpy as np
from paddle.io import Dataset, DataLoader
import time

from cornell_pro import Cornell
from grasp_pro import Grasps,Grasp,Grasp_cpaw
from ggcnn import GGCNN
from functions import post_process,detect_grasps,max_iou

lr = 0.001

batch_size = 16
epoch_nums = 50
split = 0.9
include_depth = True
include_rgb = True
random_rotate = True
random_zoom = True

train_batches = 1000
val_batches = 250

num_workers = 6


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

def train(epoch_num,train_batches,params_dir = None):
    with fluid.dygraph.guard():
        net = GGCNN(include_depth+include_rgb*3)
        # 加载之前训练的模型继续训练
        if params_dir is not None:
            model, _ = fluid.dygraph.load_dygraph(params_dir)
            net.load_dict(model)
        optimizer = fluid.optimizer.Adam(parameter_list=net.parameters(),learning_rate=lr)

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
                xc = fluid.dygraph.to_variable(x)
                yc = [fluid.dygraph.to_variable(y[0][i]) for i in range(4)]

                # 计算loss
                losses = net.compute_loss(xc,yc)
                loss = losses['loss']

                #打印一下训练过程
                if batch_id % 100 == 0:
                    print('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch_num, batch_id, loss.numpy().item()))

                # 反向传播
                loss.backward()
                # 优化参数
                optimizer.minimize(loss)
                net.clear_gradients()

                # 保存网络参数，validate 用
        fluid.dygraph.save_dygraph(net.state_dict(), "save_dir/params")
        train_result = {'loss':loss.numpy().item()}
    return train_result

def eval_net(val_batches,params_dir):
    with fluid.dygraph.guard():
        print('validating...')
        net = GGCNN(include_depth+include_rgb*3)
        model, _ = fluid.dygraph.load_dygraph(params_dir)
        net.load_dict(model)
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

                xc = fluid.dygraph.to_variable(x)
                yc = [fluid.dygraph.to_variable(y[0][i]) for i in range(4)]

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
    params_dir = None 
    for epoch_num in range(epoch_nums):
        train_result = train(epoch_num,train_batches,params_dir = params_dir)
        params_dir = "save_dir/params"
        val_result = eval_net(val_batches,params_dir = params_dir)

