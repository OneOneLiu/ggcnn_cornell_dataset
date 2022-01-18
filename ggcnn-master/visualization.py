# 写这个脚本用来可视化当前训练的不同实验的结果
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import re

font1 = {'family': 'Nimbus Roman',
        'weight': 'normal',
        'style':'normal',
        'size': 20,
        }

font2 = {'family': 'Nimbus Roman',
        'weight': 'normal',
        'style':'italic',
        'size': 20,
        }
def visual_acc(filepath,dis = True,smooth = False):
    '''visualize the result of one output log, epoch-wise
    Args:
        filepath (str) : path of the log file
        dis (bool)     : whether to displa the result
        smooth (bool)  : whether to smooth the result

    Returns:
        accuracies (tupel[list[acc],[true acc]]) : accuracies of every epoch
    '''
    with open(filepath,'r') as f:
        log = f.read()
    acc = re.findall('iou_acc:.+= (\d.\d{4})',log)
    p_acc = re.findall('perfect_acc:.+= (\d.\d{4})',log)
    true_accs = []
    for i in range(20):
        true_acc = re.findall('edge_'+str(i)+':.+= (0.\d{4})',log)
        true_accs.append(true_acc)
    acc  = np.asarray(acc).astype(np.float)
    p_acc = np.asarray(p_acc).astype(np.float)
    true_accs  = np.asarray(true_accs).astype(np.float)

    if dis:
        x = np.arange(0,len(true_accs[0]))
        y = true_accs
        lines = ['-','--','-.']
        
        markers = ['o','v','d']
        labels = ['IoU accuracy','Feasible accuracy','Perfect accuracy']
        plt.grid()
        plt.ylim(0.0,1)
        plt.xlabel('epochs',fontsize = 20)
        plt.ylabel('accuracy',fontsize = 20)
        plt.title('accuracy',fontsize = 20)
        plt.plot(x,acc,linestyle = lines[0],marker = markers[0],label = labels[0])
        if len(p_acc) > 0:
            plt.plot(x,p_acc,linestyle = lines[2],marker = markers[2],label = labels[2])
        plt.title(filepath)
        for i,true_acc in enumerate(true_accs):
            if smooth:
                model=make_interp_spline(x,true_acc)
                xs = np.linspace(0,x[-1]+1,100)
                ys = model(xs)
                markers = ['','']
            else:
                xs = x
                ys = true_acc
            plt.plot(xs,ys,linestyle = lines[1],marker = markers[1],label = labels[1] + str(i))
            plt.legend(prop={'size':15})
        plt.show()
    return (acc,true_acc)

def conmpare_acc(filepath,dis = True,smooth = False):
    '''visualize the result of one output log, epoch-wise
    Args:
        filepath (str) : path of the log file
        dis (bool)     : whether to displa the result
        smooth (bool)  : whether to smooth the result

    Returns:
        accuracies (tupel[list[acc],[true acc]]) : accuracies of every epoch
    '''
    with open(filepath,'r') as f:
        log = f.read()
    acc = re.findall('iou_acc:.+= (\d.\d{4})',log)
    p_acc = re.findall('perfect_acc:.+= (\d.\d{4})',log)
    true_accs = []
    for i in range(20):
        true_acc = re.findall('edge_'+str(i)+':.+= (0.\d{4})',log)
        true_accs.append(true_acc)
    acc  = np.asarray(acc).astype(np.float)
    p_acc = np.asarray(p_acc).astype(np.float)
    true_accs  = np.asarray(true_accs).astype(np.float)

    if dis:
        x = np.arange(0,len(true_accs[0]))
        lines = ['-','--','-.']
        
        markers = ['o','v','d']
        labels = ['iou_acc','true_acc','perfect_acc']
        plt.grid()
        plt.ylim(0.0,1)
        plt.xlabel('epochs',fontsize = 16)
        plt.ylabel('accuracy',fontsize = 16)
        plt.title('accuracy',fontsize = 16)
        plt.plot(x,acc,linestyle = lines[0],marker = markers[0],label = labels[0])
        
        if len(p_acc) > 0:
            plt.plot(x,p_acc,linestyle = lines[2],marker = markers[2],label = labels[2])
        plt.title(filepath)
        for i,true_acc in enumerate(true_accs):
            if smooth:
                model=make_interp_spline(x,true_acc)
                xs = np.linspace(0,x[-1]+1,100)
                ys = model(xs)
                markers = ['','']
            else:
                xs = x
                ys = true_acc
            plt.plot(xs,ys,linestyle = lines[1],marker = markers[1],label = labels[1]+str(i))
            plt.legend()
        plt.show()
    return (acc,true_acc)

def visual_acc_ex(filepath,dis = True,smooth = False):
    '''visualize the result of one output log, epoch-wise
    Args:
        filepath (str) : path of the log file
        dis (bool)     : whether to displa the result
        smooth (bool)  : whether to smooth the result

    Returns:
        accuracies (tupel[list[acc],[true acc]]) : accuracies of every epoch
    '''
    with open(filepath,'r') as f:
        log = f.read()
    acc = re.findall('iou_acc:.+= (\d.\d{4})',log)
    p_acc = re.findall('perfect_acc:.+= (\d.\d{4})',log)
    true_accs = []
    for i in range(2):
        true_acc = re.findall('edge_'+str(i)+':.+= (0.\d{4})',log)
        true_accs.append(true_acc)
    acc  = np.asarray(acc).astype(np.float)
    p_acc = np.asarray(p_acc).astype(np.float)
    true_accs  = np.asarray(true_accs).astype(np.float)

    if dis:
        x = np.arange(0,len(true_accs[0]))
        y = true_accs
        lines = ['-','--','-.']
        
        markers = ['o','v','d']
        labels = ['IoU accuracy','Feasible accuracy','Perfect accuracy']
        plt.grid()
        plt.ylim(0.0,1)
        plt.xlabel('epochs',fontsize = 20)
        # plt.ylabel('accuracy',fontsize = 20)
        plt.title('accuracy',fontsize = 20)
        plt.tick_params(axis='x', labelsize=20)
        plt.tick_params(axis='y', labelsize=20)
        plt.plot(x,acc,linestyle = lines[0],marker = markers[0],label = labels[0])
        plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
        plt.title('Accuracy of Original GGCNN', fontsize = 20)
        for i,true_acc in enumerate(true_accs[::]):
            if smooth:
                model=make_interp_spline(x,true_acc)
                xs = np.linspace(0,x[-1]+1,100)
                ys = model(xs)
                markers = ['','']
            else:
                xs = x
                ys = true_acc
            plt.plot(xs,ys,linestyle = lines[1],marker = markers[1],label = labels[1])
            if len(p_acc) > 0:
                plt.plot(x,p_acc,linestyle = lines[2],marker = markers[2],label = labels[2])
            plt.legend(loc = 'lower right', prop={'size':20})
            break
        plt.show()
    return (acc,true_acc)

def visual_acc_3_value(filepath,dis = True,smooth = False):
    ''' 可视化最大值最小值和平均值的实验
    Args:
        filepath (str) : path of the log file
        dis (bool)     : whether to displa the result
        smooth (bool)  : whether to smooth the result

    Returns:
        accuracies (tupel[list[acc],[true acc]]) : accuracies of every epoch
    '''
    with open(filepath,'r') as f:
        log = f.read()
        acc = re.findall('iou_acc:.+= (\d.\d{4})',log)
    p_acc = re.findall('perfect_acc:.+= (\d.\d{4})',log)
    true_accs = []
    for i in range(20):
        true_acc = re.findall('edge_'+str(i)+':.+= (0.\d{4})',log)
        true_accs.append(true_acc)
    acc  = np.asarray(acc).astype(np.float)
    p_acc = np.asarray(p_acc).astype(np.float)
    true_accs  = np.asarray(true_accs).astype(np.float)
    # 数据分组
    length = 100
    groups_num = len(acc) // length
    groups_acc = []
    groups_p_acc = []
    groups_f_acc = []
    for i in range(groups_num):
        groups_acc.append(acc[i*length:i*length+length])
        groups_p_acc.append(p_acc[i*length:i*length+length])
        groups_f_acc.append(true_accs[3][i*length:i*length+length])
    groups_acc = np.asarray(groups_acc).astype(np.float)
    groups_p_acc = np.asarray(groups_p_acc).astype(np.float)
    groups_f_acc = np.asarray(groups_f_acc).astype(np.float)
    # 各求平均值,最大值和最小值
    avg_acc = np.mean(groups_acc,axis = 0)
    max_acc = np.max(groups_acc,axis = 0)
    min_acc = np.min(groups_acc,axis = 0)

    avg_p_acc = np.mean(groups_p_acc,axis = 0)
    max_p_acc = np.max(groups_p_acc,axis = 0)
    min_p_acc = np.min(groups_p_acc,axis = 0)

    avg_f_acc = np.mean(groups_f_acc,axis = 0)
    max_f_acc = np.max(groups_f_acc,axis = 0)
    min_f_acc = np.min(groups_f_acc,axis = 0)

    x = np.arange(0,length)
    fig, ax = plt.subplots(figsize = (10,10))
    ax.fill_between(x, max_acc, min_acc, alpha=.4, linewidth=0.5)
    ax.plot(x, avg_acc, linewidth=3,label='IoU Acc')

    ax.fill_between(x, max_f_acc, min_f_acc, alpha=.4, linewidth=0.5)
    ax.plot(x, avg_f_acc, linewidth=3,label='Feasible Acc',linestyle = '-.')

    ax.fill_between(x, max_p_acc, min_p_acc, alpha=.4, linewidth=0.5)
    ax.plot(x, avg_p_acc, linewidth=3,label='Perfect Acc',linestyle = '--')

    
    # tidy the figure
    ax.set(xlim=(-1, length), xticks=np.linspace(0, length, length // 10 +1),
           ylim=(0, 1), yticks=np.linspace(0, 1, length // 10 +1))
    ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
    ax.grid(True)
    ax.legend(loc='lower right',prop = font1)
    ax.set_title('Accuracy (Jacquard typical GGCNN2 patch)',font1)
    ax.set_xlabel('Epoches',font1)
    ax.set_ylabel('Accuracy',font1)
    ax.tick_params(axis='both',labelsize = 15)
    return 0

def to_percent(temp, position):
    return '%1.0f'%(100*temp) + '%'

# 绘制三个曲线图
filepath = 'output/models/experiment3:patch/220106_0956_/logger.log'

accuracies = visual_acc_3_value(filepath,smooth=False)


pass
# 以前的图测试
filepath = 'output/models/experiment3:patch/220106_0956_/logger.log'

accuracies = visual_acc_ex(filepath,smooth=False)