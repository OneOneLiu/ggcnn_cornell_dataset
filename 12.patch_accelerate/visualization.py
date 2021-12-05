# 写这个脚本用来可视化当前训练的不同实验的结果
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import numpy as np
import re

def visual_per_epoch(filepath,dis = True,smooth = False,epoch_wise = True):
    '''visualize the result of one output log, epoch-wise
    Args:
        filepath (str) : path of the log file
        dis (bool)     : whether to displa the result
        smooth (bool)  : whether to smooth the result

    Returns:
        accuracies (list[list,list,...]) : accuracies of every epoch
    
    NOTE: warning,this function has been deprecated.
    '''
    with open(filepath,'r') as f:
        log = f.read()
    epochs = log.split('INFO:root:Validating....')
    accuracies = []
    for epoch in epochs:
        acc = re.findall('acc:(\d.\d{4})',epoch)
        if epoch_wise:
            accuracies.append(np.asarray(acc).astype(np.float))
        else:
            accuracies.extend(np.asarray(acc).astype(np.float))
    if dis:
        if epoch_wise:
            means = [np.mean(accuracy) for accuracy in accuracies]
            maxes = [max(accuracy) for accuracy in accuracies]
            mins = [min(accuracy) for accuracy in accuracies]

            x = np.arange(0,len(accuracies))
            x = np.stack((x,x,x))
            y = [means,maxes,mins]
            lines = ['-','--','-.']
            markers = ['o','v','s']
            labels = ['mean','max','min']
            plt.grid()
            plt.ylim(0.5,1)
            plt.xlabel('epochs',fontsize = 16)
            plt.ylabel('accuracy',fontsize = 16)
            plt.title('accquary',fontsize = 16)
            for i in range(3):
                if smooth:
                    model=make_interp_spline(x[i],y[i])
                    xs = np.linspace(0,x[i][-1]+1,100)
                    ys = model(xs)
                    markers = ['','','']
                else:
                    xs = x[i]
                    ys = y[i]
                plt.plot(xs,ys,linestyle = lines[i],marker = markers[i],label = labels[i])
                plt.legend()
            plt.show()
        else:
            x = np.arange(0,len(accuracies[:20]))
            model=make_interp_spline(x,accuracies[:20])
            xs = np.linspace(0,x[-1]+1,100)
            ys = model(xs)
            
            plt.grid()
            plt.ylim(0.5,1)
            plt.xlabel('epochs',fontsize = 16)
            plt.ylabel('accuracy',fontsize = 16)
            plt.title('accquary(first 4 epochs)',fontsize = 16)
            plt.plot(xs,ys,label = 'accuracies',color = 'b',linestyle = '--')
            plt.broken_barh([(0, 5)], (0, 1), facecolors='yellow',alpha = 0.2)
            plt.broken_barh([(5, 15)], (0, 1), facecolors='orange',alpha = 0.2)
            plt.annotate('without patch loss', (1, 0.95),fontsize = 20)
            plt.annotate('with patch loss', (11, 0.95),fontsize = 20)
            plt.show()
    return accuracies

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
    acc = re.findall('iou_acc:(\d.\d{4})',log)
    true_acc = re.findall('true_acc:(\d.\d{4})',log)
    acc  = np.asarray(acc).astype(np.float)
    true_acc  = np.asarray(true_acc).astype(np.float)

    if dis:
        x = np.arange(0,len(acc))
        x = np.stack((x,x))
        y = [acc,true_acc]
        lines = ['-','--']
        markers = ['o','v']
        labels = ['acc','true_acc']
        plt.grid()
        plt.ylim(0.0,1)
        plt.xlabel('epochs',fontsize = 16)
        plt.ylabel('accuracy',fontsize = 16)
        plt.title('accuracy',fontsize = 16)
        for i in range(2):
            if smooth:
                model=make_interp_spline(x[i],y[i])
                xs = np.linspace(0,x[i][-1]+1,100)
                ys = model(xs)
                markers = ['','']
            else:
                xs = x[i]
                ys = y[i]
            plt.plot(xs,ys,linestyle = lines[i],marker = markers[i],label = labels[i])
            plt.legend()
        plt.show()
    return (acc,true_acc)

filepath = '12.patch_accelerate/trained_models/patch/210911_1756/logger.log'

accuracies = visual_acc(filepath,smooth=False)