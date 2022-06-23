'''
这个版本的主要修改在于计算filter之后的损失时添加了一个蒙板过滤,消除标注映射图对于未标注区域的影响.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import random

import torchvision.transforms as T
import torchvision.transforms.functional as VisionF
import matplotlib.pyplot as plt
import numpy as np

from utils.dataset_processing.evaluation import get_edge
from utils.dataset_processing.grasp import Grasp

patch_size = 5

class GGCNN2(nn.Module):
    def __init__(self, input_channels=1, filter_sizes=None, l3_k_size=5, dilations=None):
        super().__init__()

        if filter_sizes is None:
            filter_sizes = [16,  # First set of convs
                            16,  # Second set of convs
                            32,  # Dilated convs
                            16]  # Transpose Convs

        if dilations is None:
            dilations = [2, 4]

        self.features = nn.Sequential(
            # 4 conv layers.
            nn.Conv2d(input_channels, filter_sizes[0], kernel_size=11, stride=1, padding=5, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[0], filter_sizes[0], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[1], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Dilated convolutions.
            nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_size=l3_k_size, dilation=dilations[0], stride=1, padding=(l3_k_size//2 * dilations[0]), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[2], filter_sizes[2], kernel_size=l3_k_size, dilation=dilations[1], stride=1, padding=(l3_k_size//2 * dilations[1]), bias=True),
            nn.ReLU(inplace=True),

            # Output layers
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filter_sizes[2], filter_sizes[3], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filter_sizes[3], filter_sizes[3], 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pos_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.cos_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.sin_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.width_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)

        self.filter = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.filter_cos = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.filter_sin = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.filter_width = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x, include_patch = 1):
        if include_patch:
            with torch.no_grad():
                x = self.features(x)
        else:
            x = self.features(x)
        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)

        filter = self.filter(x)
        filter_cos = self.filter_cos(x)
        filter_sin = self.filter_sin(x)
        filter_width = self.filter_width(x)

        return pos_output, cos_output, sin_output, width_output, filter, filter_cos, filter_sin, filter_width

    def compute_loss(self, xc, yc, include_patch = 1):
        y_pos, y_cos, y_sin, y_width, mask_prob, y_height = yc
        pos_pred, cos_pred, sin_pred, width_pred, filter, filter_cos, filter_sin, filter_width = self(xc, include_patch = include_patch)

        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)

        prob = filter

        prob_loss = F.mse_loss(prob,mask_prob)

        filtered_cos = filter_cos.sigmoid() * cos_pred
        filtered_sin = filter_sin.sigmoid() * sin_pred
        filtered_width = filter_width.sigmoid() * width_pred

        cos_loss1 = F.mse_loss(filtered_cos * y_pos, y_cos * y_pos)
        sin_loss1 = F.mse_loss(filtered_sin * y_pos, y_sin * y_pos)
        width_loss1 = F.mse_loss(filtered_width * y_pos, y_width * y_pos)

        prediction = (prob,filtered_cos,filtered_sin,filtered_width)

        if include_patch:
            gt_patches, pre_patches = self.get_patch_no_dis(prediction, y_height,mask_prob)
            patch_loss = F.mse_loss(pre_patches, gt_patches)
            loss = p_loss + 0.3*(cos_loss + sin_loss + width_loss + prob_loss + cos_loss1 + sin_loss1 + width_loss1) + patch_loss
        else:
            loss = p_loss + cos_loss + sin_loss + width_loss + prob_loss + cos_loss1 + sin_loss1 + width_loss1
        return {
            'loss': loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred,
                'prob' : prob,
                'filtered_cos': filtered_cos,
                'filtered_sin': filtered_sin,
                'filtered_width': filtered_width
            }
        }
    def get_patch_no_dis(self,prediction,mask_height,mask_prob):
        gt_patches = []
        pre_patches = []
        prob, cos_img, sin_img, width_img = prediction
        # 获取batch size,后面要用
        batch_size = prob.size()[0]

        ang_out = (torch.atan2(sin_img, cos_img) / 2.0)
        width_out= width_img * 150
        
        prob_g = T.GaussianBlur(kernel_size = (3,3),sigma = 2)(prob)
        ang_g = T.GaussianBlur(kernel_size = (3,3),sigma = 2)(ang_out)
        width_g = T.GaussianBlur(kernel_size = (3,3),sigma = 1)(width_out)

        position = torch.max(prob_g.reshape(-1,90000),dim = 1)[1]

        # # 求得8张图中最亮点的中心坐标
        x = position % 300
        y = position // 300
        # NOTE:multi-patch主要修改的就是这一部分,原先只是在最亮点进行优化,现在是对整个图进行优化
        
        # 生成间距为s的矩阵栅格
        # 可视化看一下
        for i in range(batch_size):
            # 根据目标对象大小确定栅格间距
            # mask_height_n = mask_height[i][0].cpu().data.numpy()
            # s = max(1,int(np.count_nonzero(mask_height_n)/400))
            # s = min(10,s)
            # grid_x = np.arange(0,300,s)
            # grid_y = np.arange(0,300,s)
            # grid_X,grid_Y = np.meshgrid(grid_x,grid_y)
            # canavas = torch.zeros(300,300)
            # canavas[grid_X,grid_Y] = 1
            # grids = canavas.numpy() * mask_height_n
            # grids[grids > 0] = 1
            # num = np.count_nonzero(grids)
            # # 提取非零位置的行列坐标
            # xs,ys = np.where(grids>0)
            # # 随机选取n个点
            n = 3
            # try:
            #     coor_indexes = random.sample(range(0,len(ys)),n)
            # except Exception as e:
            #     print(e)
            #     pass
            # x = torch.from_numpy(xs[coor_indexes]).cuda()
            # y = torch.from_numpy(ys[coor_indexes]).cuda()
            # # # NOTE 可视化用
            # selected_coors = np.stack((xs[coor_indexes],ys[coor_indexes]), axis=1)
            # grids[xs[coor_indexes],ys[coor_indexes]] = 2
            # plt.subplot(1,4,1)
            # plt.imshow(grids)
            # plt.title(str(num))
            # plt.subplot(1,4,2)
            # plt.imshow(mask_height[i][0].cpu().data.numpy())
            # plt.subplot(1,4,3)
            # mask_prob_i = mask_prob[i][0].cpu().data.numpy()
            # mean_prob = np.mean(mask_prob_i[mask_prob_i>0]) * 1.1
            # mask_prob_i[mask_prob_i<mean_prob] = 0
            # plt.imshow(mask_prob_i)
            # plt.subplot(1,4,4)
            # grids = mask_prob_i
            # xs,ys = np.where(grids>0)
            # try:
            #     coor_indexes = random.sample(range(0,len(ys)),n)
            # except Exception as e:
            #     print(e)
            #     pass
            # x = torch.from_numpy(xs[coor_indexes]).cuda()
            # y = torch.from_numpy(ys[coor_indexes]).cuda()
            # # # NOTE 可视化用
            # selected_coors = np.stack((xs[coor_indexes],ys[coor_indexes]), axis=1)
            # grids[xs[coor_indexes],ys[coor_indexes]] = 2
            # plt.imshow(grids)
            # plt.axis('off')
            # plt.show()

            mask_prob_i = mask_prob[i][0].cpu().data.numpy()
            mean_prob = np.mean(mask_prob_i[mask_prob_i>0]) * 1.1
            mask_prob_i[mask_prob_i<mean_prob] = 0
            grids = mask_prob_i
            xs,ys = np.where(grids>0)
            try:
                coor_indexes = random.sample(range(0,len(ys)),n)
            except Exception as e:
                print(e)
                continue
            x = torch.from_numpy(xs[coor_indexes]).cuda()
            y = torch.from_numpy(ys[coor_indexes]).cuda()
            # 以上面所选的点为中心,拓展成w*w的栅格矩阵
            # 下面这些都是在构造这个栅格矩阵中点的坐标
            

            w = 7
            steps = w*w
            offset_x = torch.tensor([0,2,-2,4,-4,6,-6]).repeat(n*w).cuda()
            offset_y = torch.tensor([0,2,-2,4,-4,6,-6]).repeat_interleave(w).repeat(n).cuda()
            # 扩展x和y并与offset相加,hu
            expand_x = x.repeat_interleave(steps) + offset_x
            expand_y = y.repeat_interleave(steps) + offset_y
            expand_x = torch.clip(expand_x,w,300-w-1)
            expand_y = torch.clip(expand_y,w,300-w-1)

            indice0 = (torch.ones(w*w*n)*i).type(torch.long)
            indice1 = (torch.zeros(w*w*n)).type(torch.long)

            # 索引获得每个位置对应的角度和宽度
            ang = ang_g[(indice0,indice1,expand_y,expand_x)]
            width = width_g[(indice0,indice1,expand_y,expand_x)]
            length = width / 2
            # prob = mask_prob[(indice0,indice1,expand_y,expand_x)]

            # 组合参数并分组,为后面碰撞检测做准备
            boxes_params = torch.stack((expand_y,expand_x,ang,width,length)).t().reshape(n,-1,5)
            # 每张图要转25个不同的角度,然后裁剪出来做检测,一共八张图,也就是说要转200次,只转图像就行了,gr不用转,然后在这个中心来进行裁剪
            # 目前只能想到用for循环来做

            indexes, batch_edges, batch_edges_left, batch_edges_right, batch_edges_top, batch_edges_bottom,directions  = get_indexes(mask_height[i],boxes_params,n,steps)

            # 对每一个点生成patch
            for m in range(n):
                y = int(boxes_params[m][indexes[m]][0].cpu().data.numpy())
                x = int(boxes_params[m][indexes[m]][1].cpu().data.numpy())
                # 角度还是用位置更新之前的,因为更新后的位置角度是否合适没有经过验证的
                cos_patch = cos_img[i][0][y,x].cpu().data.numpy()
                sin_patch = sin_img[i][0][y,x].cpu().data.numpy()
                prob_patch = mask_prob[i][0][y,x].cpu().data.numpy()
                angle = ang_g[i][0][y,x].cpu().data.numpy()
                width_patch = width_img[i][0][y,x].cpu().data.numpy()

                selected_edge = batch_edges[m][indexes[m]]
                left_right_diff = abs(batch_edges_left[m][indexes[m]]-batch_edges_right[m][indexes[m]])
                top_bottom_diff = abs(batch_edges_top[m][indexes[m]]-batch_edges_bottom[m][indexes[m]])

                y,x,scale = get_patch_params(m,y,x,selected_edge,width_patch,angle,directions,left_right_diff,top_bottom_diff)

                left_margin = patch_size // 2
                right_margin = patch_size // 2 + 1
                # 先裁剪出原始的预测图
                pre_patch = torch.stack([prob[i][0][y-left_margin:y+right_margin,x-left_margin:x+right_margin],
                                        cos_img[i][0][y-left_margin:y+right_margin,x-left_margin:x+right_margin],
                                        sin_img[i][0][y-left_margin:y+right_margin,x-left_margin:x+right_margin],
                                        width_img[i][0][y-left_margin:y+right_margin,x-left_margin:x+right_margin]])
                
                gt_patch = torch.stack([pre_patch[0].new_full((patch_size,patch_size),float(prob_patch)),
                                        pre_patch[1].new_full((patch_size,patch_size),float(cos_patch)),
                                        pre_patch[2].new_full((patch_size,patch_size),float(sin_patch)),
                                        pre_patch[3].new_full((patch_size,patch_size),float((width_patch*scale)))])
                pre_patches.append(pre_patch)
                gt_patches.append(gt_patch)

        return torch.stack(pre_patches),torch.stack(gt_patches)

def get_indexes(mask_height,boxes_params,n,steps):
    pi = torch.as_tensor(np.pi)
    batch_edges = []
    batch_edges_left = []
    batch_edges_right = []
    batch_edges_top = []
    batch_edges_bottom = []
    img = mask_height
    for m in range(n):
        edges = []
        left_edges = []
        right_edges = []
        top_edges = []
        bottom_edges = []
        for j in range(steps):
            y = boxes_params[m][j][0]
            x = boxes_params[m][j][1]
            angle = boxes_params[m][j][2]
            width = boxes_params[m][j][3]
            length = boxes_params[m][j][4]
            # 如果宽度过小,就直接视为不可行
            if width < 5:
                edges.append(0)
                left_edges.append(0)
                right_edges.append(0)
                top_edges.append(0)
                bottom_edges.append(0)
                continue
            top = int(y - length / 2)
            left = int(x - width / 2)
            rt_angle = -float((angle / pi *180))

            rectified_img = VisionF.rotate(img = img.view(1,1,300,300),angle = rt_angle,center = (x,y))

            crop_img = VisionF.crop(rectified_img,top,left,int(length),int(width))

            resized_img = VisionF.resize(crop_img,[50,100]).squeeze().cpu().data.numpy()
            
            # 获取图像各边缘宽度
            edge,edge_left,edge_right,edge_top,edge_bottom = get_edge(resized_img)

            edges.append(edge)
            left_edges.append(edge_left)
            right_edges.append(edge_right)
            top_edges.append(edge_top)
            bottom_edges.append(edge_bottom)
            if edge * width / 100 > 3:
                break
            # 如果这是第一个,且存在无碰撞区域,那就不往后找了,针对这个做优化就行了,跟原来的思路一样,相当于先检查一次
            if j == 1 and edge > 0:
                break
        batch_edges.append(edges)
        batch_edges_left.append(left_edges)
        batch_edges_right.append(right_edges)
        batch_edges_top.append(top_edges)
        batch_edges_bottom.append(bottom_edges)

    # 从里面确定每张图最优的参数
    indexes = []
    # 用来表征位置优化方向的state表
    directions = []
    for edges,left_edges,right_edges,top_edges,bottom_edges in zip(batch_edges,batch_edges_left,batch_edges_right,batch_edges_top,batch_edges_bottom):
        index = np.argmax(edges)
        if np.max(edges) == 0:
            edges_lr = (left_edges + right_edges)
            if max(edges_lr) > 0:
                index = np.argmax(edges_lr)
                if index >= len(edges):
                    index = index - len(edges)
        indexes.append(index)
        direction = 0
        # 通过比较各边的边缘宽度来判定位置优化的方向
        edge_range = max(left_edges[index],right_edges[index])
        if abs(left_edges[index] - right_edges[index]) > edge_range//2:
            direction = 1 if left_edges[index] > right_edges[index] else 2
        edge_range = max(top_edges[index],bottom_edges[index])
        if abs(top_edges[index] - bottom_edges[index]) > edge_range//2:
            direction = 3 if top_edges[index] > bottom_edges[index] else 4
        
        directions.append(direction)
    return indexes, batch_edges, batch_edges_left, batch_edges_right, batch_edges_top, batch_edges_bottom,directions

def get_patch_params(i,old_y,old_x,selected_edge,width_patch,angle,directions,left_right_diff,top_bottom_diff):
    # 为其生成对应的优化参数
    scale = 1
    if selected_edge * width_patch * 1.50 < 3:
        scale = 1.2
    elif selected_edge * width_patch * 1.50 < 7:
        scale = 1.1
    elif selected_edge * width_patch * 1.50 > 20:
        scale = 0.8
    elif selected_edge * width_patch * 1.50 > 15:
        scale = 0.9
    y = old_y
    x = old_x
    # 通过direction优化y,x位置
    if directions[i] == 1: # 向右移
        step = left_right_diff * width_patch *1.50 * 0.5  # 移动宽度的0.5倍
        delta_x = step * np.cos(angle) 
        delta_y = -step * np.sin(angle)
        y = int(old_y + delta_y)
        x = int(old_x + delta_x)
    elif directions[i] == 2: # 向左移
        step = left_right_diff * width_patch *1.50 * 0.5  # 移动宽度的0.5倍
        delta_x = -step * np.cos(angle)
        delta_y = step * np.sin(angle)
        y = int(old_y + delta_y)
        x = int(old_x + delta_x)
    elif directions[i] == 3: # 向下移
        step = top_bottom_diff * width_patch *1.50 * 0.5  # 移动宽度的0.5倍
        delta_x = -step * np.cos(angle + np.pi / 2)
        delta_y = step * np.sin(angle + np.pi / 2)
        y = int(old_y + delta_y)
        x = int(old_x + delta_x)
    elif directions[i] == 4: # 向上移
        step = top_bottom_diff * width_patch *1.50 * 0.5  # 移动宽度的0.5倍
        delta_x = step * np.cos(angle + np.pi / 2)
        delta_y = -step * np.sin(angle + np.pi / 2)
        y = int(old_y + delta_y)
        x = int(old_x + delta_x)
    y = min(300-patch_size//2 - 1,y)
    y = max(0+patch_size//2 + 1,y)
    x = min(300-patch_size//2 - 1,x)
    x = max(0+patch_size//2 + 1,x)

    return y,x,scale

class MaxFiltering(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3):
        super().__init__()
        self.convolutions = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm = nn.GroupNorm(8, in_channels)
        self.nonlinear = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=(kernel_size, kernel_size),stride=1,padding = 1)

    def forward(self, inputs):
        features = self.convolutions(inputs)
        max_pool = self.max_pool(features)
        output = max_pool+inputs
        output = self.nonlinear(self.norm(output))

        return output