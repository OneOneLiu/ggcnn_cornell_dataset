'''
这个版本: 不冻结前面的参数,也不添加蒙板过滤,就看直接添加patch loss的影响
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T
import torchvision.transforms.functional as VisionF
import matplotlib.pyplot as plt
import numpy as np

from utils.dataset_processing.evaluation import get_edge, show_grasp
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
        x = self.features(x)
        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)

        filter = self.filter(x)

        return pos_output, cos_output, sin_output, width_output, filter
    def compute_loss(self, xc, yc, include_patch = 1):
        y_pos, y_cos, y_sin, y_width, mask_prob, y_height = yc
        pos_pred, cos_pred, sin_pred, width_pred, filter = self(xc, include_patch = include_patch)

        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)

        prob = filter

        prob_loss = F.mse_loss(prob,mask_prob)

        prediction = (pos_pred, cos_pred, sin_pred, width_pred)

        if include_patch:
            gt_patches, pre_patches = self.get_patch_no_dis(prediction, y_height)
            patch_loss = F.mse_loss(pre_patches, gt_patches)
            loss = p_loss + cos_loss + sin_loss + width_loss + prob_loss + patch_loss
        else:
            loss = p_loss + cos_loss + sin_loss + width_loss + prob_loss
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
            }
        }
    def get_patch_no_dis(self,prediction,mask_height):
        
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

        # 求得8张图中最亮点的中心坐标
        x = position % 300
        y = position // 300
        # 以其为中心,拓展成栅格矩阵
        steps = 49
        offset_x = torch.tensor([0,2,-2,4,-4,6,-6]).repeat(batch_size*7).cuda()
        offset_y = torch.tensor([0,2,-2,4,-4,6,-6]).repeat_interleave(7).repeat(batch_size).cuda()
        # 扩展x和y并与offset相加

        expand_x = x.repeat_interleave(steps) + offset_x
        expand_y = y.repeat_interleave(steps) + offset_y
        expand_x = torch.clip(expand_x,6,293)
        expand_y = torch.clip(expand_y,6,293)

        indice0 = torch.arange(0,batch_size).repeat_interleave(steps)
        indice1 = indice0.new_full((len(indice0),),0)

        # 索引获得每个位置对应的角度和宽度
        ang = ang_g[(indice0,indice1,expand_y,expand_x)]
        width = width_g[(indice0,indice1,expand_y,expand_x)]
        length = width / 2

        # 组合参数并分组,为后面碰撞检测做准备
        boxes_params = torch.stack((expand_y,expand_x,ang,width,length)).t().reshape(batch_size,-1,5)
        # 每张图要转25个不同的角度,然后裁剪出来做检测,一共八张图,也就是说要转200次,只转图像就行了,gr不用转,然后在这个中心来进行裁剪
        # 目前只能想到用for循环来做

        indexes, batch_edges, batch_edges_left, batch_edges_right, batch_edges_top, batch_edges_bottom,directions  = get_indexes(mask_height,boxes_params,batch_size,steps)

        # 对每一张图生成patch
        for i in range(batch_size):
            y = int(boxes_params[i][indexes[i]][0].cpu().data.numpy())
            x = int(boxes_params[i][indexes[i]][1].cpu().data.numpy())
            # 角度还是用位置更新之前的,因为更新后的位置角度是否合适没有经过验证的
            cos_patch = cos_img[i][0][y,x].cpu().data.numpy()
            sin_patch = sin_img[i][0][y,x].cpu().data.numpy()

            angle = ang_g[i][0][y,x].cpu().data.numpy()
            width_patch = width_img[i][0][y,x].cpu().data.numpy()

            selected_edge = batch_edges[i][indexes[i]]
            left_right_diff = abs(batch_edges_left[i][indexes[i]]-batch_edges_right[i][indexes[i]])
            top_bottom_diff = abs(batch_edges_top[i][indexes[i]]-batch_edges_bottom[i][indexes[i]])

            y,x,scale = get_patch_params(i,y,x,selected_edge,width_patch,angle,directions,left_right_diff,top_bottom_diff)

            left_margin = patch_size // 2
            right_margin = patch_size // 2 + 1
            # 先裁剪出原始的预测图
            pre_patch = torch.stack([prob[i][0][y-left_margin:y+right_margin,x-left_margin:x+right_margin],
                                    cos_img[i][0][y-left_margin:y+right_margin,x-left_margin:x+right_margin],
                                    sin_img[i][0][y-left_margin:y+right_margin,x-left_margin:x+right_margin],
                                    width_img[i][0][y-left_margin:y+right_margin,x-left_margin:x+right_margin]])
            
            gt_patch = torch.stack([pre_patch[0].new_full((patch_size,patch_size),1),
                                    pre_patch[1].new_full((patch_size,patch_size),float(cos_patch)),
                                    pre_patch[2].new_full((patch_size,patch_size),float(sin_patch)),
                                    pre_patch[3].new_full((patch_size,patch_size),float((width_patch*scale)))])
            # NOTE 调使用
            # image = mask_height[i]
            # index = indexes[i]
            # angle = boxes_params[i][index][2].cpu().data.numpy() 
            # width = boxes_params[i][index][3].cpu().data.numpy()
            # old_y = int(boxes_params[i][indexes[i]][0].cpu().data.numpy())
            # old_x = int(boxes_params[i][indexes[i]][1].cpu().data.numpy())
            # gr = Grasp((old_y,old_x),angle,width,width/2)
            # gr_img = show_grasp(image.cpu().data.numpy()[0],gr.as_gr,50)
            # plt.subplot(121)
            # plt.imshow(gr_img)
            # plt.subplot(122)
            # gr = Grasp((y,x),angle,width*scale,width*scale/2)
            # gr_img = show_grasp(image.cpu().data.numpy()[0],gr.as_gr,50)
            # plt.imshow(gr_img)
            # plt.show()
            # indexes, batch_edges, batch_edges_left, batch_edges_right, batch_edges_top, batch_edges_bottom,directions  = get_indexes(mask_height,boxes_params,batch_size,steps)
            # for j in range(4):
            #     plt.subplot(2,4,j+1)
            #     plt.imshow(pre_patch[j].cpu().data.numpy())
            # for j in range(4):
            #     plt.subplot(2,4,j+5)
            #     plt.imshow(gt_patch[j].cpu().data.numpy())
            # plt.show()
            # NOTE 调使用
            pre_patches.append(pre_patch)
            gt_patches.append(gt_patch)

        return torch.stack(pre_patches),torch.stack(gt_patches)
    
    def get_patch(self,prediction,mask_height):
        
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

        # 求得8张图中最亮点的中心坐标
        x = position % 300
        y = position // 300
        # 以其为中心,拓展成栅格矩阵
        steps = 49
        offset_x = torch.tensor([0,2,-2,4,-4,6,-6]).repeat(batch_size*7).cuda()
        offset_y = torch.tensor([0,2,-2,4,-4,6,-6]).repeat_interleave(7).repeat(batch_size).cuda()
        # 扩展x和y并与offset相加

        expand_x = x.repeat_interleave(steps) + offset_x
        expand_y = y.repeat_interleave(steps) + offset_y
        expand_x = torch.clip(expand_x,6,293)
        expand_y = torch.clip(expand_y,6,293)

        indice0 = torch.arange(0,batch_size).repeat_interleave(steps)
        indice1 = indice0.new_full((len(indice0),),0)

        # 索引获得每个位置对应的角度和宽度
        ang = ang_g[(indice0,indice1,expand_y,expand_x)]
        width = width_g[(indice0,indice1,expand_y,expand_x)]
        length = width / 2

        # 组合参数并分组,为后面碰撞检测做准备
        boxes_params = torch.stack((expand_y,expand_x,ang,width,length)).t().reshape(batch_size,-1,5)
        # 每张图要转25个不同的角度,然后裁剪出来做检测,一共八张图,也就是说要转200次,只转图像就行了,gr不用转,然后在这个中心来进行裁剪
        # 目前只能想到用for循环来做

        indexes, batch_edges, batch_edges_left, batch_edges_right, batch_edges_top, batch_edges_bottom,directions  = get_indexes(mask_height,boxes_params,batch_size,steps)

        # 对每一张图生成patch
        for i in range(batch_size):
            y = int(boxes_params[i][indexes[i]][0].cpu().data.numpy())
            x = int(boxes_params[i][indexes[i]][1].cpu().data.numpy())
            # 角度还是用位置更新之前的,因为更新后的位置角度是否合适没有经过验证的
            cos_patch = cos_img[i][0][y,x].cpu().data.numpy()
            sin_patch = sin_img[i][0][y,x].cpu().data.numpy()

            angle = ang_g[i][0][y,x].cpu().data.numpy()
            width_patch = width_img[i][0][y,x].cpu().data.numpy()

            selected_edge = batch_edges[i][indexes[i]]
            left_right_diff = abs(batch_edges_left[i][indexes[i]]-batch_edges_right[i][indexes[i]])
            top_bottom_diff = abs(batch_edges_top[i][indexes[i]]-batch_edges_bottom[i][indexes[i]])

            y,x,scale = get_patch_params(i,y,x,selected_edge,width_patch,angle,directions,left_right_diff,top_bottom_diff)

            # 先裁剪出原始的预测图
            pre_patch = torch.stack([prob[i][0][y-2:y+3,x-2:x+3],
                                    cos_img[i][0][y-2:y+3,x-2:x+3],
                                    sin_img[i][0][y-2:y+3,x-2:x+3],
                                    width_img[i][0][y-2:y+3,x-2:x+3]])
            
            gt_patch = torch.stack([pre_patch[0].new_full((5,5),1),
                                    pre_patch[1].new_full((5,5),float(cos_patch)),
                                    pre_patch[2].new_full((5,5),float(sin_patch)),
                                    pre_patch[3].new_full((5,5),float((width_patch*scale)))])
            # NOTE 调使用
            image = mask_height[i]
            index = indexes[i]
            angle = boxes_params[i][index][2].cpu().data.numpy() 
            width = boxes_params[i][index][3].cpu().data.numpy()
            old_y = int(boxes_params[i][indexes[i]][0].cpu().data.numpy())
            old_x = int(boxes_params[i][indexes[i]][1].cpu().data.numpy())
            gr = Grasp((old_y,old_x),angle,width,width/2)
            gr_img = show_grasp(image.cpu().data.numpy()[0],gr.as_gr,50)
            plt.subplot(121)
            plt.imshow(gr_img)
            plt.subplot(122)
            gr = Grasp((y,x),angle,width*scale,width*scale/2)
            gr_img = show_grasp(image.cpu().data.numpy()[0],gr.as_gr,50)
            plt.imshow(gr_img)
            plt.show()
            indexes, batch_edges, batch_edges_left, batch_edges_right, batch_edges_top, batch_edges_bottom,directions  = get_indexes(mask_height,boxes_params,batch_size,steps)
            for j in range(4):
                plt.subplot(2,4,j+1)
                plt.imshow(pre_patch[j].cpu().data.numpy())
            for j in range(4):
                plt.subplot(2,4,j+5)
                plt.imshow(gt_patch[j].cpu().data.numpy())
            plt.show()
            # NOTE 调使用
            pre_patches.append(pre_patch)
            gt_patches.append(gt_patch)

        return torch.stack(pre_patches),torch.stack(gt_patches)
def get_indexes(mask_height,boxes_params,batch_size,steps):
    pi = torch.as_tensor(np.pi)
    batch_edges = []
    batch_edges_left = []
    batch_edges_right = []
    batch_edges_top = []
    batch_edges_bottom = []
    for i in range(batch_size):
        img = mask_height[i]
        edges = []
        left_edges = []
        right_edges = []
        top_edges = []
        bottom_edges = []
        for j in range(steps):
            y = boxes_params[i][j][0]
            x = boxes_params[i][j][1]
            angle = boxes_params[i][j][2]
            width = boxes_params[i][j][3]
            length = boxes_params[i][j][4]
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