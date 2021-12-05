import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as VisionF
import numpy as np
from skimage.filters import gaussian

from box_intersection_2d import oriented_box_intersection_2d

import copy
import matplotlib.pyplot as plt
from functions import show_grasp
from grasp_pro import Grasp_cpaw, Grasps, Grasp
class GGCNN2(nn.Module):
    def __init__(self, input_channels=1, filter_sizes=None, l3_k_size=5, dilations=None):
        super().__init__()
        self.poto_alpha = 0.8
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

        self.max2d = MaxFiltering(16)
        self.filter = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
        
        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)

        filter = self.filter(self.max2d(x))

        return pos_output, cos_output, sin_output, width_output, filter

    def compute_loss(self, batch_inputs):
        xc, yc, poto_data = batch_inputs
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred, filter = self(xc)

        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)

        # 计算POTO损失相关
        prediction = (pos_pred, cos_pred, sin_pred, width_pred, filter)
        prob, gt_probs, probs, gt_patch, pre_patch, patch_flag= self.get_ground_truth(poto_data, prediction)
        if self.training:
            prob_loss = F.mse_loss(probs, gt_probs.float())
            if patch_flag == 0:
                loss = p_loss + cos_loss + sin_loss + width_loss + prob_loss
            else:
                patch_loss = F.mse_loss(pre_patch, gt_patch)
                loss = p_loss + cos_loss + sin_loss + width_loss + prob_loss + patch_loss
        else:
            loss = p_loss + cos_loss + sin_loss + width_loss
        return {
            'loss': loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': prob,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }
    
    def get_ground_truth(self,target,prediction):
        gt_patches = []
        pre_patches = []
        gt_probs = []
        probs = []
        # target = ((pos_grid, gt_num, gt_points, gt_angles, gt_widths, gt_lengths), mask, rgb, idxs)
        gt_data, gt_mask, gt_rgb, gt_mask_d, gt_mask_prob, idxs = target

        for pos_gt, gt_num, gt_boxes, gt_center, gt_angle, gt_width, gt_length, mask, rgb, mask_d1, mask_prob, idx, pos_per_image, cos_per_image, sin_per_image, width_per_image, filter in zip(*gt_data, gt_mask, gt_rgb, gt_mask_d, gt_mask_prob, idxs, *prediction):
            # 去除多余维度
            pos_per_image = pos_per_image.squeeze()
            cos_per_image = cos_per_image.squeeze()
            sin_per_image = sin_per_image.squeeze()
            width_per_image = width_per_image.squeeze()

            # 根据位置映射图及filter计算prob分数
            prob = pos_per_image * filter.squeeze().sigmoid()
            prob = torch.clip(prob,min = 0,max = 1)
            # 从预测图中反求出预测抓取            boxes, ang_img,width_img = img2boxes(cos_per_image, sin_per_image, width_per_image)


            # 获取mask信息
            mask = (mask/230).to(torch.uint8).to(pos_per_image.device)

            # 生成辅助采样栅格
            canvas = torch.zeros((300,300))
            total = torch.sum(mask).float()      # 获取mask总面积
            step = (torch.sqrt(total) // 10)     # 采样100个点就是sqrt(100) = 10,其他的自行设置
            row = torch.arange(0,300,max(step,2)).to(torch.int64).unique()
            col = torch.arange(0,300,max(step,2)).to(torch.int64).unique()
            xx,yy = torch.meshgrid(row,col)
            canvas[xx,yy] = 1.0
            
            # 转移设备, 计算最终mask时用
            canvas = canvas.to(pos_per_image.device)
            pos_gt = pos_gt.to(pos_per_image.device)

            # 找到mask及标注位置的索引值
            mask_g = (mask + pos_gt + canvas) // 2
            mask_index = torch.where(mask_g == 1)

            # 反求所有位置的抓取框
            centers, ang_img, width_img = img2boxes(cos_per_image, sin_per_image, width_per_image)

            # index = torch.where(mask_index == quality_index)
            # gt_index = gt_iou_index[index]
            # NOTE 调试用图像显示
            # plt.subplot(241)
            # plt.title(idx.numpy())
            # plt.imshow(rgb.numpy())
            # plt.subplot(242)
            # plt.title('grasp_label')
            # grs = Grasps()
            # grs.grs = [Grasp(points.cpu().data.numpy()) for points in gt_boxes]
            # img = show_grasp(copy.deepcopy(rgb.numpy()),grs)
            # plt.imshow(img)
            # plt.subplot(243)
            # plt.title('mask_d')
            # plt.imshow(mask_prob.cpu().data.numpy())
            # plt.subplot(244)
            # plt.title('mask_g')
            # plt.imshow(mask_g.cpu().data.numpy())
            # plt.subplot(245)
            # plt.title('pos_img')
            # plt.imshow(pos_per_image.cpu().data.numpy())
            # plt.subplot(246)
            # plt.title('prob')
            # plt.imshow(prob.cpu().data.numpy())
            # 显示prob获得的最优抓取
            # NOTE 调试用图像显示

            prob_n = gaussian(prob.cpu().data.numpy(), 2.0, preserve_range=True)
            ang_img_n = gaussian(ang_img.cpu().data.numpy(), 2.0, preserve_range=True)
            width_img_n = gaussian(width_img.cpu().data.numpy(), 1.0, preserve_range=True)
            # prob_n = prob.cpu().data.numpy()
            # ang_img_n = ang_img.cpu().data.numpy()
            # width_img_n = width_img.cpu().data.numpy()
            position = np.argmax(prob_n)
            x = (position % 300)
            y = (position // 300)
            prob_ang = ang_img_n[y, x]
            prob_width = width_img_n[y, x]
            grasp = Grasp_cpaw((y,x), prob_ang, prob_width/2, prob_width).as_gr
            
            # NOTE 调试用图像显示
            # img = show_grasp(copy.deepcopy(rgb.numpy()),Grasps([grasp]))
            # plt.subplot(247)
            # plt.title('iou')
            # plt.imshow(img)
            # plt.subplot(248)
            # plt.imshow(img)
            # plt.show()
            # NOTE 调试用图像显示

            # TODO:在这下面加一个抓取合理性检测模块
            #先检查一下这个区域
            mask_n = copy.deepcopy(mask_d1.cpu().data.numpy())
            y1 = y
            x1 = x
            edge = edge_check_new(torch.from_numpy(mask_n).cuda(),(grasp.center,grasp.angle,grasp.width,grasp.width/2))
            selected_width = edge[2]
            edge_widths = []
            edge_widths_left = []
            edge_widths_right = []
            coords = []

            if selected_width < 5:
                for i in range(-7,8,3):
                    for j in range(-7,8,3):
                        xo = min(x + i,296)
                        yo = min(y + j,296)
                        x1 = max(xo,3)
                        y1 = max(yo,3)

                        prob_ang = ang_img_n[y1, x1]
                        prob_width = width_img_n[y1, x1]
                        grasp = Grasp_cpaw((y1,x1),prob_ang,prob_width/2,prob_width).as_gr
                        coords.append((y1,x1))
                        mask_n = copy.deepcopy(mask_d1.cpu().data.numpy())
                        
                        
                        edge = edge_check_new(torch.from_numpy(mask_n).cuda(),(grasp.center,grasp.angle,grasp.width,grasp.width/2))
                        edge_widths_left.append(edge[0] * prob_width)
                        edge_widths_right.append(edge[1] * prob_width)
                        edge_widths.append(edge[2])

                        # NOTE for debug delete later
                        # plt.subplot(121)
                        # img = show_grasp(copy.deepcopy(mask_n),Grasps([grasp]))
                        # plt.title(edge)
                        # plt.imshow(img)
                        # plt.subplot(122)
                        # img = show_grasp(copy.deepcopy(rgb.numpy()),Grasps([grasp]))
                        # plt.title(edge)
                        # plt.imshow(img)
                        # plt.show()
                        # NOTE for debug delete later

                        if edge[2] * prob_width * 0.01 > 3:
                            break
                        # edge = edge_check_new(torch.from_numpy(mask_n).cuda(),(grasp.center,grasp.angle,grasp.width,grasp.width/2))
                        # print(1)
                    if edge[2] * prob_width * 0.01 > 3:
                            break
                index = np.argmax(edge_widths)
                # 如果找了一圈也没找到哪个无碰撞就找一个相对最好的
                if np.max(edge_widths) == 0:
                    edge_widths_lr = (edge_widths_left + edge_widths_right)
                    if max(edge_widths_lr) > 0:
                        index = np.argmax(edge_widths_lr)
                        if index >= len(edge_widths):
                            index = index - len(edge_widths)
                y1,x1 = coords[index]
                selected_width = edge_widths[index]

            # NOTE for debug delete later
            # prob_ang = ang_img_n[y1, x1]
            # prob_width = width_img_n[y1, x1]
            # grasp = Grasp_cpaw((y1,x1),prob_ang,prob_width/2,prob_width).as_gr
            # mask_n = copy.deepcopy(mask_d1.cpu().data.numpy())
            # edge = edge_check_new(torch.from_numpy(mask_n).cuda(),(grasp.center,grasp.angle,grasp.width,grasp.width/2))
            # img = show_grasp(copy.deepcopy(rgb.numpy()),Grasps([grasp]))
            # plt.imshow(img)
            # plt.show() # 显示最终选定的抓取
            # NOTE for debug delete later

            # 首先把这个抓取中心附近的区域给裁剪出来 
            cos = cos_per_image.cpu().data.numpy()[y1,x1]
            sin = sin_per_image.cpu().data.numpy()[y1,x1]
            width = width_per_image.cpu().data.numpy()[y1,x1]
            slice_y, slice_x = (y1,x1) # 抓取中心x,y前面已经求出来了
            pre_patch = torch.stack([pos_per_image[slice_y-2:slice_y+3,slice_x-2:slice_x+3],
                        cos_per_image[slice_y-2:slice_y+3,slice_x-2:slice_x+3],
                        sin_per_image[slice_y-2:slice_y+3,slice_x-2:slice_x+3],
                        width_per_image[slice_y-2:slice_y+3,slice_x-2:slice_x+3]])

            # 为其生成对应的优化参数
            scale = 1
            if selected_width * width * 1.50 < 5:
                scale = 1.1
            elif selected_width * width * 1.50 > 10:
                scale = 0.9
            else:
                gt_probs.append(mask_prob.to(mask.device))
                probs.append(prob)
                # 释放显存
                torch.cuda.empty_cache()
                continue
            gt_patch = torch.stack([pre_patch[0].new_full((5,5),1),
                                    pre_patch[1].new_full((5,5),float(cos)),
                                    pre_patch[2].new_full((5,5),float(sin)),
                                        pre_patch[3].new_full((5,5),float((width*scale)))])
            # for i in range(4):
            #     plt.subplot(241+i)
            #     plt.imshow(pre_patch[i].cpu().data.numpy())
            # for i in range(4):
            #     plt.subplot(241+4+i)
            #     plt.imshow(gt_patch[i].cpu().data.numpy())
            # plt.show()

            # plt.subplot(121)
            # plt.imshow(cos_per_image.cpu().data.numpy())
            # plt.subplot(122)
            # plt.imshow(sin_per_image.cpu().data.numpy())
            # plt.show()
            pre_patches.append(pre_patch)
            gt_patches.append(gt_patch)
            # TODO:在这上面进行定向的参数优化生成

            gt_probs.append(mask_prob.to(mask.device))
            probs.append(prob)
            # 释放显存
            torch.cuda.empty_cache()

        gt_p = torch.stack(gt_probs)
        p = torch.stack(probs)
        if len(pre_patches) > 0:
            try:
                pp = torch.stack(pre_patches)
                gp = torch.stack(gt_patches)
                patch_flag = 1
            except:
                pp = 0
                gp = 0
                patch_flag = 0
                pass
                print('堆叠错误')
        else:
            pp = 0
            gp = 0
            patch_flag = 0
        # 这里返回一个prob是validate的时候用的,只有batch = 1 才有意义
        return prob, gt_p, p, gp, pp, patch_flag
def edge_check(img,box):
    # rotate the img to make the grasp parallel with the horizon
    pi = torch.as_tensor(np.pi)
    center, angle, width, length = box
    top = int(center[0] - length / 2)
    left = int(center[1] - width / 2)
    rt_angle = -float((angle / pi * 180)[0])
    rectified_img = VisionF.rotate(img = img.view(1,1,300,300),angle = rt_angle,center = tuple(center[::-1]))
    # 假定这个地方存在目标,而width又小于5的话,就适当放大这个width,如果这位置能抓的话不可能这么小的
    if width < 5:
        new_center, new_angle, new_width = generate_new_params(center, angle, width,1)
        return 0
    # crop the image based on the center,width and length
    crop_img = VisionF.crop(rectified_img,top,left,int(length),int(width))
    # plt.imshow(crop_img[0][0].cpu().data.numpy())
    # plt.show()
    # resize the image to a certain size of (50,100)
    resized_img = VisionF.resize(crop_img,[50,100])
    # plt.imshow(resized_img[0][0].cpu().data.numpy())
    # plt.show()
    resized_img = resized_img[0][0].cpu().data.numpy()

    # 获取中间的高度
    img_middle = resized_img[:,10:90]
    flag_m,_,_,_,height_m = check_layers(img_middle,4200,middle = True)
    for edge_width in range(0,41)[::-1]:
        img_1 = resized_img[:,0:edge_width]
        img_2 = resized_img[:,100-edge_width:100]

        flag_1,_,_,_,height_1 = check_layers(img_1,50*edge_width,base_height = height_m)
        if not flag_1:# 1都存在碰撞的话直接continue,下一位
            continue
        flag_2,_,_,_,height_2 = check_layers(img_2,50*edge_width,base_height = height_m)

        if flag_1 and flag_2:
            break
    if flag_1 and flag_2:
        return edge_width
    return 0

def edge_check_new(img,box):
    # rotate the img to make the grasp parallel with the horizon
    pi = torch.as_tensor(np.pi)
    center, angle, width, length = box
    top = int(center[0] - length / 2)
    left = int(center[1] - width / 2)
    rt_angle = -float((angle / pi * 180)[0])
    rectified_img = VisionF.rotate(img = img.view(1,1,300,300),angle = rt_angle,center = tuple(center[::-1]))
    # 假定这个地方存在目标,而width又小于5的话,就适当放大这个width,如果这位置能抓的话不可能这么小的
    if width < 5:
        return 0,0,0
    # crop the image based on the center,width and length
    crop_img = VisionF.crop(rectified_img,top,left,int(length),int(width))
    # resize the image to a certain size of (50,100)
    resized_img = VisionF.resize(crop_img,[50,100])
    resized_img = resized_img[0][0].cpu().data.numpy()

    # 获取中间的高度
    img_middle = resized_img[:,10:90]
    flag_m,_,_,_,height_m = check_layers(img_middle,4200,middle = True)
    if flag_m: # 如果中间没有,那么就放大框,暂时也没别的办法
        return 0,0,0
    h1 = np.max(resized_img,axis = 0)

    edge1 = np.where(h1[0:40] > height_m - 10)
    if len(edge1[0]) == 0:
        edge_left = 40
    else:
        edge_left = edge1[0][0]
    
    edge2 = np.where(h1[60:100] > height_m - 10)
    if len(edge2[0]) == 0:
        edge_right = 40
    else:
        edge_right = 40-edge2[0][-1]-1

    if height_m < 10:
        edge1 = np.where(h1[0:40] > 0)
        if len(edge1[0]) == 0:
            edge_left = 40
        else:
            edge_left = edge1[0][0]
        edge2 = np.where(h1[60:100] > 0)
        if len(edge2[0]) == 0:
            edge_right = 40
        else:
            edge_right = 40-edge2[0][-1]-1

    edge = min(edge_left,edge_right)

    return edge_left,edge_right,edge

def collision_check(img,box):
    '''check whether the current grasp box is collision with the current object in the img
    Args:
        img (torch.Tensor): H,W
        box (tuple)       : (center,angle,width,length,left,top)
    Returns:
        flag (torch.Tensor) : there is collision -> flag= 0, otherwise, flag = 1
    '''
    # rotate the img to make the grasp parallel with the horizon
    pi = torch.as_tensor(np.pi)
    center, angle, width, length = box
    top = int(center[0] - length / 2)
    left = int(center[1] - width / 2)
    rt_angle = -float((angle / pi * 180)[0])
    rectified_img = VisionF.rotate(img = img.view(1,1,300,300),angle = rt_angle,center = tuple(center[::-1]))
    # 假定这个地方存在目标,而width又小于5的话,就适当放大这个width,如果这位置能抓的话不可能这么小的
    if width < 5:
        new_center, new_angle, new_width = generate_new_params(center, angle, width,1)
        return 1, new_center, new_angle, new_width
    # crop the image based on the center,width and length
    crop_img = VisionF.crop(rectified_img,top,left,int(length),int(width))
    # plt.imshow(crop_img[0][0].cpu().data.numpy())
    # plt.show()
    # resize the image to a certain size of (50,100)
    resized_img = VisionF.resize(crop_img,[50,100])
    # plt.imshow(resized_img[0][0].cpu().data.numpy())
    # plt.show()
    resized_img = resized_img[0][0].cpu().data.numpy()
    # crop and check the sub-images
    # 裁剪获得前10% 后10%和中间的图像
    edge_width = 12
    img_1 = resized_img[:,0:edge_width][0:25,:]
    img_2 = resized_img[:,0:edge_width][25:50,:]
    img_3 = resized_img[:,100-edge_width:100][0:25,:]
    img_4 = resized_img[:,100-edge_width:100][25:50,:]
    
    
    img_middle = resized_img[:,10:90]
    flag_m,_,_,_,height_m = check_layers(img_middle,4200,middle = True)
    if flag_m: # 如果中间没有,那么就放大框,暂时也没别的办法
        # plt.imshow(rectified_img[0][0].cpu().data.numpy())
        # plt.show()
        # plot_grid(img_middle,img_1,img_2,img_3,img_4)
        flag_m,_,_,_,height_m = check_layers(img_middle,4200,middle = True)
        new_center, new_angle, new_width = generate_new_params(center, angle, width,1)
        return 1, new_center, new_angle, new_width
    # print('中间区域的高度为{}'.format(height_m))
    flag_1,_,_,_,height_1 = check_layers(img_1,200,base_height = height_m)
    # print('区域1的高度为{}'.format(height_1))
    # if flag_1 == 1:
    #     print('区域1为无碰撞')
    flag_2,_,_,_,height_2 = check_layers(img_2,200,base_height = height_m)
    # print('区域2的高度为{}'.format(height_2))
    # if flag_2 == 1:
    #     print('区域2为无碰撞')
    flag_3,_,_,_,height_3 = check_layers(img_3,200,base_height = height_m)
    # print('区域3的高度为{}'.format(height_3))
    # if flag_3 == 1:
    #     print('区域3为无碰撞')
    flag_4,_,_,_,height_4 = check_layers(img_4,200,base_height = height_m)
    # print('区域4的高度为{}'.format(height_4))
    # if flag_4 == 1:
    #     print('区域4为无碰撞')
    # if flag_1 and flag_2 and flag_3 and flag_4 and not flag_m: NOTE 这里有个小bug
    if flag_1 and flag_2 and flag_3 and flag_4:
        # print('合理抓取')
        return 0, center, angle, width
    else:
        status = str(flag_1) + str(flag_2) + str(flag_3) + str(flag_4)
        # print('不合理抓取')
        flag = correct[status]
        # 根据不同的flag来对参数做不同的修正.
        # new_center, new_angle, new_width = generate_new_params(center, angle, width,flag)
        return correct[status], center, angle, width
def generate_new_params(center, angle, width, flag):
    '''genrate new rectangle parameters based on the current parameters and the flag
    Args:
        center (numpy.ndarray): shape = (2,1), [y,x]
        angle  (numpy.ndarray): float,angle between rectangle and horizon, counter-clockwise is positive
        width  (numpy.ndarray): 
        flag   (int)          : flag that indicates the changes which should be conducted
    Returns:
        new_center (tuple) : new_center of the grasp,(y,x)
        new_angle
        new_width
    '''
    if flag == 1: # 四边都有,应当放大
        new_x = center[1]
        new_y = center[0]
        new_angle = angle
        new_width = width * 1.2 # enlarge scale which can be adjusted based on the experiment result
    elif flag == 2: # 应当顺时针转 NOTE 这里的旋转移动等步幅还是设置小一点,因为万一整过火了,来回震荡,永远都不会收敛
        new_x = center[1]
        new_y = center[0]
        new_angle = angle - 0.2 # rotate angle which can be adjusted based on the experiment result
        new_width = width
    elif flag == 3: # 应当逆时针转
        new_x = center[1]
        new_y = center[0]
        new_angle = angle + 0.2 # rotate angle which can be adjusted based on the experiment result
        new_width = width
    elif flag == 4: # 应当向右移动
        delta_x = 3 * np.cos(angle)
        delta_y = -3 * np.sin(angle)
        # print('应当向右移动')
        new_x = center[1] + delta_x
        new_y = center[0] + delta_y # move distance which can be adjusted based on the experiment result
        new_angle = angle
        new_width = width * 1.1
    
    elif flag == 5: # 应当向左移动
        delta_x = -3 * np.cos(angle)
        delta_y = 3 * np.sin(angle)
        # print('应当向左移动')
        new_x = center[1] + delta_x
        new_y = center[0] + delta_y # move distance which can be adjusted based on the experiment result
        new_angle = angle
        new_width = width * 1.1
    elif flag == 6: # 应当向下移动
        delta_x = 3 * np.cos(angle + np.pi/2)
        delta_y = -3 * np.sin(angle + np.pi/2)
        # print('应当向下移动')
        new_x = center[1] + delta_x
        new_y = center[0] + delta_y # move distance which can be adjusted based on the experiment result
        new_angle = angle
        new_width = width * 1.1
    elif flag == 7: # 应当向上移动
        delta_x = -3 * np.cos(angle + np.pi/2)
        delta_y = 3 * np.sin(angle + np.pi/2)
        # print('应当向上移动')
        new_x = center[1] + delta_x
        new_y = center[0] + delta_y # move distance which can be adjusted based on the experiment result
        new_angle = angle
        new_width = width * 1.1

    return (int(new_y),int(new_x)),new_angle,new_width
def img2boxes(cos_img,sin_img,width_img):
    # 区别于function里面的两个后处理,这里不会使用Gaussian过滤,也不会挑出极大值,而是会将所有的90000个预测全部返回
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0)
    width_img = (width_img * 150.0)

    # now let's start calculate
    # 1. generate global position for every points
    x = torch.arange(0,300).cuda()
    y = torch.arange(0,300).cuda()

    shift_x, shift_y = torch.meshgrid(x,y)
    centers = torch.stack((shift_x.flatten(),shift_y.flatten())).t().reshape(300,300,2)

    return centers,ang_img.reshape(300,300,1),width_img.reshape(300,300,1)

def pairwise_iou(gt_centers, gt_angles, gt_widths, gt_lengths, centers, angles, widths):
    # 根据给定的数据生成boxes
    gt_centers_xy = gt_centers[:,[1,0]].contiguous()
    gt_box_params = torch.cat((gt_centers_xy,gt_widths,gt_lengths,-gt_angles),dim = 1)
    box_params = torch.cat((centers,widths,widths/2,-angles),dim = 1)

    gt_boxes = box2corners_th(torch.unsqueeze(gt_box_params,0)).squeeze()
    boxes = box2corners_th(torch.unsqueeze(box_params,0)).squeeze()
    # 先将gt_angle及预测angle推广复制到同一尺寸,计算角度差异,生成的新图中行号代表pre,列号代表gt
    angles_mat = angles.repeat_interleave(gt_angles.shape[0],dim = 1)
    gt_angles_mat = gt_angles.repeat_interleave(angles.shape[0],dim = 1).t()
    angle_diff = (angles_mat - gt_angles_mat)
    # 取出角度差异小与0.3的位置索引
    angle_mask_tuple = torch.where(torch.abs(angle_diff) < 0.3)

    # angle_index_1 = torch.unique(angle_mask_tuple[0])
    # 获取到与预测存在角度近似的gt编号
    angle_index_2 = torch.unique(angle_mask_tuple[1])

    # 如果一个合适的gt都没有提取到,就返回 0
    if len(angle_index_2) == 0:
        return 0, 0, 0
    # 根据编号提取对应的gt
    new_gt_boxes = gt_boxes[angle_index_2]
    # 提取到了之后,生成一一对应的计算box
    box1 = new_gt_boxes.repeat([boxes.shape[0],1,1]).reshape(boxes.shape[0],-1,4,2)
    box2 = boxes.repeat([1,new_gt_boxes.shape[0],1]).reshape(box1.shape)

    # 计算两框的相交部分面积
    try:
        area, _ = oriented_box_intersection_2d(box1, box2)
    except Exception as e: # 如果采样不合理,此处易报显存错误,返回2
        print(e)
        return 2, 0, 0
    area = area.view(boxes.shape[0],new_gt_boxes.shape[0])
    # 生成原始数量(因为前面gt过滤过)的空白intersection图
    inter = torch.zeros((boxes.shape[0],gt_boxes.shape[0]),device = angles.device)
    # 下面开始执行where的逆运算,把求出来的还原到真实的位置
    inter[:,angle_index_2] = area

    # 将gt宽度,长度,预测宽度推广到复制同一尺寸,用于计算某两个框的union面积
    widths_mat = widths.repeat_interleave(gt_widths.shape[0],dim = 1)
    gt_widths_mat = gt_widths.repeat_interleave(widths.shape[0],dim = 1).t()
    gt_length_mat = gt_lengths.repeat_interleave(widths.shape[0],dim = 1).t()

    # 计算面积
    gt_area = gt_widths_mat * gt_length_mat
    pre_area = widths_mat ** 2 / 2 # 预测的没有length 按照width的一半算
    u_area = gt_area + pre_area

    # 计算iou
    iou = inter / (u_area-inter) + 0.01
    # 对iou取最大值,对每个pre框都选择一个iou最大的gt框,并返回iou值和对应gt索引
    iou_max, iou_index = torch.max(iou,dim = 1)

    return 1, iou_max, iou_index

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
    
def box2corners_th(box:torch.Tensor)-> torch.Tensor:
    """convert box coordinate to corners

    Args:
        box (torch.Tensor): (B, N, 5) with x, y, w, h, alpha

    Returns:
        torch.Tensor: (B, N, 4, 2) corners
    """
    B = box.size()[0]
    x = box[..., 0:1]
    y = box[..., 1:2]
    w = box[..., 2:3]
    h = box[..., 3:4]
    alpha = box[..., 4:5] # (B, N, 1)
    x4 = torch.FloatTensor([0.5, -0.5, -0.5, 0.5]).unsqueeze(0).unsqueeze(0).to(box.device) # (1,1,4)
    x4 = x4 * w     # (B, N, 4)
    y4 = torch.FloatTensor([0.5, 0.5, -0.5, -0.5]).unsqueeze(0).unsqueeze(0).to(box.device)
    y4 = y4 * h     # (B, N, 4)
    corners = torch.stack([x4, y4], dim=-1)     # (B, N, 4, 2)
    sin = torch.sin(alpha)
    cos = torch.cos(alpha)
    row1 = torch.cat([cos, sin], dim=-1)
    row2 = torch.cat([-sin, cos], dim=-1)       # (B, N, 2)
    rot_T = torch.stack([row1, row2], dim=-2)   # (B, N, 2, 2)
    rotated = torch.bmm(corners.view([-1,4,2]), rot_T.view([-1,2,2]))
    rotated = rotated.view([B,-1,4,2])          # (B*N, 4, 2) -> (B, N, 4, 2)
    rotated[..., 0] += x
    rotated[..., 1] += y
    return rotated

def plot_grid(img_middle,img_1,img_2,img_3,img_4):
    grid = plt.GridSpec(10, 10, wspace=0.5, hspace=0.5)
    plt.figure(figsize=(15,10))

    middle = plt.subplot(grid[0:4,1:8])
    plt.title('middle')
    plt.imshow(img_middle)

    left_1 = plt.subplot(grid[0:2,0])
    plt.title('1')
    plt.imshow(img_1)
    left_2 = plt.subplot(grid[2:4,0])
    plt.title('2')
    plt.imshow(img_2)

    right_1 = plt.subplot(grid[0:2,9])
    plt.title('3')
    plt.imshow(img_3)
    right_1 = plt.subplot(grid[2:4,9])
    plt.title('4')
    plt.imshow(img_4)
    plt.show()

def check_layers(img,total_img,base_height = None,middle = False):
    flag = 0
    array = img.ravel()[np.flatnonzero(img)]
    weights = np.ones_like(array.ravel())/float(len(array.ravel()))
    if len(array) == 0:
        flag = 1
        return flag,[],[],[],0
    bins = 6
    if max(array)-min(array) < 2:
        bins = 1
    # n,b,_ = plt.hist(array,bins = bins,weights=weights)
    n,b = np.histogram(array,bins = bins,weights=weights)
    # 这个上面的bins是个可以调的参数,其实不用特别多,前面一开始是10,现在改成6了
    # plt.show()
    ranges = []
    probs = []
    mean_heights = []
    threshold_1 = 0.10
    threshold_2 = 0.05
    # need to detect smaller height edge in the middle
    if middle:
        threshold_1 = 0.00
        threshold_2 = 0.00
    # offset参数是为了避免边界在两边都取不到的情况.
    offset = 0
    for index in np.where(n > threshold_1)[0]:
        ran = b[index:index+2]
        result = [i for i in array if i>ran[0] + offset and i <=ran[1]]
        if result == []:
            offset = -0.1
            continue
        mean_height = np.mean(result)
        if base_height:
            low_layer = base_height * 0.3
            if mean_height < low_layer:# 这个是用来剔除落点中的一些小边缘干扰平面
                continue
        prob = len(result)/total_img
        if prob < threshold_2:# 这个是用来去除一些小的噪点
            # print(prob)
            continue
        if mean_height < array.max() // 4:# 高度太小的面不考虑:
            continue
        ranges.append(ran)
        probs.append(prob)# 计算实际占比
        mean_heights.append(mean_height)
    
    # 如果一个符合条件的大面都找不到,说明该区域基本为空
    if len(mean_heights) == 0:
        return 1,ranges,probs,mean_heights,0
    # 这个是用来跟中间的高度相比比较的
    if base_height:
        if base_height - max(mean_heights) > 10: # 
            flag = 1
    # 如果是中间区域的话,返回高度最高但占比同样超过一定阈值的,否则就返回占比最高的
    height = max(mean_heights)
    if middle:
        # 如果这个最高的占比太小,就返回占比最高的那个
        if probs[mean_heights.index(height)] < 0.1:
            height = mean_heights[probs.index(max(probs))]
    return flag,ranges,probs,mean_heights,height

# 抓取修正映射表
correct = { '0000':1,  # 四边都有,应当放大
            '0001':2,  # 123有,应当顺时针转
            '0100':2,  # 134有,应当顺时针转
            '1000':3,  # 234有,应当逆时针转
            '0010':3,  # 124有,应当逆时针转
            '1001':2,  # 23有,应当顺时针转
            '0110':3,  # 14有,应当逆时针转
            '1100':4,  # 34有,应当向右移动 NOTE 移动
            '0011':5,  # 12有,应当向左移动 NOTE 移动
            '1010':6,  # 24有,应当向下移动 NOTE 移动
            '0101':7,  # 13有,应当向上移动 NOTE 移动
            '0111':3,  # 1有,应当逆时针转
            '1011':2,  # 2有,应当顺时针转
            '1101':2,  # 3有,应当顺时针转
            '1110':3,  # 4有,应当逆时针转
}
