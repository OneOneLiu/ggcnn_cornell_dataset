# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 21:52:23 2020
validate部分用到的一些函数汇总
@author: LiuDahui
"""
import torch
import cv2
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.draw import polygon
import numpy as np


from grasp_pro import Grasp_cpaw,Grasps
from image_pro import Image

def post_process(pos_img,cos_img,sin_img,width_img):
    '''
    :功能           :对原始的网络输出进行预处理，包括求解角度数据和高斯滤波
    :参数 pos_img   :cuda tensor,原始输出的抓取位置映射图
    :参数 cos_img   :cuda tensor,原始输出的抓取角度cos值映射图
    :参数 sin_img   :cuda tensor,原始输出的抓取角度sin值映射图
    :参数 wid_img   :cuda tensor,原始输出的抓取宽度值映射图
    :返回           :3个ndarray，分别是抓取质量（位置）映射图，抓取角度映射图以及抓取宽度映射图
    '''
    q_img = pos_img.cpu().data.numpy().squeeze()
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().data.numpy().squeeze()
    width_img = width_img.cpu().data.numpy().squeeze() * 150.0
    
    #一定注意，此处Guassian滤波时，batch_size一定要是1才行，这个滤波函数不支持一次输入多个样本
    q_img_g = gaussian(q_img, 2.0, preserve_range=True)
    ang_img_g = gaussian(ang_img, 2.0, preserve_range=True)
    width_img_g = gaussian(width_img, 1.0, preserve_range=True)
    
    return q_img_g, ang_img_g, width_img_g

def detect_grasps(q_out,ang_out,wid_out = None,no_grasp = 1):
    '''
    :功能          :从抓取预测处理所得到的位置，角度，宽度映射图中提取no_grasp个最有效的抓取
    :参数 q_out    :int,抓取质量（位置）映射图
    :参数 ang_out  :int,抓取角度映射图
    :参数 wid_out  :int,抓取宽度映射图
    :参数 no_grasp :int,想要提取的有效抓取个数
    :返回          :list,包含多个grasp_cpaw对象的列表
    '''
    grasps_pre = []
    local_max = peak_local_max(q_out, min_distance=20, threshold_abs=0.2,num_peaks = no_grasp)
    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)
        grasp_angle = ang_out[grasp_point]

        g = Grasp_cpaw(grasp_point,grasp_angle)
        if wid_out is not None:
            g.width = wid_out[grasp_point]
            g.length = g.width/2

        grasps_pre.append(g)

    return grasps_pre

def max_iou(grasp_pre,grasps_true):
    '''
    :功能 :对于一个给定的预测抓取框，首先将其转化为Grasp对象，然后遍历计算其与各个真是标注的iou，返回最大的iou
    :参数 : grasp_pre  :Grasp对象，单个预测结果中反求出的抓取框
    :参数 : grasps_true:Grasps对象，该对象所有的真实标注抓取框
    :返回 : 最大的iou
    '''
    grasp_pre = grasp_pre.as_gr
    max_iou = 0
    for grasp_true in grasps_true.grs:
        Iou = iou(grasp_pre,grasp_true)
        max_iou = max(max_iou,Iou)
    return max_iou

def iou(grasp_pre,grasp_true,angle_threshold = np.pi/6):
    '''
    :功能 :计算两个给定框的iou
    :参数 : grasp_pre      :Grasp对象，单个预测结果中反求出的抓取框
    :参数 : grasp_true     :Grasp对象，单个真实标注抓取框
    :参数 : angle_threshold:角度阈值，超过这个角度就认为两者不符
    :返回 : 两者的iou
    '''
    #超过这个角度阈值就认为这两者不符，下面的计算复杂是为了消除角度方向的影响
    if abs((grasp_pre.angle - grasp_true.angle + np.pi/2) % np.pi - np.pi/2) > angle_threshold:
        return 0
    #先提取出两个框的所覆盖区域
    rr1, cc1 = grasp_pre.polygon_coords()
    rr2, cc2 = polygon(grasp_true.points[:, 0], grasp_true.points[:, 1])
    try:#有时候这边返回的rr2是空的，再运行下面的就会报错，在这加个故障处理确保正常运行
        r_max = max(rr1.max(), rr2.max()) + 1
        c_max = max(cc1.max(), cc2.max()) + 1
    except:
        return 0

    #根据最大的边界来确定蒙版画布大小
    canvas = np.zeros((r_max,c_max))
    canvas[rr1,cc1] += 1
    canvas[rr2,cc2] += 1

    union = np.sum(canvas > 0)
    
    if union == 0:
        return 0

    intersection = np.sum(canvas == 2)
    #print(intersection/union)
    return intersection/union
    
# 下面的是修正参数所需要的函数
import copy

def show_grasp(img,grasp,color = (0,255,0)):
    # 给定一张图及一个抓取(Grasps和Grasp类型均可),在图上绘制出抓取框
    if not isinstance(grasp,Grasps):
        grasp = Grasps([grasp])
    for gr in grasp.grs:
        # 有时候预测出负值会报错
        try:
            for i in range(3):
                cv2.line(img,tuple(gr.points.astype(np.uint32)[i][::-1]),tuple(gr.points.astype(np.uint32)[i+1][::-1]),color,1)
            cv2.line(img,tuple(gr.points.astype(np.uint32)[3][::-1]),tuple(gr.points.astype(np.uint32)[0][::-1]),color,1)
        except:
            pass
    return img
def scale_width(gr,scale):
    grasp_cpaw = Grasp_cpaw(angle=gr.angle,center=gr.center,length=gr.length,width=gr.width)
    grasp_cpaw.width = grasp_cpaw.width*scale
    gr = grasp_cpaw.as_gr
    
    return gr

def rotate_angle(gr,angle):
    # 记录原始中心
    old_center = gr.center
    # 执行旋转
    gr.rotate(angle,gr.center)
    # 将抓取平移到原始的中心
    grasp_cpaw = Grasp_cpaw(angle=gr.angle,center=old_center,length=gr.length,width=gr.width)
    gr = grasp_cpaw.as_gr
    
    return gr

# [y,x] x为正为向右平移,y为正为向下平移
def move_position(gr,offest):
    # 先转正
    angle = gr.angle
    old_center = gr.center
    gr.rotate(-angle,gr.center)
    grasp_cpaw = Grasp_cpaw(angle=gr.angle,center=old_center,length=gr.length,width=gr.width)
    gr = grasp_cpaw.as_gr
    # 执行平移
    new_center = gr.center+offest
    # 再转回来
    old_center = gr.center
    gr.rotate(angle,gr.center)
    grasp_cpaw = Grasp_cpaw(angle=gr.angle,center=new_center,length=gr.length,width=gr.width)
    gr = grasp_cpaw.as_gr
    
    return gr
def correct_grasp(edges,gr,idx,edge_width = 8):
    # gr :: type :Grasp
    
    # 根据给定抓取从所给边缘图像中裁剪得到五个区域
    width = gr.width
    gr_bak = copy.copy(gr)
    img = Image(edges)
    a = img.img
    # 可视化的选项
    img_b = show_grasp(copy.copy(edges),gr,color = 200)
    angle = gr.angle
    center = gr.center
    img.rotate(-angle,center)
    # 将抓取也转正
    gr.rotate(-angle,center)
    points = gr.points
    # 裁剪获得抓取框取域图像
    img.crop(points[0],points[2])
    try:
        img.resize((50,100))
    except:
        return gr,0,img_b,0
    # 裁剪获得前10% 后10%和中间的图像
    img_1 = img.img[:,0:edge_width][2:23,:]
    img_2 = img.img[:,0:edge_width][27:48,:]
    img_3 = img.img[:,100 - edge_width:100][2:23,:]
    img_4 = img.img[:,100 - edge_width:100][27:48,:]
    # img_middle = img.img[:,10:90]
    img_middle_l = img.img[:,10:50]
    img_middle_r = img.img[:,50:90]

    flag_1 = np.sum(img_1) > 10
    flag_2 = np.sum(img_2) > 10
    flag_3 = np.sum(img_3) > 10
    flag_4 = np.sum(img_4) > 10
    
    flag_m_l = np.sum(img_middle_l) > 200
    flag_m_r = np.sum(img_middle_r) > 200
    # 修正量定义
    scale = 1.05
    m_angle = 0.05
    # 重新获得原始抓取
    gr = gr_bak
    # 先判断中间有没有
    if flag_m_l and flag_m_r:
        # print('中间有')
        # 再分情况探讨四周
        # 如果四个边都有
        if flag_1 and flag_2 and flag_3 and flag_4:
            # print('四边都有,应当放大一下')
            gr = scale_width(gr,scale)

        # 三个的时候只有旋转
        # 如果1,2,3或1,3,4或
        elif (flag_1 and flag_2 and flag_3) or (flag_1 and flag_3 and flag_4):
            # print('应当顺时针转一下')
            gr = rotate_angle(gr,-m_angle)

        elif (flag_2 and flag_3 and flag_4) or (flag_1 and flag_2 and flag_4):
            # print('应当逆时针转一下')
            gr = rotate_angle(gr,m_angle)

        # 两个的时候有可能平移或者旋转
        # 如果是2,3或者就顺时针转
        elif (flag_2 and flag_3):
            # print('应当顺时针转一下')
            gr = rotate_angle(gr,-m_angle)
        # 如果是1,4或者就逆时针转
        elif (flag_1 and flag_4):
            # print('应当逆时针转一下')
            gr = rotate_angle(gr,m_angle)
        # 如果是3,4就往右移
        elif (flag_3 and flag_4):
            # print('应当往右移动')
            gr = move_position(gr,[0,1])
        # 如果是1,2就往左移
        elif (flag_1 and flag_2):
            # print('应当往左移动')
            gr = move_position(gr,[0,-1])
        # 如果是2,4就往下移
        elif (flag_2 and flag_4):
            # print('应当往下移动')
            gr = move_position(gr,[-1,0])
        # 如果是1,3就往上移
        elif (flag_1 and flag_3):
            # print('应当往上移动')
            gr = move_position(gr,[1,0])

        # 一个的话就都是旋转
        # 如果是2或3就顺时针旋转
        elif flag_2 or flag_3:
            # print('应当顺时针转一下')
            gr = rotate_angle(gr,-m_angle)
        # 如果是1或4就逆时针旋转
        elif flag_1 or flag_4:
            # print('应当逆时针转一下')
            gr = rotate_angle(gr,m_angle)
        else:
            # print('这是个无碰撞的抓取')
            img_a = show_grasp(copy.copy(edges),gr,color = 200)
            cv2.imwrite('c_images/{}_img2_a.png'.format(idx.cpu().data.numpy()),img_a)
            return gr,1,img_b,img_a
    else:
        # print('中间没有,应当放大一下')
        gr = scale_width(gr,scale)
        if flag_1 and flag_2 and flag_m_l:
            # print('应该向左移一下')
            gr = move_position(gr,[0,-1])
        if flag_3 and flag_4 and flag_m_r:
            # print('应该向右移一下')
            gr = move_position(gr,[0,1])
        img_a = show_grasp(copy.copy(edges),gr)
        return gr,0,img_b,img_a
    img_a = show_grasp(copy.copy(edges),gr,color = 200)

    return gr,0,img_b,img_a

def delete_surplus(edges,gr,gr_origin,idx):
    # gr :: type :Grasp
    # 根据给定抓取从所给边缘图像中裁剪得到五个区域
    width_o = gr_origin.width
    width_0 = gr.width
    gr_bak = copy.copy(gr)
    angle = gr.angle
    center = gr.center

    img = Image(copy.copy(edges))
    img.rotate(-angle,center)

    # 将抓取也转正
    gr.rotate(-angle,center)
    points = gr.points
    # 裁剪获得抓取框取域图像
    img.crop(points[0],points[2])
    img.resize((50,100))
    gr = gr_bak
    # 裁剪获得前8% 后8%和中间的图像
    # flag_1_l = np.sum(img.img[:,0:8]) > 10
    flag_2_l = np.sum(img.img[:,0:16]) > 10
    flag_3_l = np.sum(img.img[:,0:24]) > 10
    flag_4_l = np.sum(img.img[:,0:32]) > 10
    flag_5_l = np.sum(img.img[:,0:40]) > 10

    flag_1_r = np.sum(img.img[:,60:100]) > 10
    flag_2_r = np.sum(img.img[:,68:100]) > 10
    flag_3_r = np.sum(img.img[:,76:100]) > 10
    flag_4_r = np.sum(img.img[:,84:100]) > 10
    # flag_5_r = np.sum(img.img[:,92:100]) > 10

    l_list = np.array([flag_2_l,flag_3_l,flag_4_l,flag_5_l],dtype = np.int32)
    r_list = np.array([flag_1_r,flag_2_r,flag_3_r,flag_4_r],dtype = np.int32)
    zero_l = 4-np.count_nonzero(l_list)
    zero_r = 4-np.count_nonzero(r_list)

    # 需要削去的百分比例
    scale = 1 - 0.08*(zero_l + zero_r)
    # 每个0削去部分的实际宽度除以2就是需要平移的距离
    step = (gr.width*0.08/2)
    # print('移动步幅为',step)

    # 削去多余部分
    # print(gr.width)
    # print(gr.center)
    gr = scale_width(gr,scale)
    # print(gr.width)
    # print(gr.center)
    # 执行平移哪边削得少就往哪边平移,坐标0多就是正,往右移,有边0多就是负,往左移
    move = int(round((zero_l-zero_r)*step))
    # print(move)
    gr = move_position(gr,[0,move])
    # print(gr.width)
    # print(gr.center)
    # print(l_list,r_list)
    # print(scale)

    # print('宽度缩放',gr.width/gr_origin.width)
    # print('中心偏移',gr_origin.center-gr.center)
    # print('角度差异',gr_origin.angle-gr.angle)
    img_a = show_grasp(copy.copy(edges),gr,color = 200)
    cv2.imwrite('c_images/{}_img3_d.png'.format(idx.cpu().data.numpy()),img_a)
    width = gr.width
    width_origin = gr_origin.width
    angle = gr.angle
    origin_angle = gr_origin.angle
    center = gr.center
    origin_center = gr_origin.center
    img_origin = show_grasp(copy.copy(edges),gr_origin,color = 200)
    cv2.imwrite('c_images/{}_img1_origin.png'.format(idx.cpu().data.numpy()),img_origin)
    return gr,gr.width/gr_origin.width,gr.center-gr_origin.center,gr.angle-gr_origin.angle

def detect_dep(depth_img,gr0,edge_width = 8):
    # 根据给定抓取从所给边缘图像中裁剪得到五个区域
    gr = copy.copy(gr0)# 这得copy一份,否则会影响外面
    img = Image(depth_img)
    angle = gr.angle
    center = gr.center
    img = Image(depth_img)
    img.rotate(-angle,center)
    gr.rotate(-angle,center)
    show_grasp(img.img,gr)
    points = gr.points
    img.crop(points[0],points[2])
    img.resize((50,100))
    img_1 = img.img[:,0:edge_width][2:23,:]
    img_2 = img.img[:,0:edge_width][27:48,:]
    img_3 = img.img[:,100 - edge_width:100][2:23,:]
    img_4 = img.img[:,100 - edge_width:100][27:48,:]
    img_middle = img.img[:,10:90]

    flag_m,_,_,_,height_m = check_layers(img_middle,4200,middle = True)

    flag_1,_,_,_,height_1 = check_layers(img_1,200,base_height = height_m)

    flag_2,_,_,_,height_2 = check_layers(img_2,200,base_height = height_m)

    flag_3,_,_,_,height_3 = check_layers(img_3,200,base_height = height_m)

    flag_4,_,_,_,height_4 = check_layers(img_4,200,base_height = height_m)

    if flag_1 and flag_2 and flag_3 and flag_4 and not flag_m:
        # print('合理抓取')
        return 0
    else :
        return 1

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
    n,b = np.histogram(array,bins = bins,weights=weights)
    # 这个上面的bins是个可以调的参数,其实不用特别多,前面一开始是10,现在改成6了
    # plt.show()
    ranges = []
    probs = []
    mean_heights = []
    for index in np.where(n > 0.15)[0]:
        ran = b[index:index+2]
        result = [i for i in array if i>ran[0] and i <=ran[1]]
        if result == []:
            continue
        mean_height = np.mean(result)
        if base_height:
            low_layer = base_height * 0.3
            if mean_height < low_layer:# 这个是用来剔除落点中的一些小边缘干扰平面
                continue
        prob = len(result)/total_img
        if prob <0.05:# 这个是用来去除一些小的噪点
            # print(prob)
            continue
        ranges.append(ran)
        probs.append(prob)# 计算实际占比
        mean_heights.append(mean_height)
    # print(ranges)
    # print(probs)
    # print(mean_heights)
    
    # 这个是用来筛选里面高度有没有没有超过15的
    if len(ranges)  == 0:
        flag = 2
        return flag,ranges,probs,mean_heights,0
    # 这个是用来跟中间的高度相比比较的
    if base_height:
        if max(mean_heights) - base_height > 10: # 
            flag = 3
    # 如果是中间区域的话,返回占比最高的那个,而不是最高的那个
    height = max(mean_heights)
    if middle:
        height = mean_heights[probs.index(max(probs))]
    return flag, ranges, probs, mean_heights, height