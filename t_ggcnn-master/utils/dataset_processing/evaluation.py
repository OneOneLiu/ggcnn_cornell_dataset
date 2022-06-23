import numpy as np
import matplotlib.pyplot as plt

from .grasp import GraspRectangles, detect_grasps

import torchvision.transforms.functional as VisionF

def plot_output(rgb_img, depth_img, grasp_q_img, grasp_angle_img, no_grasps=1, grasp_width_img=None):
    """
    Plot the output of a GG-CNN
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of GG-CNN
    :param grasp_angle_img: Angle output of GG-CNN
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of GG-CNN
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
    ax.set_title('RGB')
    ax.axis('off')

    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(depth_img, cmap='gray')
    for g in gs:
        g.plot(ax)
    ax.set_title('Depth')
    ax.axis('off')

    ax = fig.add_subplot(2, 2, 3)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 2, 4)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)
    plt.show()


def calculate_iou_match(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None):
    """
    Calculate grasp success using the IoU (Jacquard) metric (e.g. in https://arxiv.org/abs/1301.3592)
    A success is counted if grasp rectangle has a 25% IoU with a ground truth, and is withing 30 degrees.
    :param grasp_q: Q outputs of GG-CNN (Nx300x300x3)
    :param grasp_angle: Angle outputs of GG-CNN
    :param ground_truth_bbs: Corresponding ground-truth BoundingBoxes
    :param no_grasps: Maximum number of grasps to consider per image.
    :param grasp_width: (optional) Width output from GG-CNN
    :return: success
    """

    if not isinstance(ground_truth_bbs, GraspRectangles):
        gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
    else:
        gt_bbs = ground_truth_bbs
    gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)
    for g in gs:
        if g.max_iou(gt_bbs) > 0.25:
            return True
    else:
        return False

def get_edge(resized_img):
    # 中心高度提取
    img_middle = resized_img[:,20:80]
    # 在这计算height_m,主要就是取较高的值
    hm = np.max(img_middle,axis = 0)
    mean_height = img_middle.mean()
    new_hm = hm[np.where(hm >= mean_height)]
    mean_hm = new_hm.mean()
    up_heights = new_hm[np.where(new_hm >= mean_hm)]
    if len(up_heights) == 0:
        height_m = mean_hm
    else:
        height_m = up_heights.mean()
    if height_m == 0: # 如果中间没有,那么就放大框,暂时也没别的办法
        return 0,0,0,0,0

    # 边缘宽度检测
    # 下面进行碰撞检测边缘提取
    h1 = np.max(resized_img,axis = 0)
    threshold = height_m - 10
    if height_m < 20:
        threshold = height_m // 2
    # 获取各个边缘的宽度
    # 1.左边缘
    edge1 = np.where(h1[0:40] > threshold)
    if len(edge1[0]) == 0:
        edge_left = 40
    else:
        edge_left = edge1[0][0]
    # 2.右边缘
    edge2 = np.where(h1[60:100] > threshold)
    if len(edge2[0]) == 0:
        edge_right = 40
    else:
        edge_right = 40-edge2[0][-1]-1

    h2 = np.max(resized_img,axis = 1)
    # 3.上边缘 调整位置用
    edge3 = np.where(h2[0:20] > threshold)
    if len(edge3[0]) == 0:
        edge_top = 20
    else:
        edge_top = edge3[0][0]
    # 4.下边缘 调整位置用
    edge4 = np.where(h2[30:50] > threshold)
    if len(edge4[0]) == 0:
        edge_bottom = 20
    else:
        edge_bottom = 20-edge4[0][-1]-1
    edge = min(edge_left,edge_right)

    return edge,edge_left,edge_right,edge_top,edge_bottom

def collision_validate(gr,mask_height):
    y = gr.center[0]
    x = gr.center[1]
    angle = gr.angle
    width = gr.length
    length = 15
    if width < 5:
        return 0, 1, 1
    top = int(y - length / 2)
    left = int(x - width / 2)
    rt_angle = -float((angle / np.pi *180))

    rectified_img = VisionF.rotate(img = mask_height.view(1,1,300,300),angle = rt_angle,center = (x,y))

    crop_img = VisionF.crop(rectified_img,top,left,int(length),int(width))

    resized_img = VisionF.resize(crop_img,[50,100]).squeeze().cpu().data.numpy()
    
    
    # 获取图像各边缘宽度
    edge,edge_left,edge_right,edge_top,edge_bottom = get_edge(resized_img)

    true_edge = edge * width * 0.01
    # plt.subplot(121)
    # plt.imshow(resized_img)
    # plt.subplot(122)
    # plt.imshow(mask_height[0][0].cpu().data.numpy())
    # plt.show()
    if edge_bottom < 15 and edge_top < 15:
        edge_bottom = 0
        edge_top = 0
    edge_left,edge_right,edge_top,edge_bottom = max(edge_left,1),max(edge_right,1),max(edge_top,1),max(edge_bottom,1)
    if edge > 0:# 如果边缘都不大于0的话就是有碰撞了,还检验个毛线
        lr_congruous = abs(edge_left-edge_right) / max([edge_left,edge_right])
        tb_congruous = abs(edge_top-edge_bottom) / max([edge_top,edge_bottom])
    else:
        lr_congruous = 1
        tb_congruous = 1
    return edge, lr_congruous, tb_congruous