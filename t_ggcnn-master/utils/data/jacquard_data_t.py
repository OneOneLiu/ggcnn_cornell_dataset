import os
import glob
import matplotlib.pyplot as plt
from .grasp_data import GraspDatasetBase
from utils.dataset_processing import grasp, image
from collections import Counter
import numpy as np

class JacquardDataset(GraspDatasetBase):
    """
    Dataset wrapper for the Jacquard dataset.
    """
    def __init__(self, file_path, start=0.0, end=1.0, ds_rotate=0, ADJ = False, npy_path = None, **kwargs):
        """
        :param file_path: Jacquard Dataset directory.
        :param start: If splitting the dataset, start at this fraction [0,1]
        :param end: If splitting the dataset, finish at this fraction
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(JacquardDataset, self).__init__(**kwargs)
        if ADJ:
            graspf = np.load(os.path.join(file_path,npy_path)).tolist()
            start=0.0
            end=1.0
        else:
            graspf = glob.glob(os.path.join(file_path, '*', '*', '*_grasps.txt'))
            graspf.sort()
        l = len(graspf)
        if l == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))
        if ds_rotate:
            graspf = graspf[int(l*ds_rotate):] + graspf[:int(l*ds_rotate)]

        depthf = [f.replace('grasps.txt', 'perfect_depth.tiff') for f in graspf]
        rgbf = [f.replace('perfect_depth.tiff', 'RGB.png') for f in depthf]
        maskf = [filename.replace('grasps.txt','mask.png') for filename in graspf]
        mask_df1 = [filename.replace('grasps.txt','mask_d_1.png') for filename in graspf]
        mask_prob = [filename.replace('grasps.txt','prob_20.png') for filename in graspf]
        
        self.grasp_files = graspf[int(l*start):int(l*end)]
        self.depth_files = depthf[int(l*start):int(l*end)]
        self.rgb_files = rgbf[int(l*start):int(l*end)]
        self.maskf = maskf[int(l*start):int(l*end)]
        self.mask_df1 = mask_df1[int(l*start):int(l*end)]
        self.mask_prob = mask_prob[int(l*start):int(l*end)]
        
    def get_typical(self, idx, rot=0, zoom=1.0):
        grs = grasp.GraspRectangles.load_from_jacquard_file(self.grasp_files[idx], scale=self.output_size / 1024.0)
        c = self.output_size // 2
        grs.rotate(rot,(c,c))
        grs.zoom(zoom,(c,c))
        angles = [round(gr.angle,1) for gr in grs.grs]
        angle_dict = dict(Counter(angles))
        list1,list2 = (list(t) for t in zip(*sorted(zip(angle_dict.values(),angle_dict.keys()))))
        first = list1[-1]
        if len(list1)>1:
            second = list1[-2]
            total = first+second
        else:
            total = first
        percent = (total)/len(grs.grs)
        if percent > self.prob:
            # 生成样例学习
            angle_shift = 0.2
            typical1_min= list2[-1]-angle_shift
            typical1_max = list2[-1]+angle_shift
            grs1 = []
            grs2 = []
            for gr in grs.grs[::]:
                if gr.angle<typical1_max and gr.angle>typical1_min:
                    grs1.append(gr)
                else:
                    grs2.append(gr)
            grasp_true1 = grasp.GraspRectangles(grs1)
            grasp_true2 = grasp.GraspRectangles(grs2)
            return percent,grasp_true1,grasp_true2
        return percent,grasp.GraspRectangles(grs),grasp.GraspRectangles(grs)

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_jacquard_file(self.grasp_files[idx], scale=self.output_size / 1024.0)
        c = self.output_size//2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))
        return gtbbs
 
    def get_imgs(self, idx, rot=0, zoom=1.0):
        prob,grasps1,grasps2 = self.get_typical(idx = idx,rot = rot,zoom = zoom)
        mask_img = np.clip(self.get_mask(idx = idx,rot = rot,zoom = zoom)/200,0,1)
        pos_range = grasps1.generate_pos_img()
        not_covered = np.clip((mask_img)-pos_range,0,1)

        pos_img,ang_img,width_img = grasps1.generate_img_n(self.output_size,self.output_size)
        if len(grasps2.grs) > 1:
            pos_img_rest,ang_img_rest,width_img_rest = grasps2.generate_img_n(self.output_size,self.output_size)

            # NOTE 调使用,后删
            # pos_img_o, ang_img_o, width_img_o = self.get_gtbb(idx = idx,rot = rot,zoom = zoom).draw((self.output_size,
            # self.output_size))
            # rgb_img = self.get_rgb(idx, rot, zoom, normalise = False)
            # plt.subplot(4,4,1)
            # plt.title("rgb_img")
            # plt.imshow(rgb_img)
            # plt.subplot(4,4,2)
            # plt.title("ang_origin")
            # plt.imshow(ang_img_o)
            # plt.subplot(4,4,3)
            # plt.title("width_origin")
            # plt.imshow(width_img_o)
            # plt.subplot(4,4,4)
            # plt.title("mask")
            # plt.imshow(mask_img)
            # plt.subplot(4,4,5)
            # plt.title("pos")
            # plt.imshow(pos_img)
            # plt.subplot(4,4,6)
            # plt.title("ang")
            # plt.imshow(ang_img)
            # plt.subplot(4,4,7)
            # plt.title("width")
            # plt.imshow(width_img)
            # plt.subplot(4,4,9)
            # plt.title("pose_rest")
            # plt.imshow(pos_img_rest)
            # plt.subplot(4,4,10)
            # plt.title("ang_rest")
            # plt.imshow(ang_img_rest)
            # plt.subplot(4,4,11)
            # plt.title("width_rest")
            # plt.imshow(width_img_rest)
            # plt.subplot(4,4,12)
            # plt.title("pos_range")
            # plt.imshow(pos_range)
            # plt.subplot(4,4,13)
            # plt.title("pos_img+rest")
            # plt.imshow(pos_img_rest * not_covered + pos_img)
            # plt.subplot(4,4,14)
            # plt.title("ang_img+rest")
            # plt.imshow(ang_img_rest * not_covered + ang_img)
            # plt.subplot(4,4,15)
            # plt.title("wid_img+rest")
            # plt.imshow(width_img_rest * not_covered + width_img)
            # plt.subplot(4,4,16)
            # plt.title("not_cover")
            # plt.imshow(not_covered)
            # plt.show()
            # NOTE 调使用,后删
            return pos_img_rest * not_covered + pos_img, ang_img_rest * not_covered + ang_img, width_img_rest * not_covered + width_img
        return pos_img, ang_img, width_img

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        depth_img.rotate(rot)
        depth_img.normalise()
        # plt.subplot(121)
        # plt.imshow(depth_img.img)
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        # NOTE 调使用,后删
        # plt.subplot(122)
        # plt.imshow(depth_img.img)
        # plt.show()
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img
    def get_mask(self,idx,rot = 0,zoom = 1.0):
        '''
        :功能     :读取返回指定id的depth图像
        :参数 idx :int,要读取的数据id
        :返回     :ndarray,处理好后的depth图像
        '''
        mask_img = image.DepthImage.from_tiff(self.maskf[idx])
        mask_img.rotate(rot)
        mask_img.zoom(zoom)
        mask_img.resize((self.output_size,self.output_size))

        return mask_img.img
    def get_scale(self,idx):
        '''
        :功能     :读取返回指定id的depth图像
        :参数 idx :int,要读取的数据id
        :返回     :ndarray,处理好后的depth图像
        '''
        mask_img = image.DepthImage.from_tiff(self.maskf[idx])
        mask_img.resize((self.output_size,self.output_size))

        # 通过mask来判断这个目标的大小,然后指示后面的缩放尺度
        plt.imshow(mask_img.img)
        plt.show()
        # 它这个zoom是只能放大不能缩小的,ZOOM越小,放大越多
        # 获取面积百分比
        size_percentage = sum(sum(mask_img.img)) / 90000
        # 获取横纵尺度
        x,y = np.where(mask_img.img != 0)
        x_range = x.max()-x.min()
        y_range = y.max()-y.min()

        scale = np.array([size_percentage/0.2,x_range/200,y_range/200,0.2]).max()
        self.scale = min(0.5, scale)
        return 0

    def get_mask_d_1(self,idx,rot = 0,zoom = 1.0):
        '''
        :功能     :读取返回指定id的depth图像
        :参数 idx :int,要读取的数据id
        :返回     :ndarray,处理好后的depth图像
        '''
        mask_d_img = image.DepthImage.from_tiff(self.mask_df1[idx])
        mask_d_img.rotate(rot)
        mask_d_img.zoom(zoom)
        mask_d_img.resize((self.output_size,self.output_size))

        return mask_d_img.img
    
    def get_mask_prob(self,idx,rot = 0,zoom = 1.0):
        '''
        :功能     :读取返回指定id的depth图像
        :参数 idx :int,要读取的数据id
        :返回     :ndarray,处理好后的depth图像
        '''
        mask_prob_img = image.DepthImage.from_tiff(self.mask_prob[idx])
        # mask_prob_img 除以其最大值,来将值缩放到[0,1]范围内
        img_max = mask_prob_img.img.max()
        mask_prob_img.rotate(rot)
        mask_prob_img.zoom(zoom)
        mask_prob_img.resize((self.output_size,self.output_size))

        return mask_prob_img.img / img_max

    def get_jname(self, idx):
        return '_'.join(self.grasp_files[idx].split(os.sep)[-1].split('_')[:-1])
