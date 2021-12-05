import os
import glob
import numpy as np

from .grasp_data import GraspDatasetBase
from utils.dataset_processing import grasp, image


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

        graspf = glob.glob(os.path.join(file_path, '*','*','*_grasps.txt'))
        graspf.sort()
        if ADJ:
            graspf = np.load(os.path.join(file_path,npy_path)).tolist()
            start=0.0
            end=1.0
        l = len(graspf)
        if l == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            graspf = graspf[int(l*ds_rotate):] + graspf[:int(l*ds_rotate)]

        depthf = [f.replace('grasps.txt', 'perfect_depth.tiff') for f in graspf]
        rgbf = [f.replace('perfect_depth.tiff', 'RGB.png') for f in depthf]
        mask_df1 = [filename.replace('grasps.txt','mask_d_1.png') for filename in graspf]
        mask_prob = [filename.replace('grasps.txt','prob_20.png') for filename in graspf]

        self.grasp_files = graspf[int(l*start):int(l*end)]
        self.depth_files = depthf[int(l*start):int(l*end)]
        self.rgb_files = rgbf[int(l*start):int(l*end)]
        self.mask_df1 = mask_df1[int(l*start):int(l*end)]
        self.mask_prob = mask_prob[int(l*start):int(l*end)]

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_jacquard_file(self.grasp_files[idx], scale=self.output_size / 1024.0)
        c = self.output_size//2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))
        return gtbbs
 
    def get_imgs(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_jacquard_file(self.grasp_files[idx], scale=self.output_size / 1024.0)
        c = self.output_size//2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))
        return gtbbs.draw((self.output_size,self.output_size))

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        depth_img.rotate(rot)
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
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
