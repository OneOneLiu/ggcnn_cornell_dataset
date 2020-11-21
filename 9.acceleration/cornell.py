import nvidia.dali.ops as ops
from base import DALIDataloader
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from sklearn.utils import shuffle

import glob
import os

IMG_DIR = '../cornell'
TRAIN_BS = 256
TEST_BS = 200
NUM_WORKERS = 4

class TrainPipe_Cornell(Pipeline):
	def __init__(self, batch_size, num_threads, device_id, data_dir, crop=32, dali_cpu=False, local_rank=0,world_size=1,cutout=0):
		'''
		batch_size:#用于pipeline实例化
		num_threads:#用于pipeline实例化
		device_id:要使用的GPU编号，#用于pipeline实例化
		data_dir:数据存放的路径
		crop:裁剪尺寸，#用于ops
		dali_cpu:
		local_rank:
		world_size:
		cutout:
		'''
		#继承父类
		super(TrainPipe_Cornell, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
		#定义类内迭代器
		self.iterator = iter(CORNELL_INPUT_ITER(batch_size, 'train', root=data_dir))
		dali_device = "gpu"
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.pad = ops.Paste(device=dali_device, ratio=1.25, fill_value=0)
        self.uniform = ops.Uniform(range=(0., 1.))
        

class CORNELL_INPUT_ITER():
	def __init__(self, batch_size, root='../cornell'，start = 0.0,end = 1.0)
		'''
		初始化函数，完成一些参数的传递
		'''
		self.root = root
        self.batch_size = batch_size

        self.data = []
        self.targets = []

        #去指定路径载入数据集数据
        graspf = glob.glob(os.path.join(file_dir,'*','pcd*cpos.txt'))
        graspf.sort()
        
        
        l = len(graspf)
        if l == 0:
            raise FileNotFoundError('没有查找到数据集，请检查路径{}'.format(file_dir))
        
        rgbf = [filename.replace('cpos.txt','r.png') for filename in graspf]
        depthf = [filename.replace('cpos.txt','d.tiff') for filename in graspf]
        
        #按照设定的边界参数对数据进行划分并指定为类的属性
        self.graspf = graspf[int(l*start):int(l*end)]
        self.rgbf = rgbf[int(l*start):int(l*end)]
        self.depthf = depthf[int(l*start):int(l*end)]

    @staticmethod
    def numpy_to_torch(s):
        '''
        :功能     :将输入的numpy数组转化为torch张量，并指定数据类型，如果数据没有channel维度，就给它加上这个维度
        :参数 s   :numpy ndarray,要转换的数组
        :返回     :tensor,转换后的torch张量
        '''
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))
    
    def _get_crop_attrs(self,idx):
        '''
        :功能     :读取多抓取框中心点的坐标，并结合output_size计算要裁剪的左上角点坐标
        :参数 idx :int,
        :返回     :计算出来的多抓取框中心点坐标和裁剪区域左上角点坐标
        '''
        grasp_rectangles = Grasps.load_from_cornell_files(self.graspf[idx])
        center = grasp_rectangles.center
        #按照ggcnn里面的话，这里本该加个限制条件，防止角点坐标溢出边界，但前面分析过，加不加区别不大，就不加了
        #分析错误，后面出现bug了，所以还是加上吧
        left = max(0, min(center[0] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[1] - self.output_size // 2, 480 - self.output_size))
        
        return center,left,top
    
    def get_rgb(self,idx):
        '''
        :功能     :读取返回指定id的rgb图像
        :参数 idx :int,要读取的数据id
        :返回     :ndarray,处理好后的rgb图像
        '''
        rgb_img = Image.from_file(self.rgbf[idx])
        
        return rgb_img.img
    
    #因为有时候只输入RGB三通道信息，所以，定义两个返回函数，一个读取RGB一个读取深度
    def get_depth(self,idx):
        '''
        :功能     :读取返回指定id的depth图像
        :参数 idx :int,要读取的数据id
        :返回     :ndarray,处理好后的depth图像
        '''
        depth_img = DepthImage.from_file(self.depthf[idx])
        return depth_img.img
    
    def get_grasp(self,idx,rot=0, zoom=1.0):
        '''
        :功能       :读取返回指定id的抓取标注参数并将多个抓取框的参数返回融合
        :参数 idx   :int,要读取的数据id
        :返回       :以图片的方式返回定义一个抓取的多个参数，包括中心点，角度，宽度和长度，同时返回idx，方便后面validate的时候查找真实的抓取框用
        '''
        grs = Grasps.load_from_cornell_files(self.graspf[idx])
        center, left, top = self._get_crop_attrs(idx)
        #先旋转再偏移再缩放
        grs.rotate(rot,center)
        grs.offset((-left,-top))
        grs.zoom(zoom,(self.output_size//2,self.output_size//2))
        pos_img,angle_img,width_img = grs.generate_img(shape = (self.output_size,self.output_size))
        
        return pos_img,angle_img,width_img
    
    def get_raw_grasps(self,idx,rot,zoom_factor):
        '''
        :功能       :读取返回指定id的抓取框信息斌进行一系列预处理(裁剪，缩放等)后以Grasps对象的形式返回
        :参数 idx   :int,要读取的数据id
        :返回       :Grasps，此id中包含的抓取
        '''
        raw_grasps = Grasps.load_from_cornell_files(self.graspf[idx])
        center, left, top = self._get_crop_attrs(idx)
        raw_grasps.rotate(rot,center)
        raw_grasps.offset((-left,-top))
        raw_grasps.zoom(zoom_factor,(self.output_size//2,self.output_size//2))
        
        return raw_grasps

    def __iter__(self):
    	'''
    	迭代函数
    	'''
        self.i = 0
        self.n = len(self.graspf)
        return self

    def __next__(self):
    	'''
    	迭代函数
    	'''
        batch = []
        labels = []
        #这边随机分批读取的只是id和路径而已，然后再将其读入成为具体的数据，然后返回具体的图像信息
        for _ in range(self.batch_size):
            if self.train and self.i % self.n == 0:
                self.depth_data, self.rgb_data,self.targets = shuffle(self.depthf, self.rgbf, self.graspf, random_state=0)
            depth_imgs, rgb_imgs, labels = self.depth_data[self.i], self.rgb_data[self.i], self.targets[self.i]
            batch.append((depth_img, rgb_img))
            labels.append(label)
            self.i = (self.i + 1) % self.n
        return (batch, labels)