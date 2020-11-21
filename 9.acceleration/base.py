from nvidia.dali.plugin.pytorch import DALIGenericIterator

class DALIDataloader(DALIGenericIterator):
	'''
	Dali数据集载入库
	'''
    def __init__(self, pipeline, size, batch_size, output_map=["image", "label","idx","rot","zoom_factor"], auto_reset=True):
        '''
        size:数据集大小
        pipeline:Dali pipeline
        batch_size:不解释
        output_map:输出值，image,(pos_img,cos_img,sin_img.width_img),idx,rot,zoom_factor
        '''
        self.size = size
        self.batch_size = batch_size
        self.output_map = output_map
        super().__init__(pipelines=pipeline, size=size, auto_reset=auto_reset, output_map=output_map)

    def __next__(self):
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch
        data = super().__next__()[0]
        return [data[self.output_map[0]], data[self.output_map[1]]]
    
    def __len__(self):
    	#计算有多少个batch
        if self.size%self.batch_size==0:
            return self.size//self.batch_size
        else:
            return self.size//self.batch_size+1