import paddle
import paddle.nn as nn
import paddle.nn.functional as F

#网络参数定义
filter_sizes = [32, 16, 8, 8, 16, 32]
kernel_sizes = [9, 5, 3, 3, 5, 9]
strides = [3, 2, 2, 2, 2, 3]

class GGCNN(nn.Layer):
    def __init__(self,input_channels = 1):
        super(GGCNN,self).__init__()

        self.conv1 = nn.Conv2D(input_channels,filter_sizes[0],kernel_sizes[0],stride=strides[0],padding=3,
                                weight_attr = paddle.ParamAttr(initializer=nn.initializer.XavierUniform()))
        self.conv2 = nn.Conv2D(filter_sizes[0], filter_sizes[1],kernel_sizes[1], stride=strides[1], padding=2,
                                weight_attr = paddle.ParamAttr(initializer=nn.initializer.XavierUniform()))
        self.conv3 = nn.Conv2D(filter_sizes[1], filter_sizes[2],kernel_sizes[2], stride=strides[2], padding=1,
                                weight_attr = paddle.ParamAttr(initializer=nn.initializer.XavierUniform()))
        
        self.convt1 = nn.Conv2DTranspose(filter_sizes[2], filter_sizes[3], kernel_sizes[3], stride=strides[3], padding=1,
                                output_padding = 1, weight_attr = paddle.ParamAttr(initializer=nn.initializer.XavierUniform()))
        self.convt2 = nn.Conv2DTranspose(filter_sizes[3], filter_sizes[4], kernel_sizes[4], stride=strides[4], padding=2,
                                output_padding = 1, weight_attr = paddle.ParamAttr(initializer=nn.initializer.XavierUniform()))
        self.convt3 = nn.Conv2DTranspose(filter_sizes[4], filter_sizes[5], kernel_sizes[5], stride=strides[5], padding=3,
                                output_padding = 1, weight_attr = paddle.ParamAttr(initializer=nn.initializer.XavierUniform()))
        
        self.pos_output = nn.Conv2D(filter_sizes[5], 1, kernel_size=2,weight_attr = paddle.ParamAttr(initializer=nn.initializer.XavierUniform()))
        self.cos_output = nn.Conv2D(filter_sizes[5], 1, kernel_size=2,weight_attr = paddle.ParamAttr(initializer=nn.initializer.XavierUniform()))
        self.sin_output = nn.Conv2D(filter_sizes[5], 1, kernel_size=2,weight_attr = paddle.ParamAttr(initializer=nn.initializer.XavierUniform()))
        self.width_output = nn.Conv2D(filter_sizes[5], 1, kernel_size=2,weight_attr = paddle.ParamAttr(initializer=nn.initializer.XavierUniform()))
    
    def forward(self, x):
        '''
        :功能     :前向传播函数
        :参数 x   :tensors,一次网络输入
        :返回     :tensors，各参数的预测结果
        '''
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))
        x = F.relu(self.convt3(x))

        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)
        #print('output:{}'.format(width_output.shape))

        return pos_output, cos_output, sin_output, width_output

    def compute_loss(self, xc, yc):
        '''
        :功能      :损失计算函数
        :参数 xc   :tensors,一次网络输入
        :参数 yc   :tensors,网络输入对应真实标注信息
        :返回      :dict，各损失和预测结果
        '''
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self.forward(xc)
        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
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
                'width': width_pred
            }
        }