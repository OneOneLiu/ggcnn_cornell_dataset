import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, x):
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

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width, mask_prob, mask_height = yc
        pos_pred, cos_pred, sin_pred, width_pred, filter, filter_cos, filter_sin, filter_width = self(xc)

        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)

        prob = filter
        prob_loss = F.mse_loss(filter,mask_prob)

        filtered_cos = filter_cos.sigmoid() * cos_pred
        filtered_sin = filter_sin.sigmoid() * sin_pred
        filtered_width = filter_width.sigmoid() * width_pred

        cos_loss_f = F.mse_loss(filtered_cos, y_cos)
        sin_loss_f = F.mse_loss(filtered_sin, y_sin)
        width_loss_f = F.mse_loss(filtered_width, y_width)
        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss + prob_loss + cos_loss_f + sin_loss_f + width_loss_f,
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
                'prob': prob
            }
        }