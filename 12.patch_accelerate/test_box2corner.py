import torch

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

def img2boxes(cos_img,sin_img,width_img):
    # 区别于function里面的两个后处理,这里不会使用Gaussian过滤,也不会挑出极大值,而是会将所有的90000个预测全部返回
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0)
    width_img = (width_img * 150.0)
    length_img = width_img * 0.5

    # now let's start calculate
    # 1. generate global position for every points
    shift_x, shift_y = torch.meshgrid(0,0)
    # 2. calculate relative shift
    xo = torch.cos(ang_img)
    yo = torch.sin(ang_img)

    y1 = shift_y + width_img / 2 * yo
    x1 = shift_x - width_img / 2 * xo
    y2 = shift_y - width_img / 2 * yo
    x2 = shift_x + width_img / 2 * xo

    p1 = torch.stack((y1 - length_img/2 * xo,x1 - length_img/2 * yo),dim = 2)
    p2 = torch.stack((y2 - length_img/2 * xo,x2 - length_img/2 * yo),dim = 2)
    p3 = torch.stack((y2 + length_img/2 * xo,x2 + length_img/2 * yo),dim = 2)
    p4 = torch.stack((y1 + length_img/2 * xo,x1 + length_img/2 * yo),dim = 2)

    boxes = torch.cat((p1,p2,p3,p4),dim = 2).view(300,300,4,2)

    return boxes

if __name__ == '__main__':
    device = torch.device("cuda:0")
    box1 = torch.tensor([0,0,10,5,1]).float().to(device)
    box2 = box1.reshape(1,1,5)
    corners2 = box2corners_th(box2)
    corners1 = img2boxes(torch.cos(box1[2]),torch.sin(box1[2]),box1[3])

    
