import torch
import torch.nn as nn


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_conv2d(c1, c2, k, p, s, d, g, bias=False):
    conv = nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias)

    return conv


def get_activation(act_type=None):
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_type == 'mish':
        return nn.Mish(inplace=True)
    elif act_type == 'silu':
        return nn.SiLU(inplace=True)


def get_norm(norm_type, dim):
    if norm_type == 'BN':
        return nn.BatchNorm2d(dim)
    elif norm_type == 'GN':
        return nn.GroupNorm(num_groups=32, num_channels=dim)


# Basic conv layer
class Conv(nn.Module):
    def __init__(self, 
                 c1,                   # in channels
                 c2,                   # out channels 
                 k=1,                  # kernel size 
                 p=0,                  # padding
                 s=1,                  # padding
                 d=1,                  # dilation
                 act_type='lrelu',     # activation
                 norm_type='BN',       # normalization
                 depthwise=False):
        super(Conv, self).__init__()
        convs = []
        add_bias = False if norm_type else True
        if depthwise:
            convs.append(get_conv2d(c1, c1, k=k, p=p, s=s, d=d, g=c1, bias=add_bias))
            # depthwise conv
            if norm_type:
                convs.append(get_norm(norm_type, c1))
            if act_type:
                convs.append(get_activation(act_type))
            # pointwise conv
            convs.append(get_conv2d(c1, c2, k=1, p=0, s=1, d=d, g=1, bias=add_bias))
            if norm_type:
                convs.append(get_norm(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))

        else:
            convs.append(get_conv2d(c1, c2, k=k, p=p, s=s, d=d, g=1, bias=add_bias))
            if norm_type:
                convs.append(get_norm(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))
            
        self.convs = nn.Sequential(*convs)


    def forward(self, x):
        return self.convs(x)


# ELAN Block
class ELANBlock(nn.Module):
    """
    ELAN BLock of YOLOv7's backbone
    """
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, act_type='silu', norm_type='BN', depthwise=False):
        super(ELANBlock, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv3 = nn.Sequential(*[
            Conv(inter_dim, inter_dim, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
            for _ in range(2)
        ])
        self.cv4 = nn.Sequential(*[
            Conv(inter_dim, inter_dim, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
            for _ in range(2)
        ])

        self.out = Conv(inter_dim*4, out_dim, k=1, act_type=act_type, norm_type=norm_type)



    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, 2C, H, W]
        """
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)

        # [B, C, H, W] -> [B, 2C, H, W]
        out = self.out(torch.cat([x1, x2, x3, x4], dim=1))

        return out


# DownSample Block
class DownSample(nn.Module):
    def __init__(self, in_dim, act_type='silu', norm_type='BN'):
        super().__init__()
        inter_dim = in_dim // 2
        self.mp = nn.MaxPool2d((2, 2), 2)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(inter_dim, inter_dim, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type)
        )

    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, C, H//2, W//2]
        """
        # [B, C, H, W] -> [B, C//2, H//2, W//2]
        x1 = self.cv1(self.mp(x))
        x2 = self.cv2(x)

        # [B, C, H//2, W//2]
        out = torch.cat([x1, x2], dim=1)

        return out


# ELAN Block for PaFPN
class ELANBlockFPN(nn.Module):
    """
    ELAN BLock of YOLOv7's head
    """
    def __init__(self, in_dim, out_dim, act_type='silu', norm_type='BN', depthwise=False):
        super(ELANBlockFPN, self).__init__()
        # Basic parameters
        e1, e2 = 0.5, 0.5
        width = 4
        depth = 1
        inter_dim = int(in_dim * e1)
        inter_dim2 = int(inter_dim * e2) 
        # Network structure
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv3 = nn.ModuleList()
        for idx in range(width):
            if idx == 0:
                cvs = [Conv(inter_dim, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)]
            else:
                cvs = [Conv(inter_dim2, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)]
            # deeper
            if depth > 1:
                for _ in range(1, depth):
                    cvs.append(Conv(inter_dim2, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise))
                self.cv3.append(nn.Sequential(*cvs))
            else:
                self.cv3.append(cvs[0])

        self.out = Conv(inter_dim*2+inter_dim2*len(self.cv3), out_dim, k=1, act_type=act_type, norm_type=norm_type)


    def forward(self, x):
        """
        Input:
            x: [B, C_in, H, W]
        Output:
            out: [B, C_out, H, W]
        """
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        inter_outs = [x1, x2]
        for m in self.cv3:
            y1 = inter_outs[-1]
            y2 = m(y1)
            inter_outs.append(y2)

        # [B, C_in, H, W] -> [B, C_out, H, W]
        out = self.out(torch.cat(inter_outs, dim=1))

        return out


# DownSample Block for PaFPN
class DownSampleFPN(nn.Module):
    def __init__(self, in_dim, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        inter_dim = in_dim
        self.mp = nn.MaxPool2d((2, 2), 2)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(inter_dim, inter_dim, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )

    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, 2C, H//2, W//2]
        """
        # [B, C, H, W] -> [B, C//2, H//2, W//2]
        x1 = self.cv1(self.mp(x))
        x2 = self.cv2(x)

        # [B, C, H//2, W//2]
        out = torch.cat([x1, x2], dim=1)

        return out
