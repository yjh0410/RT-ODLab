import numpy as np
import torch
import torch.nn as nn


# ---------------------------- Base Conv Module ----------------------------
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
    elif act_type is None:
        return nn.Identity()

def get_norm(norm_type, dim):
    if norm_type == 'BN':
        return nn.BatchNorm2d(dim)
    elif norm_type == 'GN':
        return nn.GroupNorm(num_groups=32, num_channels=dim)

## Basic Conv Module
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
        p = p if d == 1 else d
        if depthwise:
            convs.append(get_conv2d(c1, c1, k=k, p=p, s=s, d=d, g=c1, bias=add_bias))
            # depthwise conv
            if norm_type is not None:
                convs.append(get_norm(norm_type, c1))
            if act_type is not None:
                convs.append(get_activation(act_type))
            # pointwise conv
            convs.append(get_conv2d(c1, c2, k=1, p=0, s=1, d=d, g=1, bias=add_bias))
            if norm_type is not None:
                convs.append(get_norm(norm_type, c2))
            if act_type is not None:
                convs.append(get_activation(act_type))

        else:
            convs.append(get_conv2d(c1, c2, k=k, p=p, s=s, d=d, g=1, bias=add_bias))
            if norm_type is not None:
                convs.append(get_norm(norm_type, c2))
            if act_type is not None:
                convs.append(get_activation(act_type))
            
        self.convs = nn.Sequential(*convs)


    def forward(self, x):
        return self.convs(x)

## Partial Conv Module
class PartialConv(nn.Module):
    def __init__(self, in_dim, out_dim, split_ratio=0.25, kernel_size=1, stride=1, act_type=None, norm_type=None):
        super().__init__()
        # ----------- Basic Parameters -----------
        assert in_dim == out_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.split_ratio = split_ratio
        self.split_dim = round(in_dim * split_ratio)
        self.untouched_dim = in_dim - self.split_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.stride = stride
        self.act_type = act_type
        self.norm_type = norm_type
        # ----------- Network Parameters -----------
        self.partial_conv = Conv(self.split_dim, self.split_dim, self.kernel_size, self.padding, self.stride, act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.split_dim, self.untouched_dim], dim=1)
        x1 = self.partial_conv(x1)
        x = torch.cat((x1, x2), 1)

        return x

## Channel Shuffle
class ChannelShuffle(nn.Module):
    def __init__(self, groups=1) -> None:
        super().__init__()
        self.groups = groups

    def forward(self, x):
        # type: (torch.Tensor, int) -> torch.Tensor
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups

        # reshape
        x = x.view(batchsize, self.groups,
                channels_per_group, height, width)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x

## Inverse BottleNeck
class InverseBottleneck(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio=2.0,
                 shortcut=False,
                 act_type='silu',
                 norm_type='BN',
                 depthwise=False):
        super(InverseBottleneck, self).__init__()
        # ----------- Basic Parameters -----------
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.expand_dim = int(in_dim * expand_ratio)           
        # ----------- Network Parameters -----------
        self.cv1 = Conv(in_dim, in_dim, k=3, p=1, act_type=None, norm_type=norm_type, depthwise=depthwise)
        self.cv2 = Conv(in_dim, self.expand_dim, k=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.cv3 = Conv(self.expand_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.shortcut = shortcut and in_dim == out_dim

    def forward(self, x):
        h = self.cv3(self.cv2(self.cv1(x)))

        return x + h if self.shortcut else h

## YOLO-style BottleNeck
class YoloBottleneck(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio=0.5,
                 shortcut=False,
                 act_type='silu',
                 norm_type='BN',
                 depthwise=False):
        super(YoloBottleneck, self).__init__()
        # ------------------ Basic parameters ------------------
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inter_dim = int(out_dim * expand_ratio)
        self.shortcut = shortcut and in_dim == out_dim
        # ------------------ Network parameters ------------------
        self.cv1 = Conv(in_dim, self.inter_dim, k=1, norm_type=norm_type, act_type=act_type)
        self.cv2 = Conv(self.inter_dim, out_dim, k=3, p=1, norm_type=norm_type, act_type=act_type, depthwise=depthwise)

    def forward(self, x):
        h = self.cv2(self.cv1(x))

        return x + h if self.shortcut else h


# ---------------------------- Base Modules ----------------------------
## ELAN Stage of Backbone
class ELAN_Stage(nn.Module):
    def __init__(self, in_dim, out_dim, squeeze_ratio :float=0.5, branch_depth :int=1, shortcut=False, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        # ----------- Basic Parameters -----------
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inter_dim = round(in_dim * squeeze_ratio)
        self.squeeze_ratio = squeeze_ratio
        self.branch_depth = branch_depth
        # ----------- Network Parameters -----------
        self.cv1 = Conv(in_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(in_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv3 = nn.Sequential(*[
            YoloBottleneck(self.inter_dim, self.inter_dim, 1.0, shortcut, act_type, norm_type, depthwise)
            for _ in range(branch_depth)
        ])
        self.cv4 = nn.Sequential(*[
            YoloBottleneck(self.inter_dim, self.inter_dim, 1.0, shortcut, act_type, norm_type, depthwise)
            for _ in range(branch_depth)
        ])
        ## output
        self.out_conv = Conv(self.inter_dim*4, out_dim, k=1, act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)
        out = self.out_conv(torch.cat([x1, x2, x3, x4], dim=1))

        return out
    
## DownSample Block
class DSBlock(nn.Module):
    def __init__(self, in_dim, out_dim, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inter_dim = out_dim // 2
        # branch-1
        self.maxpool = nn.Sequential(
            Conv(in_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            nn.MaxPool2d((2, 2), 2)
        )
        # branch-2
        self.ds_conv = nn.Sequential(
            Conv(in_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(self.inter_dim, self.inter_dim, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        ) 


    def forward(self, x):
        # branch-1
        x1 = self.maxpool(x)
        # branch-2
        x2 = self.ds_conv(x)
        # out-proj
        out = torch.cat([x1, x2], dim=1)

        return out


# ---------------------------- FPN Modules ----------------------------
## build fpn's core block
def build_fpn_block(cfg, in_dim, out_dim):
    if cfg['fpn_core_block'] == 'elan_block':
        layer = ELAN_Stage(in_dim        = in_dim,
                           out_dim       = out_dim,
                           squeeze_ratio = cfg['fpn_squeeze_ratio'],
                           branch_depth  = round(3 * cfg['depth']),
                           shortcut      = False,
                           act_type      = cfg['fpn_act'],
                           norm_type     = cfg['fpn_norm'],
                           depthwise     = cfg['fpn_depthwise']
                           )
        
    return layer

## build fpn's reduce layer
def build_reduce_layer(cfg, in_dim, out_dim):
    if cfg['fpn_reduce_layer'] == 'conv':
        layer = Conv(in_dim, out_dim, k=1, act_type=cfg['fpn_act'], norm_type=cfg['fpn_norm'])
        
    return layer

## build fpn's downsample layer
def build_downsample_layer(cfg, in_dim, out_dim):
    if cfg['fpn_downsample_layer'] == 'conv':
        layer = Conv(in_dim, out_dim, k=3, s=2, p=1,
                     act_type=cfg['fpn_act'], norm_type=cfg['fpn_norm'], depthwise=cfg['fpn_depthwise'])
    elif cfg['fpn_downsample_layer'] == 'maxpool':
        assert in_dim == out_dim
        layer = nn.MaxPool2d((2, 2), stride=2)
        
    return layer
