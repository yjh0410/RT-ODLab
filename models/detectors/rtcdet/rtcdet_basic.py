from typing import List
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
                 kernel_sizes :List[int] = [3, 3],
                 expand_ratio :float     = 0.5,
                 shortcut     :bool      = False,
                 act_type     :str       = 'silu',
                 norm_type    :str       = 'BN',
                 depthwise    :bool      = False):
        super(YoloBottleneck, self).__init__()
        # ------------------ Basic parameters ------------------
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inter_dim = int(out_dim * expand_ratio)
        self.shortcut = shortcut and in_dim == out_dim
        self.depthwise = []
        for ksize in kernel_sizes:
            if ksize > 1:
                self.depthwise.append(depthwise)
            else:
                self.depthwise.append(False)
        # ------------------ Network parameters ------------------
        self.cv1 = Conv(in_dim, self.inter_dim, k=kernel_sizes[0], p=kernel_sizes[0]//2, norm_type=norm_type, act_type=act_type, depthwise=self.depthwise[0])
        self.cv2 = Conv(self.inter_dim, out_dim, k=kernel_sizes[1], p=kernel_sizes[1]//2, norm_type=norm_type, act_type=act_type, depthwise=self.depthwise[1])

    def forward(self, x):
        h = self.cv2(self.cv1(x))

        return x + h if self.shortcut else h


# ---------------------------- Base Modules ----------------------------
## ELAN Block for Backbone
class ELANBlock(nn.Module):
    def __init__(self, in_dim, out_dim, expand_ratio :float=0.5, branch_depth :int=1, shortcut=False, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        # ----------- Basic Parameters -----------
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inter_dim = round(in_dim * expand_ratio)
        self.expand_ratio = expand_ratio
        self.branch_depth = branch_depth
        self.shortcut = shortcut
        # ----------- Network Parameters -----------
        ## branch-1
        self.cv1 = Conv(in_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        ## branch-2
        self.cv2 = Conv(in_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        ## branch-3
        self.cv3 = nn.Sequential(*[
            YoloBottleneck(self.inter_dim, self.inter_dim, [1, 3], 1.0, shortcut, act_type, norm_type, depthwise)
            for _ in range(branch_depth)
        ])
        ## branch-4
        self.cv4 = nn.Sequential(*[
            YoloBottleneck(self.inter_dim, self.inter_dim, [1, 3], 1.0, shortcut, act_type, norm_type, depthwise)
            for _ in range(branch_depth)
        ])
        ## output proj
        self.out = Conv(self.inter_dim*4, out_dim, k=1, act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)

        # [B, C, H, W] -> [B, 2C, H, W]
        out = self.out(torch.cat([x1, x2, x3, x4], dim=1))

        return out

## ELAN Block for FPN
class ELANBlockFPN(nn.Module):
    def __init__(self, in_dim, out_dim, expand_ratio :float=0.5, branch_depth :int=1, shortcut=False, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        # ----------- Basic Parameters -----------
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inter_dim1 = round(out_dim * expand_ratio)
        self.inter_dim2 = round(self.inter_dim1 * expand_ratio)
        self.expand_ratio = expand_ratio
        self.branch_depth = branch_depth
        self.shortcut = shortcut
        # ----------- Network Parameters -----------
        ## branch-1
        self.cv1 = Conv(in_dim, self.inter_dim1, k=1, act_type=act_type, norm_type=norm_type)
        ## branch-2
        self.cv2 = Conv(in_dim, self.inter_dim1, k=1, act_type=act_type, norm_type=norm_type)
        ## branch-3
        self.cv3 = []
        for i in range(branch_depth):
            if i == 0:
                self.cv3.append(YoloBottleneck(self.inter_dim1, self.inter_dim2, [1, 3], 1.0, shortcut, act_type, norm_type, depthwise))
            else:
                self.cv3.append(YoloBottleneck(self.inter_dim2, self.inter_dim2, [1, 3], 1.0, shortcut, act_type, norm_type, depthwise))
        self.cv3 = nn.Sequential(*self.cv3)
        ## branch-4
        self.cv4 = nn.Sequential(*[
            YoloBottleneck(self.inter_dim2, self.inter_dim2, [1, 3], 1.0, shortcut, act_type, norm_type, depthwise)
            for _ in range(branch_depth)
        ])
        ## branch-5
        self.cv5 = nn.Sequential(*[
            YoloBottleneck(self.inter_dim2, self.inter_dim2, [1, 3], 1.0, shortcut, act_type, norm_type, depthwise)
            for _ in range(branch_depth)
        ])
        ## branch-6
        self.cv6 = nn.Sequential(*[
            YoloBottleneck(self.inter_dim2, self.inter_dim2, [1, 3], 1.0, shortcut, act_type, norm_type, depthwise)
            for _ in range(branch_depth)
        ])
        ## output proj
        self.out = Conv(self.inter_dim1*2 + self.inter_dim2*4, out_dim, k=1, act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)
        x5 = self.cv5(x4)
        x6 = self.cv6(x5)

        # [B, C, H, W] -> [B, 2C, H, W]
        out = self.out(torch.cat([x1, x2, x3, x4, x5, x6], dim=1))

        return out
    
## DownSample Block
class DSBlock(nn.Module):
    def __init__(self, in_dim, out_dim, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        inter_dim = out_dim // 2
        self.branch_1 = nn.Sequential(
            nn.MaxPool2d((2, 2), 2),
            Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        )
        self.branch_2 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(inter_dim, inter_dim, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        out = torch.cat([x1, x2], dim=1)

        return out


# ---------------------------- FPN Modules ----------------------------
## build fpn's core block
def build_fpn_block(cfg, in_dim, out_dim):
    if cfg['fpn_core_block'] == 'elan_block':
        layer = ELANBlockFPN(in_dim        = in_dim,
                             out_dim       = out_dim,
                             expand_ratio  = cfg['fpn_expand_ratio'],
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
    elif cfg['fpn_downsample_layer'] == 'dsblock':
        layer = DSBlock(in_dim, out_dim, cfg['fpn_act'], cfg['fpn_norm'], cfg['fpn_depthwise'])
        
    return layer
