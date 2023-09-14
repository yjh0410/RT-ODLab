import numpy as np
import torch
import torch.nn as nn


# ---------------------------- 2D CNN ----------------------------
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
        p = p if d == 1 else d
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


# ---------------------------- Modified YOLOv7's Modules ----------------------------
## ELANBlock for Backbone
class ELANBlock(nn.Module):
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, depth=1.0, act_type='silu', norm_type='BN', depthwise=False):
        super(ELANBlock, self).__init__()
        if isinstance(expand_ratio, float):
            inter_dim = int(in_dim * expand_ratio)
            inter_dim2 = inter_dim
        elif isinstance(expand_ratio, list):
            assert len(expand_ratio) == 2
            e1, e2 = expand_ratio
            inter_dim = int(in_dim * e1)
            inter_dim2 = int(inter_dim * e2)
        # branch-1
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        # branch-2
        self.cv2 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        # branch-3
        for idx in range(round(3*depth)):
            if idx == 0:
                cv3 = [Conv(inter_dim, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)]
            else:
                cv3.append(Conv(inter_dim2, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise))
        self.cv3 = nn.Sequential(*cv3)
        # branch-4
        self.cv4 = nn.Sequential(*[
            Conv(inter_dim2, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
            for _ in range(round(3*depth))
        ])
        # output
        self.out = Conv(inter_dim*2 + inter_dim2*2, out_dim, k=1, act_type=act_type, norm_type=norm_type)


    def forward(self, x):
        """
        Input:
            x: [B, C_in, H, W]
        Output:
            out: [B, C_out, H, W]
        """
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
                self.cv3.append(Conv(self.inter_dim1, self.inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise))
            else:
                self.cv3.append(Conv(self.inter_dim2, self.inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise))
        self.cv3 = nn.Sequential(*self.cv3)
        ## branch-4
        self.cv4 = nn.Sequential(*[
            Conv(self.inter_dim2, self.inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
            for _ in range(branch_depth)
        ])
        ## branch-5
        self.cv5 = nn.Sequential(*[
            Conv(self.inter_dim2, self.inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
            for _ in range(branch_depth)
        ])
        ## branch-6
        self.cv6 = nn.Sequential(*[
            Conv(self.inter_dim2, self.inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
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
    
## DownSample
class DSBlock(nn.Module):
    def __init__(self, in_dim, out_dim, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        inter_dim = out_dim // 2
        self.mp = nn.MaxPool2d((2, 2), 2)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(inter_dim, inter_dim, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )

    def forward(self, x):
        x1 = self.cv1(self.mp(x))
        x2 = self.cv2(x)
        out = torch.cat([x1, x2], dim=1)

        return out


# ---------------------------- FPN Modules ----------------------------
## build fpn's core block
def build_fpn_block(cfg, in_dim, out_dim):
    if cfg['fpn_core_block'] == 'elanblock':
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
        layer = Conv(in_dim, out_dim, k=3, s=2, p=1, act_type=cfg['fpn_act'], norm_type=cfg['fpn_norm'])
    elif cfg['fpn_downsample_layer'] == 'dsblock':
        layer = DSBlock(in_dim, out_dim, act_type=cfg['fpn_act'], norm_type=cfg['fpn_norm'])
        
    return layer