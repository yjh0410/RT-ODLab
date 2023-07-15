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


# ---------------------------- Core Modules ----------------------------
## Scale Modulation Block
class SMBlock(nn.Module):
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, act_type='silu', norm_type='BN', depthwise=False):
        super(SMBlock, self).__init__()
        # -------------- Basic parameters --------------
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.expand_ratio = expand_ratio
        self.inter_dim = round(in_dim * expand_ratio)
        # -------------- Network parameters --------------
        ## Scale Modulation
        self.sm0 = Conv(self.inter_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.sm1 = Conv(self.inter_dim, self.inter_dim, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.sm2 = Conv(self.inter_dim, self.inter_dim, k=5, p=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.sm3 = Conv(self.inter_dim, self.inter_dim, k=7, p=3, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        ## Output proj
        self.cv3 = Conv(self.inter_dim*4, out_dim, k=1, act_type=act_type, norm_type=norm_type)


    def channel_shuffle(self, x, groups):
        # type: (torch.Tensor, int) -> torch.Tensor
        batchsize, num_channels, height, width = x.data.size()
        per_group_dim = num_channels // groups

        # reshape
        x = x.view(batchsize, groups, per_group_dim, height, width)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x
    

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        x3 = self.sm1(self.sm0(x2))
        x4 = self.sm2(x3)
        x5 = self.sm3(x4)
        out = torch.cat([x1, x3, x4, x5], dim=1)
        out = self.cv3(out)

        return self.channel_shuffle(out, groups=4)


# ---------------------------- FPN Modules ----------------------------
## build fpn's core block
def build_fpn_block(cfg, in_dim, out_dim):
    if cfg['fpn_core_block'] == 'smblock':
        layer = SMBlock(in_dim=in_dim,
                        out_dim=out_dim,
                        expand_ratio=cfg['fpn_expand_ratio'],
                        act_type=cfg['fpn_act'],
                        norm_type=cfg['fpn_norm'],
                        depthwise=cfg['fpn_depthwise']
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
    elif cfg['fpn_downsample_layer'] == 'maxpool':
        assert in_dim == out_dim
        layer = nn.MaxPool2d((2, 2), stride=2)
        
    return layer