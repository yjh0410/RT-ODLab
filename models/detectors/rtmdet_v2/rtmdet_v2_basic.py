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


# ---------------------------- Base Modules ----------------------------
## Multi-head Mixed Conv (MHMC)
class MultiHeadMixedConv(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, shortcut=False, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        # -------------- Basic parameters --------------
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.shortcut = shortcut
        self.head_dim = in_dim // num_heads
        # -------------- Network parameters --------------
        ## Scale Modulation
        self.mixed_convs = nn.ModuleList()
        for i in range(num_heads):
            self.mixed_convs.append(
                Conv(self.head_dim, self.head_dim, k=2*i+1, p=i, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
            )
        ## Out-proj
        self.out_proj = Conv(in_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type)


    def forward(self, x):
        xs = torch.chunk(x, self.num_heads, dim=1)
        ys = [mixed_conv(x_h) for x_h, mixed_conv in zip(xs, self.mixed_convs)]
        out = self.out_proj(torch.cat(ys, dim=1))

        return out + x if self.shortcut else out

# ---------------------------- Base Blocks ----------------------------
## Mixed Convolution Block
class MCBlock(nn.Module):
    def __init__(self, in_dim, out_dim, nblocks=1, num_heads=4, shortcut=False, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        # -------------- Basic parameters --------------
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nblocks = nblocks
        self.num_heads = num_heads
        self.shortcut = shortcut
        self.inter_dim = in_dim // 2
        # -------------- Network parameters --------------
        ## branch-1
        self.cv1 = Conv(self.in_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(self.in_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        ## branch-2
        self.smblocks = nn.Sequential(*[
            MultiHeadMixedConv(self.inter_dim, self.inter_dim, self.num_heads, self.shortcut, act_type, norm_type, depthwise)
            for _ in range(nblocks)])
        ## out proj
        self.out_proj = Conv(self.inter_dim*2, out_dim, k=1, act_type=act_type, norm_type=norm_type)


    def forward(self, x):
        # branch-1
        x1 = self.cv1(x)
        # branch-2
        x2 = self.smblocks(self.cv2(x))
        # output
        out = torch.cat([x1, x2], dim=1)
        out = self.out_proj(out)

        return out

## DownSample Block
class DSBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inter_dim = out_dim // 2
        self.num_heads = num_heads
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
    if cfg['fpn_core_block'] == 'mcblock':
        layer = MCBlock(in_dim=in_dim,
                        out_dim=out_dim,
                        nblocks=round(cfg['depth'] * 3),
                        num_heads=cfg['fpn_num_heads'],
                        shortcut=False,
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
        layer = Conv(in_dim, out_dim, k=3, s=2, p=1,
                     act_type=cfg['fpn_act'], norm_type=cfg['fpn_norm'], depthwise=cfg['fpn_depthwise'])
    elif cfg['fpn_downsample_layer'] == 'maxpool':
        assert in_dim == out_dim
        layer = nn.MaxPool2d((2, 2), stride=2)
    elif cfg['fpn_downsample_layer'] == 'dsblock':
        layer = DSBlock(in_dim, out_dim, num_heads=cfg['fpn_num_heads'],
                        act_type=cfg['fpn_act'], norm_type=cfg['fpn_norm'], depthwise=cfg['fpn_depthwise'])
        
    return layer
