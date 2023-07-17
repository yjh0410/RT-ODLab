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
## MultiHeadMixedConv
class MultiHeadMixedConv(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, shortcut=False, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        # -------------- Basic parameters --------------
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.shortcut = shortcut
        # -------------- Network parameters --------------
        ## Scale Modulation
        self.mixed_convs = nn.ModuleList([
            Conv(self.head_dim, self.head_dim, k=2*i+1, p=i, act_type=None, norm_type=None, depthwise=depthwise)
            for i in range(1, num_heads+1)])
        ## Aggregation proj
        self.out_proj = Conv(self.head_dim*num_heads, out_dim, k=1, act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        xs = torch.chunk(x, self.num_heads, dim=1)
        ys = [mixed_conv(x_h) for x_h, mixed_conv in zip(xs, self.mixed_convs)]
        ys = self.out_proj(torch.cat(ys, dim=1))

        return x + ys if self.shortcut else ys
    

## Scale Modulation Block
class SMBlock(nn.Module):
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
        self.cv1 = Conv(self.inter_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(self.inter_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        ## Scale Modulation
        self.smblocks = nn.Sequential(*[
            MultiHeadMixedConv(self.inter_dim, self.inter_dim, self.num_heads, self.shortcut, act_type, norm_type, depthwise)
            for _ in range(nblocks)])
        ## Output proj
        self.out_proj = Conv(self.inter_dim*2, out_dim, k=1, act_type=act_type, norm_type=norm_type)


    def forward(self, x):
        """
        Input:
            x: (Tensor) -> [B, C_in, H, W]
        Output:
            out: (Tensor) -> [B, C_out, H, W]
        """
        x1, x2 = torch.chunk(x, 2, dim=1)
        # branch-1
        x1 = self.cv1(x1)
        # branch-2
        x2 = self.smblocks(x2)
        # output
        out = torch.cat([x1, x2], dim=1)
        out = self.out_proj(out)

        return out


## DownSample Block
class DSBlock(nn.Module):
    def __init__(self, in_dim, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        # branch-1
        self.maxpool = nn.MaxPool2d((2, 2), 2)
        # branch-2
        inter_dim = in_dim // 2
        self.sm1 = Conv(inter_dim, inter_dim, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.sm2 = Conv(inter_dim, inter_dim, k=5, p=2, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.sm3 = Conv(inter_dim, inter_dim, k=7, p=3, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)


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
        """
        Input:
            x: (Tensor) -> [B, C, H, W]
        Output:
            out: (Tensor) -> [B, 2C, H/2, W/2]
        """
        x1, x2 = torch.chunk(x, 2, dim=1)
        # branch-1
        x1 = self.maxpool(x1)
        # branch-2
        x2 = torch.cat([self.sm1(x2), self.sm2(x2), self.sm3(x2)], dim=1)
        # channel shuffle
        out = torch.cat([x1, x2], dim=1)
        out = self.channel_shuffle(out, groups=4)

        return out


# ---------------------------- FPN Modules ----------------------------
## build fpn's core block
def build_fpn_block(cfg, in_dim, out_dim):
    if cfg['fpn_core_block'] == 'smblock':
        layer = SMBlock(in_dim=in_dim,
                        out_dim=out_dim,
                        nblocks=cfg['fpn_nblocks'],
                        num_heads=cfg['fpn_num_heads'],
                        shortcut=cfg['fpn_shortcut'],
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
        layer = Conv(in_dim, out_dim, k=3, s=2, p=1, act_type=cfg['fpn_act'], norm_type=cfg['fpn_norm'], depthwise=cfg['fpn_depthwise'])
    elif cfg['fpn_downsample_layer'] == 'maxpool':
        assert in_dim == out_dim
        layer = nn.MaxPool2d((2, 2), stride=2)
        
    return layer