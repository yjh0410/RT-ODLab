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
    def __init__(self, in_dim, out_dim=None, act_type='silu', norm_type='BN', depthwise=False):
        super(SMBlock, self).__init__()
        # -------------- Basic parameters --------------
        self.in_dim = in_dim
        self.inter_dim = in_dim // 2
        # -------------- Network parameters --------------
        self.cv1 = Conv(self.inter_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(self.inter_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        ## Scale Modulation
        self.sm1 = nn.Sequential(
            Conv(self.inter_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(self.inter_dim, self.inter_dim, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
            )
        self.sm2 = nn.Sequential(
            Conv(self.inter_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(self.inter_dim, self.inter_dim, k=5, p=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
            )
        self.sm3 = nn.Sequential(
            Conv(self.inter_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(self.inter_dim, self.inter_dim, k=7, p=3, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
            )
        ## Aggregation proj
        self.sm_aggregation = Conv(self.inter_dim*3, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type)

        # Output proj
        self.out_proj = None
        if out_dim is not None:
            self.out_proj = Conv(self.inter_dim*2, out_dim, k=1, act_type=act_type, norm_type=norm_type)


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
            x: (Tensor) -> [B, C_in, H, W]
        Output:
            out: (Tensor) -> [B, C_out, H, W]
        """
        x1, x2 = torch.chunk(x, 2, dim=1)
        # branch-1
        x1 = self.cv1(x1)
        # branch-2
        x2 = self.cv2(x2)
        x2 = torch.cat([self.sm1(x2), self.sm2(x2), self.sm3(x2)], dim=1)
        x2 = self.sm_aggregation(x2)
        # channel shuffle
        out = torch.cat([x1, x2], dim=1)
        out = self.channel_shuffle(out, groups=2)

        if self.out_proj:
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
        self.sm_aggregation = Conv(inter_dim*3, inter_dim*3, k=1, act_type=act_type, norm_type=norm_type)


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
        x2 = self.sm_aggregation(x2)
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