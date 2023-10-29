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
    elif norm_type is None:
        return nn.Identity()
        
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
            # Depthwise Conv
            assert c1 == c2
            convs.append(get_conv2d(c1, c2, k=k, p=p, s=s, d=d, g=c1, bias=add_bias))
            # depthwise conv
            if norm_type:
                convs.append(get_norm(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))
        else:
            # Naive Conv
            convs.append(get_conv2d(c1, c2, k=k, p=p, s=s, d=d, g=1, bias=add_bias))
            if norm_type:
                convs.append(get_norm(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))
            
        self.convs = nn.Sequential(*convs)


    def forward(self, x):
        return self.convs(x)


# ----------------------------  Modules ----------------------------
## Mixed ConvModule
class MixedConvModule(nn.Module):
    def __init__(self,
                 in_dim       :int,
                 out_dim      :int,
                 expand_ratio :float = 0.25,
                 num_branches :int   = 4,
                 shortcut     :bool  = True,
                 act_type     :str   = 'relu',
                 norm_type    :str   = 'BN',
                 depthwise    :bool  = False):
        super(MixedConvModule, self).__init__()
        # ----------- Basic Parameters -----------
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.expand_ratio = expand_ratio
        self.num_branches = num_branches
        self.shortcut = shortcut
        self.inter_dim = round(in_dim * expand_ratio)
        # ----------- Network Parameters -----------
        self.input_proj = Conv(in_dim, self.inter_dim, k=1, act_type=None, norm_type=norm_type)
        self.branches = nn.ModuleList([
            Conv(self.inter_dim, self.inter_dim, k=3, p=1, s=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
            for _ in range(num_branches)])
        self.output_proj = Conv(self.inter_dim * self.num_branches, out_dim, k=1, act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        y = self.input_proj(x)
        outs = []
        for layer in self.branches:
            y = layer(y)
            outs.append(y)
        outs = torch.cat(outs, dim=1)

        return x + self.output_proj(outs) if self.shortcut else self.output_proj(outs)

## Conv-style FFN
class ConvFFN(nn.Module):
    def __init__(self,
                 in_dim       :int,
                 out_dim      :int,
                 expand_ratio :float = 2.0,
                 shortcut     :bool  = True,
                 act_type     :str   = 'silu',
                 norm_type    :str   = 'BN',
                 depthwise    :bool  = False):
        super(ConvFFN, self).__init__()
        # ----------- Basic Parameters -----------
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.shortcut = shortcut
        self.expand_dim = round(in_dim * expand_ratio)
        # ----------- Network Parameters -----------
        self.conv_ffn = nn.Sequential(
            Conv(in_dim, self.expand_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(self.expand_dim, in_dim, k=1, act_type=None, norm_type=norm_type)
        )

    def forward(self, x):
        return x + self.conv_ffn(x) if self.shortcut else self.conv_ffn(x)

## ResBlock
class ResXBlock(nn.Module):
    def __init__(self,
                 in_dim       :int,
                 out_dim      :int,
                 expand_ratio :float = 0.25,
                 ffn_ratio    :float = 2.0,
                 num_branches :int   = 4,
                 shortcut     :bool  = True,
                 act_type     :str   ='silu',
                 norm_type    :str   ='BN',
                 depthwise    :bool  = False):
        super(ResXBlock, self).__init__()
        self.layer1 = MixedConvModule(in_dim, out_dim, expand_ratio, num_branches, shortcut, act_type, norm_type, depthwise)
        self.layer2 = ConvFFN(out_dim, out_dim, ffn_ratio, shortcut, act_type, norm_type, depthwise)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

## ResXStage
class ResXStage(nn.Module):
    def __init__(self,
                 in_dim       :int,
                 out_dim      :int,
                 expand_ratio :float = 0.25,
                 ffn_ratio    :float = 2.0,
                 num_branches :int   = 4,
                 num_blocks   :int   = 1,
                 shortcut     :bool  = True,
                 act_type     :str   ='silu',
                 norm_type    :str   ='BN',
                 depthwise    :bool  = False):
        super(ResXStage, self).__init__()
        stages = []
        for i in range(num_blocks):
            if i == 0:
                stages.append(ResXBlock(in_dim, out_dim, expand_ratio, ffn_ratio, num_branches, shortcut, act_type, norm_type, depthwise))
            else:
                stages.append(ResXBlock(out_dim, out_dim, expand_ratio, ffn_ratio, num_branches, shortcut, act_type, norm_type, depthwise))
        self.stages = nn.Sequential(*stages)

    def forward(self, x):
        return self.stages(x)
