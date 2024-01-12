import math
import torch
import torch.nn as nn
import torchvision.ops


# --------------------- Basic modules ---------------------
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
    else:
        raise NotImplementedError
        
def get_norm(norm_type, dim):
    if norm_type == 'BN':
        return nn.BatchNorm2d(dim)
    elif norm_type == 'GN':
        return nn.GroupNorm(num_groups=32, num_channels=dim)
    elif norm_type is None:
        return nn.Identity()
    else:
        raise NotImplementedError

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

class DeConv(nn.Module):
    def __init__(self,
                 in_dim      :int,
                 out_dim     :int,
                 kernel_size :int  = 4,
                 stride      :int  = 2,
                 act_type    :str  = 'silu',
                 norm_type   :str  = 'BN'
                 ):
        super(DeConv, self).__init__()
        # ----------- Basic parameters -----------
        if kernel_size == 4:
            padding = 1
            output_padding = 0
        elif kernel_size == 3:
            padding = 1
            output_padding = 1
        elif kernel_size == 2:
            padding = 0
            output_padding = 0

        # ----------- Network parameters -----------
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, output_padding=output_padding),
            get_norm(norm_type, out_dim),
            get_activation(act_type)
        )

    def forward(self, x):
        return self.convs(x)
    
class DeformableConv(nn.Module):
    def __init__(self,
                 in_dim  :int,
                 out_dim :int,
                 kernel_size  :int = 3,
                 stride       :int = 1,
                 padding      :int = 1):
        super(DeformableConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        # init weight and bias
        self.weight = nn.Parameter(torch.Tensor(out_dim, in_dim, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_dim))

        # offset conv
        self.conv_offset_mask = nn.Conv2d(in_dim, 
                                          3 * kernel_size * kernel_size,
                                          kernel_size=kernel_size, 
                                          stride=stride,
                                          padding=self.padding, 
                                          bias=True)
        
        # init        
        self.reset_parameters()
        self._init_weight()


    def reset_parameters(self):
        n = self.in_dim * (self.kernel_size**2)
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()


    def _init_weight(self):
        # init offset_mask conv
        nn.init.constant_(self.conv_offset_mask.weight, 0.)
        nn.init.constant_(self.conv_offset_mask.bias, 0.)


    def forward(self, x):
        out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.weight, 
                                          bias=self.bias, 
                                          padding=self.padding,
                                          mask=mask,
                                          stride=self.stride)
        return x
    

# --------------------- Yolov8 modules ---------------------
## Yolov8-style BottleNeck
class Bottleneck(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio = 0.5,
                 kernel_sizes = [3, 3],
                 shortcut     = True,
                 act_type     = 'silu',
                 norm_type    = 'BN',
                 depthwise    = False,):
        super(Bottleneck, self).__init__()
        inter_dim = int(out_dim * expand_ratio)  # hidden channels            
        self.cv1 = Conv(in_dim, inter_dim, k=kernel_sizes[0], p=kernel_sizes[0]//2, norm_type=norm_type, act_type=act_type, depthwise=depthwise)
        self.cv2 = Conv(inter_dim, out_dim, k=kernel_sizes[1], p=kernel_sizes[1]//2, norm_type=norm_type, act_type=act_type, depthwise=depthwise)
        self.shortcut = shortcut and in_dim == out_dim

    def forward(self, x):
        h = self.cv2(self.cv1(x))

        return x + h if self.shortcut else h

# Yolov8-style StageBlock
class RTCBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_blocks = 1,
                 shortcut   = False,
                 act_type   = 'silu',
                 norm_type  = 'BN',
                 depthwise  = False,):
        super(RTCBlock, self).__init__()
        self.inter_dim = out_dim // 2
        self.input_proj = Conv(in_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.m = nn.Sequential(*(
            Bottleneck(self.inter_dim, self.inter_dim, 1.0, [3, 3], shortcut, act_type, norm_type, depthwise)
            for _ in range(num_blocks)))
        self.output_proj = Conv((2 + num_blocks) * self.inter_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        # Input proj
        x1, x2 = torch.chunk(self.input_proj(x), 2, dim=1)
        out = list([x1, x2])

        # Bottlenecl
        out.extend(m(out[-1]) for m in self.m)

        # Output proj
        out = self.output_proj(torch.cat(out, dim=1))

        return out
