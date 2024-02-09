import torch
import torch.nn as nn


# --------------------- Basic modules ---------------------
class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def get_conv2d(c1, c2, k, p, s, g, bias=False):
    conv = nn.Conv2d(c1, c2, k, stride=s, padding=p, groups=g, bias=bias)

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

class BasicConv(nn.Module):
    def __init__(self, 
                 in_dim,                     # in channels
                 out_dim,                    # out channels 
                 kernel_size :int = 1,       # kernel size 
                 padding     :int = 0,       # padding
                 stride      :int = 1,       # padding
                 act_type    :str = 'silu',  # activation
                 norm_type   :str = 'BN',    # normalization
                 depthwise   :bool = False,
                ):
        super(BasicConv, self).__init__()
        self.depthwise = depthwise
        add_bias = False if norm_type else True
        if not depthwise:
            self.conv = get_conv2d(in_dim, out_dim, k=kernel_size, p=padding, s=stride, g=1, bias=add_bias)
            self.norm = get_norm(norm_type, out_dim)
            self.act  = get_activation(act_type)
        else:
            self.conv1 = get_conv2d(in_dim, in_dim, k=kernel_size, p=padding, s=stride, g=in_dim, bias=add_bias)
            self.norm1 = get_norm(norm_type, in_dim)
            self.conv2 = get_conv2d(in_dim, out_dim, k=1, d=0, s=1, g=1, bias=add_bias)
            self.norm2 = get_norm(norm_type, out_dim)
        self.act  = get_activation(act_type)

    def forward(self, x):
        if not self.depthwise:
            return self.act(self.norm(self.conv(x)))
        else:
            return self.act(self.norm2(self.conv2(self.norm1(self.conv1(x)))))


# --------------------- Yolov8 modules ---------------------
## Yolov8 BottleNeck
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
        padding_sizes = [k // 2 for k in kernel_sizes]           
        self.cv1 = BasicConv(in_dim, inter_dim, kernel_size=kernel_sizes[0], padding=padding_sizes[0], act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.cv2 = BasicConv(inter_dim, out_dim, kernel_size=kernel_sizes[1], padding=padding_sizes[1], act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.shortcut = shortcut and in_dim == out_dim

    def forward(self, x):
        h = self.cv2(self.cv1(x))

        return x + h if self.shortcut else h

# Yolov8 StageBlock
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
        self.input_proj = BasicConv(in_dim, out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.m = nn.Sequential(*(
            Bottleneck(self.inter_dim, self.inter_dim, 1.0, [3, 3], shortcut, act_type, norm_type, depthwise)
            for _ in range(num_blocks)))
        self.output_proj = BasicConv((2 + num_blocks) * self.inter_dim, out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        # Input proj
        x1, x2 = torch.chunk(self.input_proj(x), 2, dim=1)
        out = list([x1, x2])

        # Bottleneck
        out.extend(m(out[-1]) for m in self.m)

        # Output proj
        out = self.output_proj(torch.cat(out, dim=1))

        return out
    