import torch
import torch.nn as nn


# ----------------- MLP modules -----------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([in_dim] + h, h + [out_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class FFN(nn.Module):
    def __init__(self, d_model=256, mlp_ratio=4.0, dropout=0., act_type='relu'):
        super().__init__()
        self.fpn_dim = round(d_model * mlp_ratio)
        self.linear1 = nn.Linear(d_model, self.fpn_dim)
        self.activation = get_activation(act_type)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(self.fpn_dim, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm(src)
        
        return src
    

# ----------------- Basic CNN Ops -----------------
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
    elif act_type == 'gelu':
        return nn.GELU()
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

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class FrozenBatchNorm2d(torch.nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias
    
class BasicConv(nn.Module):
    def __init__(self, 
                 in_dim,                   # in channels
                 out_dim,                  # out channels 
                 kernel_size=1,            # kernel size 
                 padding=0,                # padding
                 stride=1,                 # padding
                 act_type  :str = 'lrelu', # activation
                 norm_type :str = 'BN',    # normalization
                ):
        super(BasicConv, self).__init__()
        add_bias = False if norm_type else True
        self.conv = get_conv2d(in_dim, out_dim, k=kernel_size, p=padding, s=stride, g=1, bias=add_bias)
        self.norm = get_norm(norm_type, out_dim)
        self.act  = get_activation(act_type)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class DepthwiseConv(nn.Module):
    def __init__(self, 
                 in_dim,                 # in channels
                 out_dim,                # out channels 
                 kernel_size=1,          # kernel size 
                 padding=0,              # padding
                 stride=1,               # padding
                 act_type  :str = None,  # activation
                 norm_type :str = 'BN',  # normalization
                ):
        super(DepthwiseConv, self).__init__()
        assert in_dim == out_dim
        add_bias = False if norm_type else True
        self.conv = get_conv2d(in_dim, out_dim, k=kernel_size, p=padding, s=stride, g=out_dim, bias=add_bias)
        self.norm = get_norm(norm_type, out_dim)
        self.act  = get_activation(act_type)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class PointwiseConv(nn.Module):
    def __init__(self, 
                 in_dim,                   # in channels
                 out_dim,                  # out channels 
                 act_type  :str = 'lrelu', # activation
                 norm_type :str = 'BN',    # normalization
                ):
        super(DepthwiseConv, self).__init__()
        assert in_dim == out_dim
        add_bias = False if norm_type else True
        self.conv = get_conv2d(in_dim, out_dim, k=1, p=0, s=1, g=1, bias=add_bias)
        self.norm = get_norm(norm_type, out_dim)
        self.act  = get_activation(act_type)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))



# ----------------- CNN Modules -----------------
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
        inter_dim = int(out_dim * expand_ratio)
        if depthwise:
            self.cv1 = nn.Sequential(
                DepthwiseConv(in_dim, in_dim, kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2, act_type=act_type, norm_type=norm_type),
                PointwiseConv(in_dim, inter_dim, act_type=act_type, norm_type=norm_type),
            )
            self.cv2 = nn.Sequential(
                DepthwiseConv(inter_dim, inter_dim, kernel_size=kernel_sizes[1], padding=kernel_sizes[1]//2, act_type=act_type, norm_type=norm_type),
                PointwiseConv(inter_dim, out_dim, act_type=act_type, norm_type=norm_type),
            )
        else:
            self.cv1 = BasicConv(in_dim, inter_dim,  kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2, act_type=act_type, norm_type=norm_type)
            self.cv2 = BasicConv(inter_dim, out_dim, kernel_size=kernel_sizes[1], padding=kernel_sizes[1]//2, act_type=act_type, norm_type=norm_type)
        self.shortcut = shortcut and in_dim == out_dim

    def forward(self, x):
        h = self.cv2(self.cv1(x))

        return x + h if self.shortcut else h

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

        # Bottlenecl
        out.extend(m(out[-1]) for m in self.m)

        # Output proj
        out = self.output_proj(torch.cat(out, dim=1))

        return out

class RepVggBlock(nn.Module):
    def __init__(self, in_dim, out_dim, act_type='relu', norm_type='BN', alpha=False):
        super(RepVggBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv1 = BasicConv(in_dim, out_dim, kernel_size=3, padding=1, act_type=None, norm_type=norm_type)
        self.conv2 = BasicConv(in_dim, out_dim, kernel_size=3, padding=1, act_type=None, norm_type=norm_type)
        self.act = get_activation(act_type)

        if alpha:
            self.alpha = nn.Parameter(torch.as_tensor([1.0]).float())
        else:
            self.alpha = None

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            if self.alpha:
                y = self.conv1(x) + self.alpha * self.conv2(x)
            else:
                y = self.conv1(x) + self.conv2(x)
        y = self.act(y)
        return y

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(
                self.in_dim,
                self.out_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1)
        kernel, bias = self.get_equivalent_kernel_bias()
        # self.conv.weight.set_value(kernel)
        # self.conv.bias.set_value(bias)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        self.__delattr__('conv1')
        self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        if self.alpha:
            return kernel3x3 + self.alpha * self._pad_1x1_to_3x3_tensor(
                kernel1x1), bias3x3 + self.alpha * bias1x1
        else:
            return kernel3x3 + self._pad_1x1_to_3x3_tensor(
                kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.bn._mean
        running_var = branch.bn._variance
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))

        return kernel * t, beta - running_mean * gamma / std

class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_dim     :int,
                 out_dim    :int,
                 num_blocks :int   = 3,
                 expansion  :float = 1.0,
                 act_type   :str   ="silu",
                 norm_type  :str   = 'BN'):
        super(CSPRepLayer, self).__init__()
        hidden_dim = int(out_dim * expansion)
        self.conv1 = BasicConv(
            in_dim, hidden_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.conv2 = BasicConv(
            in_dim, hidden_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(
                hidden_dim, hidden_dim, act_type=act_type, norm_type=norm_type)
            for _ in range(num_blocks)
        ])
        if hidden_dim != out_dim:
            self.conv3 = BasicConv(hidden_dim, out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)

        return self.conv3(x_1 + x_2)
