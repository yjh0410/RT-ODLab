import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------- NMS ----------------------------
## basic NMS
def nms(bboxes, scores, nms_thresh):
    """"Pure Python NMS."""
    x1 = bboxes[:, 0]  #xmin
    y1 = bboxes[:, 1]  #ymin
    x2 = bboxes[:, 2]  #xmax
    y2 = bboxes[:, 3]  #ymax

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # compute iou
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-10, xx2 - xx1)
        h = np.maximum(1e-10, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
        #reserve all the boundingbox whose ovr less than thresh
        inds = np.where(iou <= nms_thresh)[0]
        order = order[inds + 1]

    return keep

## class-agnostic NMS 
def multiclass_nms_class_agnostic(scores, labels, bboxes, nms_thresh):
    # nms
    keep = nms(bboxes, scores, nms_thresh)
    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return scores, labels, bboxes

## class-aware NMS 
def multiclass_nms_class_aware(scores, labels, bboxes, nms_thresh, num_classes):
    # nms
    keep = np.zeros(len(bboxes), dtype=np.int32)
    for i in range(num_classes):
        inds = np.where(labels == i)[0]
        if len(inds) == 0:
            continue
        c_bboxes = bboxes[inds]
        c_scores = scores[inds]
        c_keep = nms(c_bboxes, c_scores, nms_thresh)
        keep[inds[c_keep]] = 1
    keep = np.where(keep > 0)
    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return scores, labels, bboxes

## multi-class NMS 
def multiclass_nms(scores, labels, bboxes, nms_thresh, num_classes, class_agnostic=False):
    if class_agnostic:
        return multiclass_nms_class_agnostic(scores, labels, bboxes, nms_thresh)
    else:
        return multiclass_nms_class_aware(scores, labels, bboxes, nms_thresh, num_classes)


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
    def __init__(self, d_model=256, ffn_dim=1024, dropout=0., act_type='relu'):
        super().__init__()
        self.ffn_dim = ffn_dim
        self.linear1 = nn.Linear(d_model, self.ffn_dim)
        self.activation = get_activation(act_type)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(self.ffn_dim, d_model)
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
                 depthwise :bool = False
                ):
        super(BasicConv, self).__init__()
        add_bias = False if norm_type else True
        self.depthwise = depthwise
        if not depthwise:
            self.conv = get_conv2d(in_dim, out_dim, k=kernel_size, p=padding, s=stride, g=1, bias=add_bias)
            self.norm = get_norm(norm_type, out_dim)
        else:
            self.conv1 = get_conv2d(in_dim, in_dim, k=kernel_size, p=padding, s=stride, g=1, bias=add_bias)
            self.norm1 = get_norm(norm_type, in_dim)
            self.conv2 = get_conv2d(in_dim, out_dim, k=kernel_size, p=padding, s=stride, g=1, bias=add_bias)
            self.norm2 = get_norm(norm_type, out_dim)
        self.act  = get_activation(act_type)

    def forward(self, x):
        if not self.depthwise:
            return self.act(self.norm(self.conv(x)))
        else:
            # Depthwise conv
            x = self.norm1(self.conv1(x))
            # Pointwise conv
            x = self.norm2(self.conv2(x))
            return x


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
        paddings = [k // 2 for k in kernel_sizes]
        self.cv1 = BasicConv(in_dim, inter_dim,
                             kernel_size=kernel_sizes[0], padding=paddings[0],
                             act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.cv2 = BasicConv(inter_dim, out_dim,
                             kernel_size=kernel_sizes[1], padding=paddings[1],
                             act_type=act_type, norm_type=norm_type, depthwise=depthwise)
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
        self.conv1 = BasicConv(in_dim, self.inter_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.conv2 = BasicConv(in_dim, self.inter_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.cmodules = nn.ModuleList([Bottleneck(self.inter_dim, self.inter_dim,
                                                   1.0, [3, 3], shortcut,
                                                   act_type, norm_type, depthwise)
                                                   for _ in range(num_blocks)])
        self.conv3 = BasicConv(self.inter_dim * (2 + num_blocks), out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        x1, x2 = self.conv1(x), self.conv2(x)
        out = [x1, x2]
        for m in self.cmodules:
            x2 = m(x2)
            out.append(x2)

        return self.conv3(torch.cat(out, dim=1))
    