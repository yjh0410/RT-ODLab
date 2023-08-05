import torch
import torch.nn as nn

from .rtcdet_v2_basic import Conv


# Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
class SPPF(nn.Module):
    """
        This code referenced to https://github.com/ultralytics/yolov5
    """
    def __init__(self, cfg, in_dim, out_dim, expand_ratio=0.5):
        super().__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.out_dim = out_dim
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=cfg['neck_act'], norm_type=cfg['neck_norm'])
        self.cv2 = Conv(inter_dim * 4, out_dim, k=1, act_type=cfg['neck_act'], norm_type=cfg['neck_norm'])
        self.m = nn.MaxPool2d(kernel_size=cfg['pooling_size'], stride=1, padding=cfg['pooling_size'] // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)

        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

# Mixed Spatial Pyramid Pooling
class MixedSPP(nn.Module):
    def __init__(self, cfg, in_dim, out_dim, expand_ratio=2.0):
        super().__init__()
        # ------------- Basic parameters -------------
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.expand_dim = round(in_dim * expand_ratio)
        self.num_maxpools = len(cfg['pooling_size']) + 1
        # ------------- Network parameters -------------
        self.input_proj = Conv(in_dim, self.expand_dim, k=1, act_type=cfg['neck_act'], norm_type=cfg['neck_norm'])
        self.maxpools = nn.ModuleList()
        for ksize in cfg['pooling_size']:
            self.maxpools.append(nn.MaxPool2d(kernel_size=ksize, stride=1, padding=ksize// 2))
        self.output_proj = Conv(self.expand_dim, out_dim, k=1, act_type=cfg['neck_act'], norm_type=cfg['neck_norm'])

    def forward(self, x):
        x = self.input_proj(x)
        x_chunks = torch.chunk(x, self.num_maxpools, dim=1)
        out = [x_chunks[0]]
        for x_chunk, maxpool in zip(x_chunks[1:], self.maxpools):
            out.append(maxpool(x_chunk))
        out = torch.cat(out, dim=1)

        return self.output_proj(out)

# SPPF block with CSP module
class SPPFBlockCSP(nn.Module):
    """
        CSP Spatial Pyramid Pooling Block
    """
    def __init__(self, cfg, in_dim, out_dim, expand_ratio):
        super(SPPFBlockCSP, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.out_dim = out_dim
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=cfg['neck_act'], norm_type=cfg['neck_norm'])
        self.cv2 = Conv(in_dim, inter_dim, k=1, act_type=cfg['neck_act'], norm_type=cfg['neck_norm'])
        self.m = nn.Sequential(
            Conv(inter_dim, inter_dim, k=3, p=1, 
                 act_type=cfg['neck_act'], norm_type=cfg['neck_norm'], 
                 depthwise=cfg['neck_depthwise']),
            SPPF(cfg, inter_dim, inter_dim, expand_ratio=1.0),
            Conv(inter_dim, inter_dim, k=3, p=1, 
                 act_type=cfg['neck_act'], norm_type=cfg['neck_norm'], 
                 depthwise=cfg['neck_depthwise'])
        )
        self.cv3 = Conv(inter_dim * 2, self.out_dim, k=1, act_type=cfg['neck_act'], norm_type=cfg['neck_norm'])

        
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m(x2)
        y = self.cv3(torch.cat([x1, x3], dim=1))

        return y


# build neck
def build_neck(cfg, in_dim, out_dim):
    model = cfg['neck']
    print('==============================')
    print('Neck: {}'.format(model))
    # build neck
    if model == 'sppf':
        neck = SPPF(cfg, in_dim, out_dim, cfg['neck_expand_ratio'])
    elif model == 'csp_sppf':
        neck = SPPFBlockCSP(cfg, in_dim, out_dim, cfg['neck_expand_ratio'])
    elif model == 'mixed_spp':
        neck = MixedSPP(cfg, in_dim, out_dim, cfg['neck_expand_ratio'])

    return neck
        