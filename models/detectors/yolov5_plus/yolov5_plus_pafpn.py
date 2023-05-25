import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolov5_plus_basic import (Conv, build_downsample_layer, build_fpn_block)


# YOLO-Style PaFPN
class Yolov5PlusPaFPN(nn.Module):
    def __init__(self, cfg, in_dims=[256, 512, 1024], out_dim=None):
        super(Yolov5PlusPaFPN, self).__init__()
        # --------------------------- Basic Parameters ---------------------------
        self.in_dims = in_dims
        c3, c4, c5 = in_dims
        width = cfg['width']
        ratio = cfg['ratio']

        # --------------------------- Network Parameters ---------------------------
        ## top dwon
        ### P5 -> P4
        self.top_down_layer_1 = build_fpn_block(cfg, c4 + c5, round(512*width))

        ### P4 -> P3
        self.top_down_layer_2 = build_fpn_block(cfg, c3 + round(512*width), round(256*width))

        ## bottom up
        ### P3 -> P4
        self.downsample_layer_1 = build_downsample_layer(cfg, round(256*width), round(256*width))
        self.bottom_up_layer_1 = build_fpn_block(cfg, round(256*width) + round(512*width), round(512*width))

        ### P4 -> P5
        self.downsample_layer_2 = build_downsample_layer(cfg, round(512*width), round(512*width))
        self.bottom_up_layer_2 = build_fpn_block(cfg, c5 + round(512*width), round(512*width*ratio))
                
        ## output proj layers
        if out_dim is not None:
            self.out_layers = nn.ModuleList([
                Conv(in_dim, out_dim, k=1,
                     act_type=cfg['fpn_act'], norm_type=cfg['fpn_norm'])
                     for in_dim in [round(256*width), round(512*width), round(512*width*ratio)]
                     ])
            self.out_dim = [out_dim] * 3
        else:
            self.out_layers = None
            self.out_dim = [round(256*width), round(512*width), round(512*width*ratio)]


    def forward(self, features):
        c3, c4, c5 = features

        # Top down
        ## P5 -> P4
        c6 = F.interpolate(c5, scale_factor=2.0)
        c7 = torch.cat([c6, c4], dim=1)
        c8 = self.top_down_layer_1(c7)
        ## P4 -> P3
        c9 = F.interpolate(c8, scale_factor=2.0)
        c10 = torch.cat([c9, c3], dim=1)
        c11 = self.top_down_layer_2(c10)

        # Bottom up
        ## p3 -> P4
        c12 = self.downsample_layer_1(c11)
        c13 = torch.cat([c12, c8], dim=1)
        c14 = self.bottom_up_layer_1(c13)
        ## P4 -> P5
        c15 = self.downsample_layer_2(c14)
        c16 = torch.cat([c15, c5], dim=1)
        c17 = self.bottom_up_layer_2(c16)

        out_feats = [c11, c14, c17] # [P3, P4, P5]
        
        # output proj layers
        if self.out_layers is not None:
            out_feats_proj = []
            for feat, layer in zip(out_feats, self.out_layers):
                out_feats_proj.append(layer(feat))
            return out_feats_proj

        return out_feats


def build_fpn(cfg, in_dims, out_dim=None):
    model = cfg['fpn']
    # build pafpn
    if model == 'yolov5_plus_pafpn':
        fpn_net = Yolov5PlusPaFPN(cfg, in_dims, out_dim)

    return fpn_net
