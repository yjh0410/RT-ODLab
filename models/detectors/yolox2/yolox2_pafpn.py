import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .yolox2_basic import Conv, Yolov8StageBlock
except:
    from yolox2_basic import Conv, Yolov8StageBlock


# PaFPN-ELAN
class Yolox2PaFPN(nn.Module):
    def __init__(self, 
                 in_dims   = [256, 512, 512],
                 out_dim   = None,
                 width     = 1.0,
                 depth     = 1.0,
                 ratio     = 1.0,
                 act_type  = 'silu',
                 norm_type = 'BN',
                 depthwise = False):
        super(Yolox2PaFPN, self).__init__()
        print('==============================')
        print('FPN: {}'.format("Yolov8 PaFPN"))
        # ---------------- Basic parameters ----------------
        self.in_dims = in_dims
        self.width = width
        self.depth = depth
        c3, c4, c5 = in_dims

        # ---------------- Top dwon ----------------
        ## P5 -> P4
        self.top_down_layer_1 = Yolov8StageBlock(in_dim       = c5 + c4,
                                                 out_dim      = round(512*width),
                                                 num_blocks   = round(3*depth),
                                                 shortcut     = False,
                                                 act_type     = act_type,
                                                 norm_type    = norm_type,
                                                 depthwise    = depthwise,
                                                 )
        ## P4 -> P3
        self.top_down_layer_2 = Yolov8StageBlock(in_dim       = round(512*width) + c3,
                                                 out_dim      = round(256*width),
                                                 num_blocks   = round(3*depth),
                                                 shortcut     = False,
                                                 act_type     = act_type,
                                                 norm_type    = norm_type,
                                                 depthwise    = depthwise,
                                                 )
        # ---------------- Bottom up ----------------
        ## P3 -> P4
        self.dowmsample_layer_1 = Conv(round(256*width), round(256*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.bottom_up_layer_1 = Yolov8StageBlock(in_dim       = round(256*width) + round(512*width),
                                                  out_dim      = round(512*width),
                                                  num_blocks   = round(3*depth),
                                                  shortcut     = False,
                                                  act_type     = act_type,
                                                  norm_type    = norm_type,
                                                  depthwise    = depthwise,
                                                  )
        ## P4 -> P5
        self.dowmsample_layer_2 = Conv(round(512*width), round(512*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.bottom_up_layer_2 = Yolov8StageBlock(in_dim       = round(512 * width) + c5,
                                                  out_dim      = round(512 * width * ratio),
                                                  num_blocks   = round(3*depth),
                                                  shortcut     = False,
                                                  act_type     = act_type,
                                                  norm_type    = norm_type,
                                                  depthwise    = depthwise,
                                                  )
        ## output proj layers
        if out_dim is not None:
            self.out_layers = nn.ModuleList([
                Conv(in_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type)
                     for in_dim in [round(256*width), round(512*width), round(512 * width * ratio)]
                     ])
            self.out_dim = [out_dim] * 3
        else:
            self.out_layers = None
            self.out_dim = [round(256*width), round(512*width), round(512 * width * ratio)]

        self.init_weights()
        
    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()

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
        # p3 -> P4
        c12 = self.dowmsample_layer_1(c11)
        c13 = torch.cat([c12, c8], dim=1)
        c14 = self.bottom_up_layer_1(c13)
        # P4 -> P5
        c15 = self.dowmsample_layer_2(c14)
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
    # build neck
    if model == 'yolox2_pafpn':
        fpn_net = Yolox2PaFPN(in_dims   = in_dims,
                              out_dim   = out_dim,
                              width     = cfg['width'],
                              depth     = cfg['depth'],
                              ratio     = cfg['ratio'],
                              act_type  = cfg['fpn_act'],
                              norm_type = cfg['fpn_norm'],
                              depthwise = cfg['fpn_depthwise']
                              )
    return fpn_net


if __name__ == '__main__':
    import time
    from thop import profile
    cfg = {
        'fpn': 'yolox2_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        'width': 1.0,
        'depth': 1.0,
        'ratio': 1.0
    }
    fpn_dims = [256, 512, 512]
    out_dim=256
    model = build_fpn(cfg, fpn_dims, out_dim)
    pyramid_feats = [torch.randn(1, fpn_dims[0], 80, 80), torch.randn(1, fpn_dims[1], 40, 40), torch.randn(1, fpn_dims[2], 20, 20)]
    t0 = time.time()
    outputs = model(pyramid_feats)
    t1 = time.time()
    print('Time: ', t1 - t0)
    for out in outputs:
        print(out.shape)

    print('==============================')
    flops, params = profile(model, inputs=(pyramid_feats, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))