import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .yolox2_basic import Conv, YoloStageBlock
except:
    from yolox2_basic import Conv, YoloStageBlock


# PaFPN-ELAN
class Yolox2PaFPN(nn.Module):
    def __init__(self, 
                 in_dims   = [256, 512, 1024],
                 out_dim   = None,
                 width     = 1.0,
                 depth     = 1.0,
                 act_type  = 'silu',
                 norm_type = 'BN',
                 depthwise = False):
        super(Yolox2PaFPN, self).__init__()
        print('==============================')
        print('FPN: {}'.format("Yolox2 PaFPN"))
        # ---------------- Basic parameters ----------------
        self.in_dims = in_dims
        self.width = width
        self.depth = depth
        c3, c4, c5 = in_dims

        # ---------------- Top dwon ----------------
        ## P5 -> P4
        self.reduce_layer_1 = Conv(c5, round(512*width), k=1, act_type=act_type, norm_type=norm_type)
        self.top_down_layer_1 = YoloStageBlock(in_dim       = round(512*width) + c4,
                                               out_dim      = round(512*width),
                                               num_blocks   = round(3*depth),
                                               shortcut     = False,
                                               act_type     = act_type,
                                               norm_type    = norm_type,
                                               depthwise    = depthwise,
                                               )
        ## P4 -> P3
        self.reduce_layer_2 = Conv(round(512*width), round(256*width), k=1, act_type=act_type, norm_type=norm_type)
        self.top_down_layer_2 = YoloStageBlock(in_dim       = round(256*width) + c3,
                                               out_dim      = round(256*width),
                                               num_blocks   = round(3*depth),
                                               shortcut     = False,
                                               act_type     = act_type,
                                               norm_type    = norm_type,
                                               depthwise    = depthwise,
                                               )
        # ---------------- Bottom up ----------------
        ## P3 -> P4
        self.downsample_layer_1 = Conv(round(256*width), round(256*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.bottom_up_layer_1 = YoloStageBlock(in_dim       = round(256*width) + round(256*width),
                                                out_dim      = round(512*width),
                                                num_blocks   = round(3*depth),
                                                shortcut     = False,
                                                act_type     = act_type,
                                                norm_type    = norm_type,
                                                depthwise    = depthwise,
                                                )
        ## P4 -> P5
        self.downsample_layer_2 = Conv(round(512*width), round(512*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.bottom_up_layer_2 = YoloStageBlock(in_dim       = round(512 * width) + round(512 * width),
                                                out_dim      = round(1024 * width),
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
                     for in_dim in [round(256*width), round(512*width), round(1024*width)]
                     ])
            self.out_dim = [out_dim] * 3
        else:
            self.out_layers = None
            self.out_dim = [round(256*width), round(512*width), round(1024*width)]


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
        c6 = self.reduce_layer_1(c5)
        c7 = F.interpolate(c6, scale_factor=2.0)
        c8 = torch.cat([c7, c4], dim=1)
        c9 = self.top_down_layer_1(c8)
        ## P4 -> P3
        c10 = self.reduce_layer_2(c9)
        c11 = F.interpolate(c10, scale_factor=2.0)
        c12 = torch.cat([c11, c3], dim=1)
        c13 = self.top_down_layer_2(c12)

        # Bottom up
        ## p3 -> P4
        c14 = self.downsample_layer_1(c13)
        c15 = torch.cat([c14, c10], dim=1)
        c16 = self.bottom_up_layer_1(c15)
        ## P4 -> P5
        c17 = self.downsample_layer_2(c16)
        c18 = torch.cat([c17, c6], dim=1)
        c19 = self.bottom_up_layer_2(c18)

        out_feats = [c13, c16, c19] # [P3, P4, P5]
        
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
    }
    fpn_dims = [256, 512, 1024]
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