import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .rtcdetv2_basic import Conv, ResXStage
except:
    from rtcdetv2_basic import Conv, ResXStage


# PaFPN-CSP
class RTCDetv2PaFPN(nn.Module):
    def __init__(self, 
                 in_dims=[256, 512, 1024],
                 out_dim=256,
                 width=1.0,
                 depth=1.0,
                 act_type='silu',
                 norm_type='BN',
                 depthwise=False):
        super(RTCDetv2PaFPN, self).__init__()
        # ------------- Basic parameters -------------
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.expand_ratios = [0.25, 0.25, 0.25, 0.25]
        self.ffn_ratios = [4.0, 4.0, 4.0, 4.0]
        self.num_branches = [4, 4, 4, 4]
        self.num_blocks = [round(2 * depth), round(2 * depth), round(2 * depth), round(2 * depth)]
        c3, c4, c5 = in_dims

        # top down
        ## P5 -> P4
        self.reduce_layer_1 = Conv(c5, round(384*width), k=1, act_type=act_type, norm_type=norm_type)
        self.top_down_layer_1 = ResXStage(in_dim       = c4 + round(384*width),
                                          out_dim      = int(384*width),
                                          expand_ratio = self.expand_ratios[0],
                                          ffn_ratio    = self.ffn_ratios[0],
                                          num_branches = self.num_branches[0],
                                          num_blocks   = self.num_blocks[0],
                                          shortcut     = False,
                                          act_type     = act_type,
                                          norm_type    = norm_type,
                                          depthwise    = depthwise
                                          )

        ## P4 -> P3
        self.reduce_layer_2 = Conv(c4, round(192*width), k=1, norm_type=norm_type, act_type=act_type)
        self.top_down_layer_2 = ResXStage(in_dim       = c3 + round(192*width), 
                                          out_dim      = round(192*width),
                                          expand_ratio = self.expand_ratios[1],
                                          ffn_ratio    = self.ffn_ratios[1],
                                          num_branches = self.num_branches[1],
                                          num_blocks   = self.num_blocks[1],
                                          shortcut     = False,
                                          act_type     = act_type,
                                          norm_type    = norm_type,
                                          depthwise    = depthwise
                                          )

        # bottom up
        ## P3 -> P4
        self.downsample_layer_1 = Conv(round(192*width), round(192*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.bottom_up_layer_1 = ResXStage(in_dim       = round(192*width) + round(192*width),
                                           out_dim      = round(384*width),
                                           expand_ratio = self.expand_ratios[2],
                                           ffn_ratio    = self.ffn_ratios[2],
                                           num_branches = self.num_branches[2],
                                           num_blocks   = self.num_blocks[2],
                                           shortcut     = False,
                                           act_type     = act_type,
                                           norm_type    = norm_type,
                                           depthwise    = depthwise
                                           )

        ## P4 -> P5
        self.downsample_layer_2 = Conv(round(384*width), round(384*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.bottom_up_layer_2 = ResXStage(in_dim       = round(384*width) + round(384*width),
                                           out_dim      = round(768*width),
                                           expand_ratio = self.expand_ratios[3],
                                           ffn_ratio    = self.ffn_ratios[3],
                                           num_branches = self.num_branches[3],
                                           num_blocks   = self.num_blocks[3],
                                           shortcut     = False,
                                           act_type     = act_type,
                                           norm_type    = norm_type,
                                           depthwise    = depthwise
                                           )

        # output proj layers
        if out_dim is not None:
            # output proj layers
            self.out_layers = nn.ModuleList([
                Conv(in_dim, out_dim, k=1,
                        norm_type=norm_type, act_type=act_type)
                        for in_dim in [round(192 * width), round(384 * width), round(768 * width)]
                        ])
            self.out_dim = [out_dim] * 3

        else:
            self.out_layers = None
            self.out_dim = [round(192 * width), round(384 * width), round(768 * width)]


    def forward(self, features):
        c3, c4, c5 = features

        c6 = self.reduce_layer_1(c5)
        c7 = F.interpolate(c6, scale_factor=2.0)   # s32->s16
        c8 = torch.cat([c7, c4], dim=1)
        c9 = self.top_down_layer_1(c8)
        # P3/8
        c10 = self.reduce_layer_2(c9)
        c11 = F.interpolate(c10, scale_factor=2.0)   # s16->s8
        c12 = torch.cat([c11, c3], dim=1)
        c13 = self.top_down_layer_2(c12)  # to det
        # p4/16
        c14 = self.downsample_layer_1(c13)
        c15 = torch.cat([c14, c10], dim=1)
        c16 = self.bottom_up_layer_1(c15)  # to det
        # p5/32
        c17 = self.downsample_layer_2(c16)
        c18 = torch.cat([c17, c6], dim=1)
        c19 = self.bottom_up_layer_2(c18)  # to det

        out_feats = [c13, c16, c19] # [P3, P4, P5]

        # output proj layers
        if self.out_layers is not None:
            # output proj layers
            out_feats_proj = []
            for feat, layer in zip(out_feats, self.out_layers):
                out_feats_proj.append(layer(feat))
            return out_feats_proj

        return out_feats


def build_fpn(cfg, in_dims, out_dim=None):
    model = cfg['fpn']
    # build neck
    if model == 'rtcdetv2_pafpn':
        fpn_net = RTCDetv2PaFPN(in_dims   = in_dims,
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
        'width': 1.0,
        'depth': 1.0,
        'fpn': 'rtcdetv2_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
    }
    fpn_dims = [192, 384, 768]
    out_dim = 192
    # Head-1
    model = build_fpn(cfg, fpn_dims, out_dim)
    fpn_feats = [torch.randn(1, fpn_dims[0], 80, 80), torch.randn(1, fpn_dims[1], 40, 40), torch.randn(1, fpn_dims[2], 20, 20)]
    t0 = time.time()
    outputs = model(fpn_feats)
    t1 = time.time()
    print('Time: ', t1 - t0)
    # for out in outputs:
    #     print(out.shape)

    print('==============================')
    flops, params = profile(model, inputs=(fpn_feats, ), verbose=False)
    print('==============================')
    print('FPN: GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('FPN: Params : {:.2f} M'.format(params / 1e6))