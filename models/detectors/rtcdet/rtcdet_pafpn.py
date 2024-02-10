import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .rtcdet_basic import BasicConv, RTCBlock
except:
    from  rtcdet_basic import BasicConv, RTCBlock


# PaFPN-ELAN
class RTCPaFPN(nn.Module):
    def __init__(self, 
                 in_dims   = [256, 512, 1024],
                 out_dim   = 256,
                 width     = 1.0,
                 depth     = 1.0,
                 act_type  = 'silu',
                 norm_type = 'BN',
                 depthwise = False):
        super(RTCPaFPN, self).__init__()
        print('==============================')
        print('FPN: {}'.format("RTC PaFPN"))
        # ---------------- Basic parameters ----------------
        self.in_dims = in_dims
        self.width = width
        self.depth = depth
        c3, c4, c5 = in_dims

        # ---------------- Top-dwon FPN----------------
        ## P5 -> P4
        self.reduce_layer_1   = BasicConv(c5, round(512*width),
                                          kernel_size=1, padding=0, stride=1,
                                          act_type=act_type, norm_type=norm_type)
        self.top_down_layer_1 = RTCBlock(in_dim      = round(512*width) + c4,
                                         out_dim     = round(512*width),
                                         num_blocks  = round(3*depth),
                                         shortcut    = False,
                                         act_type    = act_type,
                                         norm_type   = norm_type,
                                         depthwise   = depthwise,
                                         )

        ## P4 -> P3
        self.reduce_layer_2   = BasicConv(round(512*width), round(256*width),
                                          kernel_size=1, padding=0, stride=1,
                                          act_type=act_type, norm_type=norm_type)
        self.top_down_layer_2 = RTCBlock(in_dim      = round(256*width) + c3,
                                         out_dim     = round(256*width),
                                         num_blocks  = round(3*depth),
                                         shortcut    = False,
                                         act_type    = act_type,
                                         norm_type   = norm_type,
                                         depthwise   = depthwise,
                                         )

        # ---------------- Bottom-up PAN ----------------
        ## P3 -> P4
        self.dowmsample_layer_1 = BasicConv(round(256*width), round(256*width),
                                            kernel_size=3, padding=1, stride=2,
                                            act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.bottom_up_layer_1  = RTCBlock(in_dim      = round(256*width) + round(256*width),
                                           out_dim     = round(512*width),
                                           num_blocks  = round(3*depth),
                                           shortcut    = False,
                                           act_type    = act_type,
                                           norm_type   = norm_type,
                                           depthwise   = depthwise,
                                           )

        ## P4 -> P5
        self.dowmsample_layer_2 = BasicConv(round(512*width), round(512*width),
                                            kernel_size=3, padding=1, stride=2,
                                            act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.bottom_up_layer_2  = RTCBlock(in_dim      = round(512*width) + round(512*width),
                                           out_dim     = round(1024*width),
                                           num_blocks  = round(3*depth),
                                           shortcut    = False,
                                           act_type    = act_type,
                                           norm_type   = norm_type,
                                           depthwise   = depthwise,
                                           )

        # ---------------- Output projection ----------------
        ## Output projs
        self.out_layers = nn.ModuleList([
            BasicConv(in_dim, out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
            for in_dim in [round(256*width), round(512*width), round(1024*width)]
            ])
        self.out_dims = [out_dim] * 3

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
        c14 = self.dowmsample_layer_1(c13)
        c15 = torch.cat([c14, c10], dim=1)
        c16 = self.bottom_up_layer_1(c15)
        ## P4 -> P5
        c17 = self.dowmsample_layer_2(c16)
        c18 = torch.cat([c17, c6], dim=1)
        c19 = self.bottom_up_layer_2(c18)

        out_feats = [c13, c16, c19] # [P3, P4, P5]
        
        # output proj layers
        out_feats_proj = []
        for feat, layer in zip(out_feats, self.out_layers):
            out_feats_proj.append(layer(feat))

        return out_feats_proj


def build_fpn(cfg, in_dims, out_dim):
    # build neck
    if cfg['fpn'] == 'rtc_pafpn':
        fpn_net = RTCPaFPN(in_dims   = in_dims,
                           out_dim   = out_dim,
                           width     = cfg['width'],
                           depth     = cfg['depth'],
                           act_type  = cfg['fpn_act'],
                           norm_type = cfg['fpn_norm'],
                           depthwise = cfg['fpn_depthwise']
                           )
    else:
        raise NotImplementedError("Unknown fpn: {}".format(cfg['fpn']))
    return fpn_net


if __name__ == '__main__':
    import time
    from thop import profile
    cfg = {
        'fpn': 'rtc_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        'width': 1.0,
        'depth': 1.0,
        'ratio': 1.0,
    }
    model = build_fpn(cfg, in_dims=[256, 512, 1024], out_dim=256)
    pyramid_feats = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
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