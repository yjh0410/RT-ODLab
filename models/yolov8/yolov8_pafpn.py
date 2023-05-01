import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .yolov8_basic import Conv, ELAN_CSP_Block
except:
    from yolov8_basic import Conv, ELAN_CSP_Block


# PaFPN-ELAN
class Yolov8PaFPN(nn.Module):
    def __init__(self, 
                 in_dims=[256, 512, 512],
                 width=1.0,
                 depth=1.0,
                 ratio=1.0,
                 act_type='silu',
                 norm_type='BN',
                 depthwise=False):
        super(Yolov8PaFPN, self).__init__()
        print('==============================')
        print('FPN: {}'.format("ELAN_PaFPN"))
        self.in_dims = in_dims
        self.width = width
        self.depth = depth
        self.out_dim = [int(256 * width), int(512 * width), int(512 * width * ratio)]
        c3, c4, c5 = in_dims

        # --------------------------- Top-dwon ---------------------------
        ## P5 -> P4
        self.head_elan_1 = ELAN_CSP_Block(in_dim=c5 + c4,
                                          out_dim=int(512*width),
                                          expand_ratio=0.5,
                                          nblocks=int(3*depth),
                                          shortcut=False,
                                          depthwise=depthwise,
                                          norm_type=norm_type,
                                          act_type=act_type
                                          )
        ## P4 -> P3
        self.head_elan_2 = ELAN_CSP_Block(in_dim=int(512*width) + c3,
                                          out_dim=int(256*width),
                                          expand_ratio=0.5,
                                          nblocks=int(3*depth),
                                          shortcut=False,
                                          depthwise=depthwise,
                                          norm_type=norm_type,
                                          act_type=act_type
                                          )
        # --------------------------- Bottom-up ---------------------------
        ## P3 -> P4
        self.mp1 = Conv(int(256*width), int(256*width), k=3, p=1, s=2,
                        act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.head_elan_3 = ELAN_CSP_Block(in_dim=int(256*width) + int(512*width),
                                          out_dim=int(512*width),
                                          expand_ratio=0.5,
                                          nblocks=int(3*depth),
                                          shortcut=False,
                                          depthwise=depthwise,
                                          norm_type=norm_type,
                                          act_type=act_type
                                          )
        ## P4 -> P5
        self.mp2 = Conv(int(512 * width), int(512 * width), k=3, p=1, s=2,
                        act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.head_elan_4 = ELAN_CSP_Block(in_dim=int(512 * width) + c5,
                                          out_dim=int(512 * width * ratio),
                                          expand_ratio=0.5,
                                          nblocks=int(3*depth),
                                          shortcut=False,
                                          depthwise=depthwise,
                                          norm_type=norm_type,
                                          act_type=act_type
                                          )


    def forward(self, features):
        c3, c4, c5 = features

        # Top down
        ## P5 -> P4
        c6 = F.interpolate(c5, scale_factor=2.0)
        c7 = torch.cat([c6, c4], dim=1)
        c8 = self.head_elan_1(c7)
        ## P4 -> P3
        c9 = F.interpolate(c8, scale_factor=2.0)
        c10 = torch.cat([c9, c3], dim=1)
        c11 = self.head_elan_2(c10)

        # Bottom up
        # p3 -> P4
        c12 = self.mp1(c11)
        c13 = torch.cat([c12, c8], dim=1)
        c14 = self.head_elan_3(c13)
        # P4 -> P5
        c15 = self.mp2(c14)
        c16 = torch.cat([c15, c5], dim=1)
        c17 = self.head_elan_4(c16)

        out_feats = [c11, c14, c17] # [P3, P4, P5]
        
        return out_feats


def build_fpn(cfg, in_dims):
    model = cfg['fpn']
    # build neck
    if model == 'yolov8_pafpn':
        fpn_net = Yolov8PaFPN(in_dims=in_dims,
                             width=cfg['width'],
                             depth=cfg['depth'],
                             ratio=cfg['ratio'],
                             act_type=cfg['fpn_act'],
                             norm_type=cfg['fpn_norm'],
                             depthwise=cfg['fpn_depthwise']
                             )
    return fpn_net


if __name__ == '__main__':
    import time
    from thop import profile
    cfg = {
        'fpn': 'Yolov8PaFPN',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        'width': 1.0,
        'depth': 1.0,
        'ratio': 1.0,
    }
    model = build_fpn(cfg, in_dims=[256, 512, 512])
    pyramid_feats = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 512, 20, 20)]
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