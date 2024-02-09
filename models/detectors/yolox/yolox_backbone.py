import torch
import torch.nn as nn

try:
    from .yolox_basic import Conv, CSPBlock
    from .yolox_neck import SPPF
except:
    from yolox_basic import Conv, CSPBlock
    from yolox_neck import SPPF


# CSPDarkNet
class CSPDarkNet(nn.Module):
    def __init__(self, depth=1.0, width=1.0, act_type='silu', norm_type='BN', depthwise=False):
        super(CSPDarkNet, self).__init__()
        self.feat_dims = [round(64 * width), round(128 * width), round(256 * width), round(512 * width), round(1024 * width)]
        # P1/2
        self.layer_1 = Conv(3, self.feat_dims[0], k=6, p=2, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        # P2/4
        self.layer_2 = nn.Sequential(
            Conv(self.feat_dims[0], self.feat_dims[1], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            CSPBlock(in_dim       = self.feat_dims[1],
                     out_dim      = self.feat_dims[1],
                     expand_ratio = 0.5,
                     nblocks      = round(3*depth),
                     shortcut     = True,
                     act_type     = act_type,
                     norm_type    = norm_type,
                     depthwise    = depthwise)
        )
        # P3/8
        self.layer_3 = nn.Sequential(
            Conv(self.feat_dims[1], self.feat_dims[2], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            CSPBlock(in_dim       = self.feat_dims[2],
                     out_dim      = self.feat_dims[2],
                     expand_ratio = 0.5,
                     nblocks      = round(9*depth),
                     shortcut     = True,
                     act_type     = act_type,
                     norm_type    = norm_type,
                     depthwise    = depthwise)
        )
        # P4/16
        self.layer_4 = nn.Sequential(
            Conv(self.feat_dims[2], self.feat_dims[3], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            CSPBlock(in_dim       = self.feat_dims[3],
                     out_dim      = self.feat_dims[3],
                     expand_ratio = 0.5,
                     nblocks      = round(9*depth),
                     shortcut     = True,
                     act_type     = act_type,
                     norm_type    = norm_type,
                     depthwise    = depthwise)
        )
        # P5/32
        self.layer_5 = nn.Sequential(
            Conv(self.feat_dims[3], self.feat_dims[4], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            SPPF(self.feat_dims[4], self.feat_dims[4], expand_ratio=0.5),
            CSPBlock(in_dim       = self.feat_dims[4],
                     out_dim      = self.feat_dims[4],
                     expand_ratio = 0.5,
                     nblocks      = round(3*depth),
                     shortcut     = True,
                     act_type     = act_type,
                     norm_type    = norm_type,
                     depthwise    = depthwise)
        )


    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        outputs = [c3, c4, c5]

        return outputs


# ---------------------------- Functions ----------------------------
## build CSPDarkNet
def build_backbone(cfg): 
    # Build backbone
    backbone = CSPDarkNet(cfg['depth'], cfg['width'], cfg['bk_act'], cfg['bk_norm'], cfg['bk_dpw'])
    feat_dims = backbone.feat_dims[-3:]

    return backbone, feat_dims


if __name__ == '__main__':
    import time
    from thop import profile
    cfg = {
        'bk_act': 'lrelu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'p6_feat': False,
        'p7_feat': False,
        'width': 1.0,
        'depth': 1.0,
    }
    model, feats = build_backbone(cfg)
    x = torch.randn(1, 3, 640, 640)
    t0 = time.time()
    outputs = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    for out in outputs:
        print(out.shape)

    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))