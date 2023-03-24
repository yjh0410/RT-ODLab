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
        self.feat_dims = [int(256*width), int(512*width), int(1024*width)]

        # P1
        self.layer_1 = Conv(3, int(64*width), k=6, p=2, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        
        # P2
        self.layer_2 = nn.Sequential(
            Conv(int(64*width), int(128*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            CSPBlock(int(128*width), int(128*width), expand_ratio=0.5, nblocks=int(3*depth),
                     shortcut=True, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P3
        self.layer_3 = nn.Sequential(
            Conv(int(128*width), int(256*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            CSPBlock(int(256*width), int(256*width), expand_ratio=0.5, nblocks=int(9*depth),
                     shortcut=True, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P4
        self.layer_4 = nn.Sequential(
            Conv(int(256*width), int(512*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            CSPBlock(int(512*width), int(512*width), expand_ratio=0.5, nblocks=int(9*depth),
                     shortcut=True, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P5
        self.layer_5 = nn.Sequential(
            Conv(int(512*width), int(1024*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            SPPF(int(1024*width), int(1024*width), expand_ratio=0.5, act_type=act_type, norm_type=norm_type),
            CSPBlock(int(1024*width), int(1024*width), expand_ratio=0.5, nblocks=int(3*depth),
                     shortcut=True, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
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
def build_backbone(cfg): 
    """Constructs a darknet-53 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    backbone = CSPDarkNet(cfg['depth'], cfg['width'], cfg['bk_act'], cfg['bk_norm'], cfg['bk_dpw'])
    feat_dims = backbone.feat_dims

    return backbone, feat_dims


if __name__ == '__main__':
    import time
    from thop import profile
    cfg = {
        'pretrained': False,
        'bk_act': 'lrelu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'p6_feat': False,
        'p7_feat': False,
        'width': 1.0,
        'depth': 1.0,
    }
    model, feats = build_backbone(cfg)
    x = torch.randn(1, 3, 256, 256)
    t0 = time.time()
    outputs = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    for out in outputs:
        print(out.shape)

    x = torch.randn(1, 3, 256, 256)
    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))