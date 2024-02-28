import torch
import torch.nn as nn

try:
    from .yolov8_basic import Conv, Yolov8StageBlock
except:
    from yolov8_basic import Conv, Yolov8StageBlock


# ---------------------------- Basic functions ----------------------------
## ELAN-CSPNet
class Yolov8Backbone(nn.Module):
    def __init__(self, width=1.0, depth=1.0, ratio=1.0, act_type='silu', norm_type='BN', depthwise=False):
        super(Yolov8Backbone, self).__init__()
        self.feat_dims = [round(64 * width), round(128 * width), round(256 * width), round(512 * width), round(512 * width * ratio)]
        # P1/2
        self.layer_1 = Conv(3, self.feat_dims[0], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type)
        # P2/4
        self.layer_2 = nn.Sequential(
            Conv(self.feat_dims[0], self.feat_dims[1], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            Yolov8StageBlock(in_dim     = self.feat_dims[1],
                             out_dim    = self.feat_dims[1],
                             num_blocks = round(3*depth),
                             shortcut   = True,
                             act_type   = act_type,
                             norm_type  = norm_type,
                             depthwise  = depthwise)
        )
        # P3/8
        self.layer_3 = nn.Sequential(
            Conv(self.feat_dims[1], self.feat_dims[2], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            Yolov8StageBlock(in_dim     = self.feat_dims[2],
                             out_dim    = self.feat_dims[2],
                             num_blocks = round(6*depth),
                             shortcut   = True,
                             act_type   = act_type,
                             norm_type  = norm_type,
                             depthwise  = depthwise)
        )
        # P4/16
        self.layer_4 = nn.Sequential(
            Conv(self.feat_dims[2], self.feat_dims[3], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            Yolov8StageBlock(in_dim     = self.feat_dims[3],
                             out_dim    = self.feat_dims[3],
                             num_blocks = round(6*depth),
                             shortcut   = True,
                             act_type   = act_type,
                             norm_type  = norm_type,
                             depthwise  = depthwise)
        )
        # P5/32
        self.layer_5 = nn.Sequential(
            Conv(self.feat_dims[3], self.feat_dims[4], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            Yolov8StageBlock(in_dim     = self.feat_dims[4],
                             out_dim    = self.feat_dims[4],
                             num_blocks = round(3*depth),
                             shortcut   = True,
                             act_type   = act_type,
                             norm_type  = norm_type,
                             depthwise  = depthwise)
        )

        self.init_weights()
        
    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()

    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)
        outputs = [c3, c4, c5]

        return outputs


# ---------------------------- Functions ----------------------------
## build Yolov8's Backbone
def build_backbone(cfg): 
    # model
    backbone = Yolov8Backbone(width=cfg['width'],
                              depth=cfg['depth'],
                              ratio=cfg['ratio'],
                              act_type=cfg['bk_act'],
                              norm_type=cfg['bk_norm'],
                              depthwise=cfg['bk_depthwise']
                              )
    feat_dims = backbone.feat_dims[-3:]
        
    return backbone, feat_dims


if __name__ == '__main__':
    import time
    from thop import profile
    cfg = {
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'width': 0.25,
        'depth': 0.34,
        'ratio': 2.0,
    }
    model, feats = build_backbone(cfg)
    x = torch.randn(1, 3, 640, 640)
    t0 = time.time()
    outputs = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    for out in outputs:
        print(out.shape)

    x = torch.randn(1, 3, 640, 640)
    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))