import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .basic_modules.backbone import build_backbone
    from .basic_modules.fpn      import build_fpn
except:
    from  basic_modules.backbone import build_backbone
    from  basic_modules.fpn      import build_fpn


# ----------------- Image Encoder -----------------
def build_image_encoder(cfg, trainable=False):
    return ImageEncoder(cfg, trainable)

class ImageEncoder(nn.Module):
    def __init__(self, cfg, trainable=False):
        super().__init__()
        # ---------------- Basic settings ----------------
        ## Basic parameters
        self.cfg = cfg
        ## Network parameters
        self.strides = cfg['out_stride']
        self.hidden_dim = cfg['hidden_dim']
        self.num_levels = len(self.strides)
        
        # ---------------- Network settings ----------------
        ## Backbone Network
        self.backbone, fpn_feat_dims = build_backbone(cfg, pretrained=cfg['pretrained']&trainable)

        ## Feature Pyramid Network
        self.fpn = build_fpn(cfg, fpn_feat_dims, self.hidden_dim)
        
    def forward(self, x):
        pyramid_feats = self.backbone(x)
        pyramid_feats = self.fpn(pyramid_feats)

        return pyramid_feats


if __name__ == '__main__':
    import time
    from thop import profile
    cfg = {
        'width': 1.0,
        'depth': 1.0,
        'out_stride': [8, 16, 32],
        # Image Encoder - Backbone
        'backbone': 'resnet18',
        'backbone_norm': 'BN',
        'res5_dilation': False,
        'pretrained': True,
        'pretrained_weight': 'imagenet1k_v1',
        # Image Encoder - FPN
        'fpn': 'hybrid_encoder',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        'hidden_dim': 256,
        'en_num_heads': 8,
        'en_num_layers': 1,
        'en_mlp_ratio': 4.0,
        'en_dropout': 0.1,
        'pe_temperature': 10000.,
        'en_act': 'gelu',
    }
    x = torch.rand(2, 3, 640, 640)
    model = build_image_encoder(cfg, True)

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
