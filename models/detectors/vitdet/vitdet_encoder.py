import torch
import torch.nn as nn

try:
    from .basic_modules.basic    import BasicConv, UpSampleWrapper
    from .basic_modules.backbone import build_backbone
except:
    from  basic_modules.basic    import BasicConv, UpSampleWrapper
    from  basic_modules.backbone import build_backbone


# ----------------- Image Encoder -----------------
def build_image_encoder(cfg):
    return ImageEncoder(cfg)

class ImageEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # ---------------- Basic settings ----------------
        ## Basic parameters
        self.cfg = cfg
        ## Network parameters
        self.stride = 16
        self.fpn_dims = [cfg['hidden_dim']] * 3
        self.hidden_dim = cfg['hidden_dim']
        
        # ---------------- Network settings ----------------
        ## Backbone Network
        self.backbone, backbone_dim = build_backbone(cfg, cfg['pretrained'])

        ## Input projection
        self.input_proj = BasicConv(backbone_dim, cfg['hidden_dim'],
                                    kernel_size=1,
                                    act_type=None, norm_type='BN')

        ## Upsample layer
        self.upsample = UpSampleWrapper(cfg['hidden_dim'], 2.0)
        
        ## Downsample layer
        self.downsample = BasicConv(cfg['hidden_dim'], cfg['hidden_dim'],
                                    kernel_size=3, padding=1, stride=2,
                                    act_type=None, norm_type='BN')

        ## Output projection
        self.output_projs = nn.ModuleList([BasicConv(cfg['hidden_dim'], cfg['hidden_dim'],
                                                     kernel_size=3, padding=1,
                                                     act_type='silu', norm_type='BN')
                                                     ] * 3)


    def forward(self, x):
        # Backbone
        feat = self.backbone(x)

        # Input proj
        feat = self.input_proj(feat)

        # FPN
        feat_up = self.upsample(feat)
        feat_ds = self.downsample(feat)

        # Multi level features: [P3, P4, P5]
        pyramid_feats = [self.output_projs[0](feat_up),
                         self.output_projs[1](feat),
                         self.output_projs[2](feat_ds)]

        return pyramid_feats


if __name__ == '__main__':
    import time
    from thop import profile
    cfg = {
        'width': 1.0,
        'depth': 1.0,
        'out_stride': 16,
        'hidden_dim': 256,
        # Image Encoder - Backbone
        'backbone': 'resnet50',
        'backbone_norm': 'FrozeBN',
        'pretrained': True,
        'freeze_at': 0,
        'freeze_stem_only': False,
    }
    x = torch.rand(2, 3, 640, 640)
    model = build_image_encoder(cfg)
    model.train()

    t0 = time.time()
    outputs = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    print(outputs.shape)

    print('==============================')
    model.eval()
    x = torch.rand(1, 3, 640, 640)
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
