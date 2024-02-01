import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.stride = cfg['out_stride']
        self.upsample_factor = 32 // self.stride
        self.hidden_dim = cfg['hidden_dim']
        
        # ---------------- Network settings ----------------
        ## Backbone Network
        self.backbone, fpn_feat_dims = build_backbone(cfg, pretrained=cfg['pretrained']&self.training)

        ## Upsample layer
        self.upsample = UpSampleWrapper(fpn_feat_dims[-1], self.upsample_factor)
        
        ## Input projection
        self.input_proj = BasicConv(self.upsample.out_dim, self.hidden_dim, kernel_size=1, act_type=None, norm_type='BN')


    def forward(self, x):
        pyramid_feats = self.backbone(x)
        feat = self.upsample(pyramid_feats[-1])
        feat = self.input_proj(feat)

        return feat


if __name__ == '__main__':
    import time
    from thop import profile
    cfg = {
        'width': 1.0,
        'depth': 1.0,
        'out_stride': 16,
        # Image Encoder - Backbone
        'backbone': 'resnet50',
        'backbone_norm': 'BN',
        'res5_dilation': False,
        'pretrained': True,
        'pretrained_weight': 'imagenet1k_v1',        
        'hidden_dim': 256,
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
