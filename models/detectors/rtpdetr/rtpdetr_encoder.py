import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .basic_modules.basic    import BasicConv, UpSampleWrapper
    from .basic_modules.backbone import build_backbone
    from .basic_modules.transformer import TransformerEncoder
except:
    from  basic_modules.basic    import BasicConv, UpSampleWrapper
    from  basic_modules.backbone import build_backbone
    from  basic_modules.transformer import TransformerEncoder


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

        ## Input projection
        self.input_proj = BasicConv(fpn_feat_dims[-1], cfg['hidden_dim'], kernel_size=1, act_type=None, norm_type='BN')

        # ---------------- Transformer Encoder ----------------
        self.transformer_encoder = TransformerEncoder(d_model        = cfg['hidden_dim'],
                                                      num_heads      = cfg['en_num_heads'],
                                                      num_layers     = cfg['en_num_layers'],
                                                      mlp_ratio      = cfg['en_mlp_ratio'],
                                                      dropout        = cfg['en_dropout'],
                                                      act_type       = cfg['en_act']
                                                      )

        ## Upsample layer
        self.upsample = UpSampleWrapper(cfg['hidden_dim'], self.upsample_factor)
        
        ## Output projection
        self.output_proj = BasicConv(cfg['hidden_dim'], cfg['hidden_dim'], kernel_size=1, act_type=None, norm_type='BN')


    def forward(self, x):
        pyramid_feats = self.backbone(x)
        feat = self.input_proj(pyramid_feats[-1])
        feat = self.transformer_encoder(feat)
        feat = self.upsample(feat)
        feat = self.output_proj(feat)

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
        'backbone_norm': 'FrozeBN',
        'pretrained': True,
        'freeze_at': 0,
        'freeze_stem_only': False,
        'hidden_dim': 256,
        'en_num_heads': 8,
        'en_num_layers': 1,
        'en_mlp_ratio': 4.0,
        'en_dropout': 0.0,
        'en_act': 'gelu',
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
