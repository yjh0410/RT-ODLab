import torch
import torch.nn as nn

from .cnn_backbone import build_backbone
from .cnn_neck import build_neck
from .cnn_pafpn import build_fpn


# ------------------------ Image Encoder ------------------------
class ImageEncoder(nn.Module):
    def __init__(self, cfg, trainable=False) -> None:
        super().__init__()
        ## Backbone
        self.backbone, feats_dim = build_backbone(cfg, cfg['pretrained']*trainable)

        ## Encoder
        self.encoder = build_neck(cfg, feats_dim[-1], feats_dim[-1])

        ## CSFM
        self.csfm = build_fpn(cfg=cfg, in_dims=feats_dim, out_dim=round(cfg['d_model']*cfg['width']), input_proj=True)


    def forward(self, x):
        # Backbone
        pyramid_feats = self.backbone(x)

        # Encoder
        pyramid_feats[-1] = self.encoder(pyramid_feats[-1])

        # CSFM
        pyramid_feats = self.csfm(pyramid_feats)

        return pyramid_feats


# build img-encoder
def build_img_encoder(cfg, trainable):
    return ImageEncoder(cfg, trainable)

