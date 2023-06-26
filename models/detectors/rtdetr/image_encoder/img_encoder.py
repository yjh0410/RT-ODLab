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
        self.csfm = build_fpn(cfg=cfg, in_dims=feats_dim, out_dim=round(cfg['d_model']*cfg['width']))


    def position_embedding(self, x, temperature=10000):
        hs, ws = x.shape[-2:]
        device = x.device
        num_pos_feats = x.shape[1] // 2       
        scale = 2 * 3.141592653589793

        # generate xy coord mat
        y_embed, x_embed = torch.meshgrid(
            [torch.arange(1, hs+1, dtype=torch.float32),
             torch.arange(1, ws+1, dtype=torch.float32)])
        y_embed = y_embed / (hs + 1e-6) * scale
        x_embed = x_embed / (ws + 1e-6) * scale
    
        # [H, W] -> [1, H, W]
        y_embed = y_embed[None, :, :].to(device)
        x_embed = x_embed[None, :, :].to(device)

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=device)
        dim_t_ = torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats
        dim_t = temperature ** (2 * dim_t_)

        pos_x = torch.div(x_embed[:, :, :, None], dim_t)
        pos_y = torch.div(y_embed[:, :, :, None], dim_t)
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        # [B, C, H, W]
        pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        return pos_embed
        

    def forward(self, x):
        # Backbone
        pyramid_feats = self.backbone(x)

        # Encoder
        pyramid_feats[-1] = self.encoder(pyramid_feats[-1])

        # CSFM
        pyramid_feats = self.csfm(pyramid_feats)

        # Prepare memory & memoery_pos for Decoder
        memory = torch.cat([feat.flatten(2) for feat in pyramid_feats], dim=-1)
        memory = memory.permute(0, 2, 1).contiguous()
        memory_pos = torch.cat([self.position_embedding(feat).flatten(2)
                                for feat in pyramid_feats], dim=-1)
        memory_pos = memory_pos.permute(0, 2, 1).contiguous()

        return memory, memory_pos


# build img-encoder
def build_img_encoder(cfg, trainable):
    return ImageEncoder(cfg, trainable)

