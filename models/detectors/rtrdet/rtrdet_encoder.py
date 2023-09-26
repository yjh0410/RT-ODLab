import torch
import torch.nn as nn

from .rtrdet_basic import get_clones, TREncoderLayer


# Transformer Encoder Module
class TransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # -------------------- Basic Parameters ---------------------
        self.d_model = round(cfg['d_model']*cfg['width'])
        self.num_encoder = cfg['num_encoder']

        # -------------------- Network Parameters ---------------------
        encoder_layer = TREncoderLayer(d_model   = self.d_model,
                                       num_heads = cfg['encoder_num_head'],
                                       mlp_ratio = cfg['encoder_mlp_ratio'],
                                       dropout   = cfg['encoder_dropout'],
                                       act_type  = cfg['encoder_act']
                                       )
        self.encoder_layers = get_clones(encoder_layer, self.num_encoder)


    def forward(self, feat, pos_embed, adapt_pos2d):
        # reshape: [B, C, H, W] -> [B, N, C], N = HW
        feat = feat.flatten(2).permute(0, 2, 1).contiguous()
        pos_embed = adapt_pos2d(pos_embed.flatten(2).permute(0, 2, 1).contiguous())

        # Transformer encoder
        for encoder in self.encoder_layers:
            feat = encoder(feat, pos_embed)

        return feat


# build detection head
def build_encoder(cfg):
    transformer_encoder = TransformerEncoder(cfg) 

    return transformer_encoder
