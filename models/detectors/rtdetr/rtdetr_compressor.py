import torch
import torch.nn as nn

from .rtdetr_basic import TRDecoderLayer


# Transformer Decoder Module
class MemoryCompressor(nn.Module):
    def __init__(self, cfg, in_dim):
        super().__init__()
        # -------------------- Basic Parameters ---------------------
        self.d_model = in_dim
        self.ffn_dim = round(cfg['com_dim_feedforward']*cfg['width'])
        self.compressed_vector = nn.Embedding(cfg['dim_compressed'], in_dim)
        # -------------------- Network Parameters ---------------------
        self.compress_layer = TRDecoderLayer(
            d_model=in_dim,
            dim_feedforward=self.ffn_dim,
            num_heads=cfg['com_num_heads'],
            dropout=cfg['com_dropout'],
            act_type=cfg['com_act']
        )


    def forward(self, memory, memory_pos):
        bs = memory.size(0)
        output = self.compressed_vector.weight[None].repeat(bs, 1, 1)
        output = self.compress_layer(output, None, memory, memory_pos)

        return output


def build_compressor(cfg, in_dim):
    return MemoryCompressor(cfg, in_dim)
