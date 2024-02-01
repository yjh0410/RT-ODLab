import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

try:
    from .basic import get_activation, MLP, FFN
except:
    from  basic import get_activation, MLP, FFN


def get_clones(module, N):
    if N <= 0:
        return None
    else:
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0., max=1.)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


# ----------------- Transformer modules -----------------
## Transformer Encoder layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model         :int   = 256,
                 num_heads       :int   = 8,
                 mlp_ratio       :float = 4.0,
                 dropout         :float = 0.1,
                 act_type        :str   = "relu",
                 ):
        super().__init__()
        # ----------- Basic parameters -----------
        self.d_model = d_model
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.act_type = act_type
        # ----------- Basic parameters -----------
        # Multi-head Self-Attn
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # Feedforwaed Network
        self.ffn = FFN(d_model, mlp_ratio, dropout, act_type)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos


    def forward(self, src, pos_embed):
        """
        Input:
            src:       [torch.Tensor] -> [B, N, C]
            pos_embed: [torch.Tensor] -> [B, N, C]
        Output:
            src:       [torch.Tensor] -> [B, N, C]
        """
        q = k = self.with_pos_embed(src, pos_embed)

        # -------------- MHSA --------------
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout(src2)
        src = self.norm(src)

        # -------------- FFN --------------
        src = self.ffn(src)
        
        return src

## Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self,
                 d_model        :int   = 256,
                 num_heads      :int   = 8,
                 num_layers     :int   = 1,
                 mlp_ratio      :float = 4.0,
                 pe_temperature : float = 10000.,
                 dropout        :float = 0.1,
                 act_type       :str   = "relu",
                 ):
        super().__init__()
        # ----------- Basic parameters -----------
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.act_type = act_type
        self.pe_temperature = pe_temperature
        self.pos_embed = None
        # ----------- Basic parameters -----------
        self.encoder_layers = get_clones(
            TransformerEncoderLayer(d_model, num_heads, mlp_ratio, dropout, act_type), num_layers)

    def build_2d_sincos_position_embedding(self, device, w, h, embed_dim=256, temperature=10000.):
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        
        # ----------- Check cahed pos_embed -----------
        if self.pos_embed is not None and \
            self.pos_embed.shape[2:] == [h, w]:
            return self.pos_embed
        
        # ----------- Generate grid coords -----------
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid([grid_w, grid_h])  # shape: [H, W]

        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None] # shape: [N, C]
        out_h = grid_h.flatten()[..., None] @ omega[None] # shape: [N, C]

        # shape: [1, N, C]
        pos_embed = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h),torch.cos(out_h)], dim=1)[None, :, :]
        pos_embed = pos_embed.to(device)
        self.pos_embed = pos_embed

        return pos_embed

    def forward(self, src):
        """
        Input:
            src:  [torch.Tensor] -> [B, C, H, W]
        Output:
            src:  [torch.Tensor] -> [B, C, H, W]
        """
        # -------- Transformer encoder --------
        channels, fmp_h, fmp_w = src.shape[1:]
        # [B, C, H, W] -> [B, N, C], N=HxW
        src_flatten = src.flatten(2).permute(0, 2, 1)
        memory = src_flatten

        # PosEmbed: [1, N, C]
        pos_embed = self.build_2d_sincos_position_embedding(
            src.device, fmp_w, fmp_h, channels, self.pe_temperature)
        
        # Transformer Encoder layer
        for encoder in self.encoder_layers:
            memory = encoder(memory, pos_embed=pos_embed)

        # Output: [B, N, C] -> [B, C, N] -> [B, C, H, W]
        src = memory.permute(0, 2, 1).reshape([-1, channels, fmp_h, fmp_w])

        return src

## Transformer Decoder layer
class PlainTransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model     :int   = 256,
                 num_heads   :int   = 8,
                 num_levels  :int   = 3,
                 num_points  :int   = 4,
                 mlp_ratio   :float = 4.0,
                 dropout     :float = 0.1,
                 act_type    :str   = "relu",
                 ):
        super().__init__()
        # ----------- Basic parameters -----------
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.act_type = act_type
        # ---------------- Network parameters ----------------
        ## Multi-head Self-Attn
        self.self_attn  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        ## CrossAttention
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        ## FFN
        self.ffn = FFN(d_model, mlp_ratio, dropout, act_type)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None):
        # ---------------- MSHA for Object Query -----------------
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if attn_mask is not None:
            attn_mask = torch.where(
                attn_mask.bool(),
                torch.zeros(attn_mask.shape, dtype=tgt.dtype, device=attn_mask.device),
                torch.full(attn_mask.shape, float("-inf"), dtype=tgt.dtype, device=attn_mask.device))
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ---------------- CMHA for Object Query and Image-feature -----------------
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos_embed),
                               reference_points,
                               memory,
                               memory_spatial_shapes,
                               memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ---------------- FeedForward Network -----------------
        tgt = self.ffn(tgt)

        return tgt

## Transformer Decoder
class PlainTransformerDecoder(nn.Module):
    def __init__(self,
                 d_model        :int   = 256,
                 num_heads      :int   = 8,
                 num_layers     :int   = 1,
                 num_levels     :int   = 3,
                 num_points     :int   = 4,
                 mlp_ratio      :float = 4.0,
                 dropout        :float = 0.1,
                 act_type       :str   = "relu",
                 return_intermediate :bool = False,
                 ):
        super().__init__()
        # ----------- Basic parameters -----------
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.act_type = act_type
        self.pos_embed = None
        # ----------- Network parameters -----------
        self.decoder_layers = get_clones(
            TransformerDecoderLayer(d_model, num_heads, num_levels, num_points, mlp_ratio, dropout, act_type), num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)
        for i, layer in enumerate(self.decoder_layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(output, ref_points_input, memory,
                           memory_spatial_shapes, attn_mask,
                           memory_mask, query_pos_embed)

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                ref_points_detach))

            dec_out_logits.append(score_head[i](output))
            if i == 0:
                dec_out_bboxes.append(inter_ref_bbox)
            else:
                dec_out_bboxes.append(
                    F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                        ref_points)))

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach()

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)

