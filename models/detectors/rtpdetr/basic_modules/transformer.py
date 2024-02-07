import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

try:
    from .basic import FFN, GlobalCrossAttention
    from .basic import trunc_normal_
except:
    from  basic import FFN, GlobalCrossAttention
    from  basic import trunc_normal_


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
                 ffn_dim         :int = 1024,
                 dropout         :float = 0.1,
                 act_type        :str   = "relu",
                 ):
        super().__init__()
        # ----------- Basic parameters -----------
        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.act_type = act_type
        # ----------- Basic parameters -----------
        # Multi-head Self-Attn
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # Feedforwaed Network
        self.ffn = FFN(d_model, ffn_dim, dropout, act_type)

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
                 ffn_dim        :int = 1024,
                 pe_temperature : float = 10000.,
                 dropout        :float = 0.1,
                 act_type       :str   = "relu",
                 ):
        super().__init__()
        # ----------- Basic parameters -----------
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.act_type = act_type
        self.pe_temperature = pe_temperature
        self.pos_embed = None
        # ----------- Basic parameters -----------
        self.encoder_layers = get_clones(
            TransformerEncoderLayer(d_model, num_heads, ffn_dim, dropout, act_type), num_layers)

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

## PlainDETR's Decoder layer
class GlobalDecoderLayer(nn.Module):
    def __init__(self,
                 d_model    :int   = 256,
                 num_heads  :int   = 8,
                 ffn_dim    :int = 1024,
                 dropout    :float = 0.1,
                 act_type   :str   = "relu",
                 pre_norm   :bool  = False,
                 rpe_hidden_dim :int = 512,
                 feature_stride :int = 16,
                 ) -> None:
        super().__init__()
        # ------------ Basic parameters ------------
        self.d_model = d_model
        self.num_heads = num_heads
        self.rpe_hidden_dim = rpe_hidden_dim
        self.ffn_dim = ffn_dim
        self.act_type = act_type
        self.pre_norm = pre_norm

        # ------------ Network parameters ------------
        ## Multi-head Self-Attn
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        ## Box-reparam Global Cross-Attn
        self.cross_attn = GlobalCrossAttention(d_model, num_heads, rpe_hidden_dim=rpe_hidden_dim, feature_stride=feature_stride)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        ## FFN
        self.ffn = FFN(d_model, ffn_dim, dropout, act_type, pre_norm)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_pre_norm(self,
                         tgt,
                         query_pos,
                         reference_points,
                         src,
                         src_pos_embed,
                         src_spatial_shapes,
                         src_padding_mask=None,
                         self_attn_mask=None,
                         ):
        # ----------- Multi-head self attention -----------
        tgt1 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt1, query_pos)
        tgt1 = self.self_attn(q.transpose(0, 1),        # [B, N, C] -> [N, B, C], batch_first = False
                              k.transpose(0, 1),        # [B, N, C] -> [N, B, C], batch_first = False
                              tgt1.transpose(0, 1),     # [B, N, C] -> [N, B, C], batch_first = False
                              attn_mask=self_attn_mask,
                              )[0].transpose(0, 1)      # [N, B, C] -> [B, N, C]
        tgt = tgt + self.dropout1(tgt1)

        # ----------- Global corss attention -----------
        tgt1 = self.norm2(tgt)
        tgt1 = self.cross_attn(self.with_pos_embed(tgt1, query_pos),
                               reference_points,
                               self.with_pos_embed(src, src_pos_embed),
                               src,
                               src_spatial_shapes,
                               src_padding_mask,
                               )
        tgt = tgt + self.dropout2(tgt1)

        # ----------- FeedForward Network -----------
        tgt = self.ffn(tgt)

        return tgt

    def forward_post_norm(self,
                          tgt,
                          query_pos,
                          reference_points,
                          src,
                          src_pos_embed,
                          src_spatial_shapes,
                          src_padding_mask=None,
                          self_attn_mask=None,
                          ):
        # ----------- Multi-head self attention -----------
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt1 = self.self_attn(q.transpose(0, 1),        # [B, N, C] -> [N, B, C], batch_first = False
                              k.transpose(0, 1),        # [B, N, C] -> [N, B, C], batch_first = False
                              tgt.transpose(0, 1),     # [B, N, C] -> [N, B, C], batch_first = False
                              attn_mask=self_attn_mask,
                              )[0].transpose(0, 1)      # [N, B, C] -> [B, N, C]
        tgt = tgt + self.dropout1(tgt1)
        tgt = self.norm1(tgt)

        # ----------- Global corss attention -----------
        tgt1 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               self.with_pos_embed(src, src_pos_embed),
                               src,
                               src_spatial_shapes,
                               src_padding_mask,
                               )
        tgt = tgt + self.dropout2(tgt1)
        tgt = self.norm2(tgt)

        # ----------- FeedForward Network -----------
        tgt = self.ffn(tgt)

        return tgt

    def forward(self,
                tgt,
                query_pos,
                reference_points,
                src,
                src_pos_embed,
                src_spatial_shapes,
                src_padding_mask=None,
                self_attn_mask=None,
                ):
        if self.pre_norm:
            return self.forward_pre_norm(tgt, query_pos, reference_points, src, src_pos_embed, src_spatial_shapes,
                                         src_padding_mask, self_attn_mask)
        else:
            return self.forward_post_norm(tgt, query_pos, reference_points, src, src_pos_embed, src_spatial_shapes,
                                          src_padding_mask, self_attn_mask)

## PlainDETR's Decoder
class GlobalDecoder(nn.Module):
    def __init__(self,
                 # Decoder layer params
                 d_model    :int   = 256,
                 num_heads  :int   = 8,
                 ffn_dim    :int = 1024,
                 dropout    :float = 0.1,
                 act_type   :str   = "relu",
                 pre_norm   :bool  = False,
                 rpe_hidden_dim :int = 512,
                 feature_stride :int = 16,
                 num_layers     :int = 6,
                 # Decoder params
                 return_intermediate :bool = False,
                 use_checkpoint      :bool = False,
                 ):
        super().__init__()
        # ------------ Basic parameters ------------
        self.d_model = d_model
        self.num_heads = num_heads
        self.rpe_hidden_dim = rpe_hidden_dim
        self.ffn_dim = ffn_dim
        self.act_type = act_type
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.use_checkpoint = use_checkpoint

        # ------------ Network parameters ------------
        decoder_layer = GlobalDecoderLayer(
            d_model, num_heads, ffn_dim, dropout, act_type, pre_norm, rpe_hidden_dim, feature_stride,)
        self.layers = get_clones(decoder_layer, num_layers)
        self.bbox_embed = None
        self.class_embed = None

        if pre_norm:
            self.final_layer_norm = nn.LayerNorm(d_model)
        else:
            self.final_layer_norm = None

    def _reset_parameters(self):            
        # stolen from Swin Transformer
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def inverse_sigmoid(self, x, eps=1e-5):
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)

        return torch.log(x1 / x2)

    def box_xyxy_to_cxcywh(self, x):
        x0, y0, x1, y1 = x.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
        
        return torch.stack(b, dim=-1)

    def delta2bbox(self, proposals,
                   deltas,
                   max_shape=None,
                   wh_ratio_clip=16 / 1000,
                   clip_border=True,
                   add_ctr_clamp=False,
                   ctr_clamp=32):

        dxy = deltas[..., :2]
        dwh = deltas[..., 2:]

        # Compute width/height of each roi
        pxy = proposals[..., :2]
        pwh = proposals[..., 2:]

        dxy_wh = pwh * dxy
        wh_ratio_clip = torch.as_tensor(wh_ratio_clip)
        max_ratio = torch.abs(torch.log(wh_ratio_clip)).item()
        
        if add_ctr_clamp:
            dxy_wh = torch.clamp(dxy_wh, max=ctr_clamp, min=-ctr_clamp)
            dwh = torch.clamp(dwh, max=max_ratio)
        else:
            dwh = dwh.clamp(min=-max_ratio, max=max_ratio)

        gxy = pxy + dxy_wh
        gwh = pwh * dwh.exp()
        x1y1 = gxy - (gwh * 0.5)
        x2y2 = gxy + (gwh * 0.5)
        bboxes = torch.cat([x1y1, x2y2], dim=-1)
        if clip_border and max_shape is not None:
            bboxes[..., 0::2].clamp_(min=0).clamp_(max=max_shape[1])
            bboxes[..., 1::2].clamp_(min=0).clamp_(max=max_shape[0])

        return bboxes

    def forward(self,
                tgt,
                reference_points,
                src,
                src_pos_embed,
                src_spatial_shapes,
                query_pos=None,
                src_padding_mask=None,
                self_attn_mask=None,
                max_shape=None,
                ):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[:, :, None]
            if self.use_checkpoint:
                output = checkpoint.checkpoint(
                    layer,
                    output,
                    query_pos,
                    reference_points_input,
                    src,
                    src_pos_embed,
                    src_spatial_shapes,
                    src_padding_mask,
                    self_attn_mask,
                )
            else:
                output = layer(
                    output,
                    query_pos,
                    reference_points_input,
                    src,
                    src_pos_embed,
                    src_spatial_shapes,
                    src_padding_mask,
                    self_attn_mask,
                )

            if self.final_layer_norm is not None:
                output_after_norm = self.final_layer_norm(output)
            else:
                output_after_norm = output

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output_after_norm)
                new_reference_points = self.box_xyxy_to_cxcywh(
                    self.delta2bbox(reference_points, tmp, max_shape)) 
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output_after_norm)
                intermediate_reference_points.append(new_reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output_after_norm, reference_points
