import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .basic import FFN
except:
    from  basic import FFN


def get_clones(module, N):
    if N <= 0:
        return None
    else:
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0., max=1.)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


# ----------------- Basic Transformer Ops -----------------
def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = (
            value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        )
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()

class MSDeformableAttention(nn.Module):
    def __init__(self,
                 embed_dim=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4):
        """
        Multi-Scale Deformable Attention Module
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        try:
            # use cuda op
            from deformable_detr_ops import ms_deformable_attn
            self.ms_deformable_attn_core = ms_deformable_attn
        except:
            # use torch func
            self.ms_deformable_attn_core = multi_scale_deformable_attn_pytorch

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Default initialization for Parameters of Module.
        """
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        # attention weight
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)

        # proj
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (Tensor): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, num_query = query.shape[:2]
        num_value = value.shape[1]
        assert sum([s[0] * s[1] for s in value_spatial_shapes]) == num_value

        # Value projection
        value = self.value_proj(value)
        # fill "0" for the padding part
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        # [bs, all_hw, 256] -> [bs, all_hw, num_head, head_dim]
        value = value.reshape([bs, num_value, self.num_heads, -1])

        # [bs, all_hw, num_head, nun_level, num_sample_point, num_offset]
        sampling_offsets = self.sampling_offsets(query).reshape(
            [bs, num_query, self.num_heads, self.num_levels, self.num_points, 2])
        # [bs, all_hw, num_head, nun_level*num_sample_point]
        attention_weights = self.attention_weights(query).reshape(
            [bs, num_query, self.num_heads, self.num_levels * self.num_points])
        # [bs, all_hw, num_head, nun_level, num_sample_point]
        attention_weights = attention_weights.softmax(-1).reshape(
            [bs, num_query, self.num_heads, self.num_levels, self.num_points])

        # [bs, num_query, num_heads, num_levels, num_points, 2]
        if reference_points.shape[-1] == 2:
            # reference_points   [bs, all_hw, num_sample_point, 2] -> [bs, all_hw, 1, num_sample_point, 1, 2]
            # sampling_offsets   [bs, all_hw, nun_head, num_level, num_sample_point, 2]
            # offset_normalizer  [4, 2] -> [1, 1, 1, num_sample_point, 1, 2]
            # references_points + sampling_offsets
            offset_normalizer = value_spatial_shapes.flip([1]).reshape(
                [1, 1, 1, self.num_levels, 1, 2])
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        # Multi-scale Deformable attention
        output = self.ms_deformable_attn_core(
            value, value_spatial_shapes, sampling_locations, attention_weights)
        
        # Output project
        output = self.output_proj(output)

        return output


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
class DeformableTransformerDecoderLayer(nn.Module):
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
        self.self_attn  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        ## CrossAttention
        self.cross_attn = MSDeformableAttention(d_model, num_heads, num_levels, num_points)
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
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)[0]
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
class DeformableTransformerDecoder(nn.Module):
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
            DeformableTransformerDecoderLayer(d_model, num_heads, num_levels, num_points, mlp_ratio, dropout, act_type), num_layers)
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

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            dec_out_logits.append(score_head[i](output))
            if i == 0:
                dec_out_bboxes.append(inter_ref_bbox)
            else:
                dec_out_bboxes.append(
                    F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)

