import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_clones(module, N):
    if N <= 0:
        return None
    else:
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# ----------------- MLP modules -----------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([in_dim] + h, h + [out_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class FFN(nn.Module):
    def __init__(self, d_model=256, mlp_ratio=4.0, dropout=0., act_type='relu'):
        super().__init__()
        self.fpn_dim = round(d_model * mlp_ratio)
        self.linear1 = nn.Linear(d_model, self.fpn_dim)
        self.activation = get_activation(act_type)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(self.fpn_dim, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm(src)
        
        return src
    

# ----------------- CNN modules -----------------
def get_conv2d(c1, c2, k, p, s, g, bias=False):
    conv = nn.Conv2d(c1, c2, k, stride=s, padding=p, groups=g, bias=bias)

    return conv

def get_activation(act_type=None):
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_type == 'mish':
        return nn.Mish(inplace=True)
    elif act_type == 'silu':
        return nn.SiLU(inplace=True)
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type is None:
        return nn.Identity()
    else:
        raise NotImplementedError
        
def get_norm(norm_type, dim):
    if norm_type == 'BN':
        return nn.BatchNorm2d(dim)
    elif norm_type == 'GN':
        return nn.GroupNorm(num_groups=32, num_channels=dim)
    elif norm_type is None:
        return nn.Identity()
    else:
        raise NotImplementedError

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class FrozenBatchNorm2d(torch.nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias
    
class BasicConv(nn.Module):
    def __init__(self, 
                 in_dim,                   # in channels
                 out_dim,                  # out channels 
                 kernel_size=1,            # kernel size 
                 padding=0,                # padding
                 stride=1,                 # padding
                 act_type  :str = 'lrelu', # activation
                 norm_type :str = 'BN',    # normalization
                ):
        super(BasicConv, self).__init__()
        add_bias = False if norm_type else True
        self.conv = get_conv2d(in_dim, out_dim, k=kernel_size, p=padding, s=stride, g=1, bias=add_bias)
        self.norm = get_norm(norm_type, out_dim)
        self.act  = get_activation(act_type)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class DepthwiseConv(nn.Module):
    def __init__(self, 
                 in_dim,                 # in channels
                 out_dim,                # out channels 
                 kernel_size=1,          # kernel size 
                 padding=0,              # padding
                 stride=1,               # padding
                 act_type  :str = None,  # activation
                 norm_type :str = 'BN',  # normalization
                ):
        super(DepthwiseConv, self).__init__()
        assert in_dim == out_dim
        add_bias = False if norm_type else True
        self.conv = get_conv2d(in_dim, out_dim, k=kernel_size, p=padding, s=stride, g=out_dim, bias=add_bias)
        self.norm = get_norm(norm_type, out_dim)
        self.act  = get_activation(act_type)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class PointwiseConv(nn.Module):
    def __init__(self, 
                 in_dim,                   # in channels
                 out_dim,                  # out channels 
                 act_type  :str = 'lrelu', # activation
                 norm_type :str = 'BN',    # normalization
                ):
        super(DepthwiseConv, self).__init__()
        assert in_dim == out_dim
        add_bias = False if norm_type else True
        self.conv = get_conv2d(in_dim, out_dim, k=1, p=0, s=1, g=1, bias=add_bias)
        self.norm = get_norm(norm_type, out_dim)
        self.act  = get_activation(act_type)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

## Yolov8's BottleNeck
class Bottleneck(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio = 0.5,
                 kernel_sizes = [3, 3],
                 shortcut     = True,
                 act_type     = 'silu',
                 norm_type    = 'BN',
                 depthwise    = False,):
        super(Bottleneck, self).__init__()
        inter_dim = int(out_dim * expand_ratio)
        if depthwise:
            self.cv1 = nn.Sequential(
                DepthwiseConv(in_dim, in_dim, kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2, act_type=act_type, norm_type=norm_type),
                PointwiseConv(in_dim, inter_dim, act_type=act_type, norm_type=norm_type),
            )
            self.cv2 = nn.Sequential(
                DepthwiseConv(inter_dim, inter_dim, kernel_size=kernel_sizes[1], padding=kernel_sizes[1]//2, act_type=act_type, norm_type=norm_type),
                PointwiseConv(inter_dim, out_dim, act_type=act_type, norm_type=norm_type),
            )
        else:
            self.cv1 = BasicConv(in_dim, inter_dim,  kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2, act_type=act_type, norm_type=norm_type)
            self.cv2 = BasicConv(inter_dim, out_dim, kernel_size=kernel_sizes[1], padding=kernel_sizes[1]//2, act_type=act_type, norm_type=norm_type)
        self.shortcut = shortcut and in_dim == out_dim

    def forward(self, x):
        h = self.cv2(self.cv1(x))

        return x + h if self.shortcut else h

# Yolov8's StageBlock
class RTCBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_blocks = 1,
                 shortcut   = False,
                 act_type   = 'silu',
                 norm_type  = 'BN',
                 depthwise  = False,):
        super(RTCBlock, self).__init__()
        self.inter_dim = out_dim // 2
        self.input_proj = BasicConv(in_dim, out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.m = nn.Sequential(*(
            Bottleneck(self.inter_dim, self.inter_dim, 1.0, [3, 3], shortcut, act_type, norm_type, depthwise)
            for _ in range(num_blocks)))
        self.output_proj = BasicConv((2 + num_blocks) * self.inter_dim, out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        # Input proj
        x1, x2 = torch.chunk(self.input_proj(x), 2, dim=1)
        out = list([x1, x2])

        # Bottlenecl
        out.extend(m(out[-1]) for m in self.m)

        # Output proj
        out = self.output_proj(torch.cat(out, dim=1))

        return out


# ----------------- Transformer modules -----------------
## Basic ops of Deformable Attn
def deformable_attention_core_func(value, value_spatial_shapes,
                                   value_level_start_index, sampling_locations,
                                   attention_weights):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, axis=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(
            [0, 2, 1]).reshape([bs * n_head, c, h, w])
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(
            [0, 2, 1, 3, 4]).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose([0, 2, 1, 3, 4]).reshape(
        [bs * n_head, 1, Len_q, n_levels * n_points])
    output = (torch.stack(
        sampling_value_list, axis=-2).flatten(-2) *
              attention_weights).sum(-1).reshape([bs, n_head * c, Len_q])

    return output.transpose([0, 2, 1])

class MSDeformableAttention(nn.Layer):
    def __init__(self,
                 embed_dim=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 lr_mult=0.1):
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

        self.sampling_offsets = nn.Linear(
            embed_dim,
            self.total_points * 2,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult))

        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        try:
            # use cuda op
            from deformable_detr_ops import ms_deformable_attn
            self.ms_deformable_attn_core = ms_deformable_attn
        except:
            # use paddle func
            self.ms_deformable_attn_core = deformable_attention_core_func

        self._reset_parameters()

    def _reset_parameters(self):
        # sampling_offsets
        constant_(self.sampling_offsets.weight)
        thetas = paddle.arange(
            self.num_heads,
            dtype=paddle.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = paddle.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)
        grid_init = grid_init.reshape([self.num_heads, 1, 1, 2]).tile(
            [1, self.num_levels, self.num_points, 1])
        scaling = paddle.arange(
            1, self.num_points + 1,
            dtype=paddle.float32).reshape([1, 1, -1, 1])
        grid_init *= scaling
        self.sampling_offsets.bias.set_value(grid_init.flatten())
        # attention_weights
        constant_(self.attention_weights.weight)
        constant_(self.attention_weights.bias)
        # proj
        xavier_uniform_(self.value_proj.weight)
        constant_(self.value_proj.bias)
        xavier_uniform_(self.output_proj.weight)
        constant_(self.output_proj.bias)

    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_level_start_index,
                value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (Tensor): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (Tensor(int64)): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]
        assert int(value_spatial_shapes.prod(1).sum()) == Len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape([bs, Len_v, self.num_heads, self.head_dim])

        sampling_offsets = self.sampling_offsets(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2])
        attention_weights = self.attention_weights(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels * self.num_points])
        attention_weights = F.softmax(attention_weights).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points])

        if reference_points.shape[-1] == 2:
            offset_normalizer = value_spatial_shapes.flip([1]).reshape(
                [1, 1, 1, self.num_levels, 1, 2])
            sampling_locations = reference_points.reshape([
                bs, Len_q, 1, self.num_levels, 1, 2
            ]) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :, None, :, None, 2:] *
                0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output

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

    def build_2d_sincos_position_embedding(self, w, h, embed_dim=256, temperature=10000.):
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
        pos_embed = torch.concat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h),torch.cos(out_h)], axis=1)[None, :, :]
        self.pos_embed = pos_embed

        return pos_embed

    def forward(self, src):
        """
        Input:
            src:       [torch.Tensor] -> [B, C, H, W]
        Output:
            src:       [torch.Tensor] -> [B, N, C]
        """
        # -------- Transformer encoder --------
        for encoder in self.encoder_layers:
            channels, fmp_h, fmp_w = src.shape[1:]
            # [B, C, H, W] -> [B, N, C], N=HxW
            src_flatten = src.flatten(2).permute(0, 2, 1)
            pos_embed = self.build_2d_sincos_position_embedding(
                    fmp_w, fmp_h, channels, self.pe_temperature)
            memory = encoder(src_flatten, pos_embed=pos_embed)
            # [B, N, C] -> [B, C, N] -> [B, C, H, W]
            src = memory.permute(0, 2, 1).reshape([-1, channels, fmp_h, fmp_w])

        return src

## Transformer Decoder layer
class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model         :int   = 256,
                 num_heads       :int   = 8,
                 num_levels      :int   = 3,
                 num_points      :int   = 4,
                 mlp_ratio       :float = 4.0,
                 dropout         :float = 0.1,
                 act_type        :str   = "relu",
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
        self.cross_attn = MSDeformableAttention(d_model, num_heads, num_levels, num_points, 1.0)
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
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None):
        # ---------------- MSHA for Object Query -----------------
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if attn_mask is not None:
            attn_mask = torch.where(
                attn_mask.astype('bool'),
                torch.zeros(attn_mask.shape, tgt.dtype),
                torch.full(attn_mask.shape, float("-inf"), tgt.dtype))
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ---------------- CMHA for Object Query and Image-feature -----------------
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos_embed),
                               reference_points,
                               memory,
                               memory_spatial_shapes,
                               memory_level_start_index,
                               memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ---------------- FeedForward Network -----------------
        tgt = self.ffn(tgt)

        return tgt

## Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self,
                 d_model        :int   = 256,
                 num_heads      :int   = 8,
                 num_layers     :int   = 1,
                 mlp_ratio      :float = 4.0,
                 pe_temperature :float = 10000.,
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
        self.decoder_layers = None

    def build_2d_sincos_position_embedding(self, w, h, embed_dim=256, temperature=10000.):
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
        pos_embed = torch.concat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h),torch.cos(out_h)], axis=1)[None, :, :]
        self.pos_embed = pos_embed

        return pos_embed

