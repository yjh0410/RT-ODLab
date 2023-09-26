import copy
import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor


# ---------------------------- Basic functions ----------------------------
class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def get_conv2d(c1, c2, k, p, s, d, g, bias=False):
    conv = nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias)

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
    elif act_type is None:
        return nn.Identity()

def get_norm(norm_type, dim):
    if norm_type == 'BN':
        return nn.BatchNorm2d(dim)
    elif norm_type == 'GN':
        return nn.GroupNorm(num_groups=32, num_channels=dim)

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    

# ---------------------------- 2D CNN ----------------------------
class Conv(nn.Module):
    def __init__(self, 
                 c1,                   # in channels
                 c2,                   # out channels 
                 k=1,                  # kernel size 
                 p=0,                  # padding
                 s=1,                  # padding
                 d=1,                  # dilation
                 act_type='lrelu',     # activation
                 norm_type='BN',       # normalization
                 depthwise=False):
        super(Conv, self).__init__()
        convs = []
        add_bias = False if norm_type else True
        p = p if d == 1 else d
        if depthwise:
            convs.append(get_conv2d(c1, c1, k=k, p=p, s=s, d=d, g=c1, bias=add_bias))
            # depthwise conv
            if norm_type:
                convs.append(get_norm(norm_type, c1))
            if act_type:
                convs.append(get_activation(act_type))
            # pointwise conv
            convs.append(get_conv2d(c1, c2, k=1, p=0, s=1, d=d, g=1, bias=add_bias))
            if norm_type:
                convs.append(get_norm(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))

        else:
            convs.append(get_conv2d(c1, c2, k=k, p=p, s=s, d=d, g=1, bias=add_bias))
            if norm_type:
                convs.append(get_norm(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))
            
        self.convs = nn.Sequential(*convs)


    def forward(self, x):
        return self.convs(x)


# ------------------------------- MLP -------------------------------
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

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
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src
    
# ---------------------------- Attention ----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.) -> None:
        super().__init__()
        # --------------- Basic parameters ---------------
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.scale = (d_model // num_heads) ** -0.5

        # --------------- Network parameters ---------------
        self.q_proj = nn.Linear(d_model, d_model, bias = False) # W_q, W_k, W_v
        self.k_proj = nn.Linear(d_model, d_model, bias = False) # W_q, W_k, W_v
        self.v_proj = nn.Linear(d_model, d_model, bias = False) # W_q, W_k, W_v

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, query, key, value):
        """
        Inputs:
            query : (Tensor) -> [B, Nq, C]
            key   : (Tensor) -> [B, Nk, C]
            value : (Tensor) -> [B, Nk, C]
        """
        bs = query.shape[0]
        Nq = query.shape[1]
        Nk = key.shape[1]

        # ----------------- Input proj -----------------
        query = self.q_proj(query)
        key   = self.k_proj(key)
        value = self.v_proj(value)

        # ----------------- Multi-head Attn -----------------
        ## [B, N, C] -> [B, N, H, C_h] -> [B, H, N, C_h]
        query = query.view(bs, Nq, self.num_heads, self.d_model // self.num_heads)
        query = query.permute(0, 2, 1, 3).contiguous()
        key   = key.view(bs, Nk, self.num_heads, self.d_model // self.num_heads)
        key   = key.permute(0, 2, 1, 3).contiguous()
        value = value.view(bs, Nk, self.num_heads, self.d_model // self.num_heads)
        value = value.permute(0, 2, 1, 3).contiguous()
        # Attention
        ## [B, H, Nq, C_h] X [B, H, C_h, Nk] = [B, H, Nq, Nk]
        sim_matrix = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        sim_matrix = torch.softmax(sim_matrix, dim=-1)

        # ----------------- Output -----------------
        out = torch.matmul(sim_matrix, value)  # [B, H, Nq, C_h]
        out = out.permute(0, 2, 1, 3).contiguous().view(bs, Nq, -1)
        out = self.out_proj(out)

        return out
        

# ---------------------------- Modified YOLOv7's Modules ----------------------------
class ELANBlock(nn.Module):
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, depth=1.0, act_type='silu', norm_type='BN', depthwise=False):
        super(ELANBlock, self).__init__()
        if isinstance(expand_ratio, float):
            inter_dim = int(in_dim * expand_ratio)
            inter_dim2 = inter_dim
        elif isinstance(expand_ratio, list):
            assert len(expand_ratio) == 2
            e1, e2 = expand_ratio
            inter_dim = int(in_dim * e1)
            inter_dim2 = int(inter_dim * e2)
        # branch-1
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        # branch-2
        self.cv2 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        # branch-3
        for idx in range(round(3*depth)):
            if idx == 0:
                cv3 = [Conv(inter_dim, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)]
            else:
                cv3.append(Conv(inter_dim2, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise))
        self.cv3 = nn.Sequential(*cv3)
        # branch-4
        self.cv4 = nn.Sequential(*[
            Conv(inter_dim2, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
            for _ in range(round(3*depth))
        ])
        # output
        self.out = Conv(inter_dim*2 + inter_dim2*2, out_dim, k=1, act_type=act_type, norm_type=norm_type)


    def forward(self, x):
        """
        Input:
            x: [B, C_in, H, W]
        Output:
            out: [B, C_out, H, W]
        """
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)

        # [B, C, H, W] -> [B, 2C, H, W]
        out = self.out(torch.cat([x1, x2, x3, x4], dim=1))

        return out

class ELANBlockFPN(nn.Module):
    def __init__(self, in_dim, out_dim, expand_ratio :float=0.5, branch_depth :int=1, shortcut=False, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        # ----------- Basic Parameters -----------
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inter_dim1 = round(out_dim * expand_ratio)
        self.inter_dim2 = round(self.inter_dim1 * expand_ratio)
        self.expand_ratio = expand_ratio
        self.branch_depth = branch_depth
        self.shortcut = shortcut
        # ----------- Network Parameters -----------
        ## branch-1
        self.cv1 = Conv(in_dim, self.inter_dim1, k=1, act_type=act_type, norm_type=norm_type)
        ## branch-2
        self.cv2 = Conv(in_dim, self.inter_dim1, k=1, act_type=act_type, norm_type=norm_type)
        ## branch-3
        self.cv3 = []
        for i in range(branch_depth):
            if i == 0:
                self.cv3.append(Conv(self.inter_dim1, self.inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise))
            else:
                self.cv3.append(Conv(self.inter_dim2, self.inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise))
        self.cv3 = nn.Sequential(*self.cv3)
        ## branch-4
        self.cv4 = nn.Sequential(*[
            Conv(self.inter_dim2, self.inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
            for _ in range(branch_depth)
        ])
        ## branch-5
        self.cv5 = nn.Sequential(*[
            Conv(self.inter_dim2, self.inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
            for _ in range(branch_depth)
        ])
        ## branch-6
        self.cv6 = nn.Sequential(*[
            Conv(self.inter_dim2, self.inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
            for _ in range(branch_depth)
        ])
        ## output proj
        self.out = Conv(self.inter_dim1*2 + self.inter_dim2*4, out_dim, k=1, act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)
        x5 = self.cv5(x4)
        x6 = self.cv6(x5)

        # [B, C, H, W] -> [B, 2C, H, W]
        out = self.out(torch.cat([x1, x2, x3, x4, x5, x6], dim=1))

        return out
    
class DSBlock(nn.Module):
    def __init__(self, in_dim, out_dim, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        inter_dim = out_dim // 2
        self.mp = nn.MaxPool2d((2, 2), 2)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(inter_dim, inter_dim, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )

    def forward(self, x):
        x1 = self.cv1(self.mp(x))
        x2 = self.cv2(x)
        out = torch.cat([x1, x2], dim=1)

        return out


# ---------------------------- Transformer Modules ----------------------------
class TREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 num_heads,
                 mlp_ratio=4.0,
                 dropout=0.1,
                 act_type="relu",
                 ):
        super().__init__()
        # Multi-head Self-Attn
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # Feedforwaed Network
        self.ffn = FFN(d_model, mlp_ratio, dropout, act_type)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos):
        """
        Input:
            src: [torch.Tensor] -> [B, N, C]
            pos: [torch.Tensor] -> [B, N, C]
        Output:
            src: [torch.Tensor] -> [B, N, C]
        """
        q = k = self.with_pos_embed(src, pos)

        # self-attn
        src2 = self.self_attn(q, k, value=src)

        # reshape: [B, N, C] -> [B, C, H, W]
        src = src + self.dropout(src2)
        src = self.norm(src)

        # ffpn
        src = self.ffn(src)
        
        return src

class TRDecoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 num_heads,
                 mlp_ratio=4.0,
                 dropout=0.1,
                 act_type="relu"):
        super().__init__()
        self.d_model = d_model
        # self attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # cross attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # FFN
        self.ffn = FFN(d_model, mlp_ratio, dropout, act_type)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, query_pos, memory, memory_pos):
        # self attention
        q1 = k1 = self.with_pos_embed(tgt, query_pos)
        v1 = tgt
        tgt2 = self.self_attn(q1, k1, v1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        q2 = self.with_pos_embed(tgt, query_pos)
        k2 = self.with_pos_embed(memory, memory_pos)
        v2 = memory
        tgt2 = self.cross_attn(q2, k2, v2)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt = self.ffn(tgt)

        return tgt
    