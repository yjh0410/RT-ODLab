import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor


# ------------------------------- Basic Modules -------------------------------
def get_activation(act_type=None):
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_type == 'mish':
        return nn.Mish(inplace=True)
    elif act_type == 'silu':
        return nn.SiLU(inplace=True)


def get_norm(norm_type, dim):
    if norm_type == 'BN':
        return nn.BatchNorm2d(dim)
    elif norm_type == 'GN':
        return nn.GroupNorm(num_groups=32, num_channels=dim)
    elif norm_type == 'LN':
        return nn.LayerNorm(dim)


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    

def build_multi_head_attention(d_model, num_heads, dropout, attn_type='mhsa'):
    if attn_type == 'mhsa':
        attn_layer = MultiHeadAttention(d_model, num_heads, dropout)
    elif attn_type == 's_mhsa':
        attn_layer = None

    return attn_layer


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
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# ------------------------------- Transformer Modules -------------------------------
## Vanilla Multi-Head Attention
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
        
## Transformer Encoder layer
class TREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 num_heads,
                 dim_feedforward=2048,
                 dropout=0.1,
                 act_type="relu",
                 attn_type='mhsa'
                 ):
        super().__init__()
        # Multi-head Self-Attn
        self.self_attn = build_multi_head_attention(d_model, num_heads, dropout, attn_type)

        # Feedforwaed Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation(act_type)


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
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffpn
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

## Transformer Decoder layer
class TRDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1, act_type="relu", attn_type='mhsa'):
        super().__init__()
        # Multi-head Self-Attn
        self.self_attn = build_multi_head_attention(d_model, num_heads, dropout, attn_type)
        self.cross_attn = build_multi_head_attention(d_model, num_heads, dropout)
        # Feedforward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation(act_type)


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos


    def forward(self, tgt, tgt_query_pos, memory, memory_pos):
        # self attention
        tgt2 = self.self_attn(
            query=self.with_pos_embed(tgt, tgt_query_pos),
            key=self.with_pos_embed(tgt, tgt_query_pos),
            value=tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, tgt_query_pos),
            key=self.with_pos_embed(memory, memory_pos),
            value=memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt
