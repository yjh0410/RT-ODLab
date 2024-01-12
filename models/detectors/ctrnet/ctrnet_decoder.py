import math
import torch.nn as nn

from .ctrnet_basic import DeConv, RTCBlock


def build_decoder(cfg, in_dim, out_dim):
    return CTRDecoder(in_dim     = in_dim,
                      out_dim    = out_dim,
                      max_stride = cfg['max_stride'],
                      out_stride = cfg['out_stride'],
                      act_type   = cfg['dec_act'],
                      norm_type  = cfg['dec_norm'],
                      depthwise  = cfg['dec_depthwise']
                      )


class CTRDecoder(nn.Module):
    def __init__(self,
                 in_dim     :int,
                 out_dim    :int,
                 max_stride :int,
                 out_stride :int,
                 act_type   :str,
                 norm_type  :str,
                 depthwise  :bool
                 ):
        super().__init__()
        # ---------- Basic parameters ----------
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.out_stride = out_stride
        self.num_layers = round(math.log2(max_stride // out_stride))

        # ---------- Network parameters ----------
        layers = []
        for _ in range(self.num_layers):
            layer = nn.Sequential(
                RTCBlock(in_dim, out_dim, 1, False, act_type, norm_type, depthwise),
                DeConv(out_dim, out_dim, kernel_size=4, stride=2, act_type=act_type, norm_type=norm_type)
            )
            layers.append(layer)
            in_dim = out_dim
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
