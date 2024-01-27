import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import BasicConv, RTCBlock


# Build PaFPN
def build_pafpn(cfg, in_dims, out_dim):
    return


# ----------------- Feature Pyramid Network -----------------
## Real-time Convolutional PaFPN
class HybridEncoder(nn.Module):
    def __init__(self, 
                 in_dims   = [256, 512, 512],
                 out_dim   = 256,
                 width     = 1.0,
                 depth     = 1.0,
                 act_type  = 'silu',
                 norm_type = 'BN',
                 depthwise = False):
        super(HybridEncoder, self).__init__()
        print('==============================')
        print('FPN: {}'.format("RTC-PaFPN"))
        # ---------------- Basic parameters ----------------
        self.in_dims = in_dims
        self.out_dim = round(out_dim * width)
        self.width = width
        self.depth = depth
        c3, c4, c5 = in_dims

        # ---------------- Input projs ----------------
        self.reduce_layer_1 = BasicConv(c5, self.out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.reduce_layer_2 = BasicConv(c4, self.out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.reduce_layer_3 = BasicConv(c3, self.out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)

        # ---------------- Downsample ----------------
        self.dowmsample_layer_1 = BasicConv(self.out_dim, self.out_dim, kernel_size=3, padding=1, stride=2, act_type=act_type, norm_type=norm_type)
        self.dowmsample_layer_2 = BasicConv(self.out_dim, self.out_dim, kernel_size=3, padding=1, stride=2, act_type=act_type, norm_type=norm_type)

        # ---------------- Top dwon FPN ----------------
        ## P5 -> P4
        self.top_down_layer_1 = RTCBlock(in_dim       = self.out_dim * 2,
                                         out_dim      = self.out_dim,
                                         num_blocks   = round(3*depth),
                                         shortcut     = False,
                                         act_type     = act_type,
                                         norm_type    = norm_type,
                                         depthwise    = depthwise,
                                         )
        ## P4 -> P3
        self.top_down_layer_2 = RTCBlock(in_dim       = self.out_dim * 2,
                                         out_dim      = self.out_dim,
                                         num_blocks   = round(3*depth),
                                         shortcut     = False,
                                         act_type     = act_type,
                                         norm_type    = norm_type,
                                         depthwise    = depthwise,
                                         )
        
        # ---------------- Bottom up PAN----------------
        ## P3 -> P4
        self.bottom_up_layer_1 = RTCBlock(in_dim       = self.out_dim * 2,
                                          out_dim      = self.out_dim,
                                          num_blocks   = round(3*depth),
                                          shortcut     = False,
                                          act_type     = act_type,
                                          norm_type    = norm_type,
                                          depthwise    = depthwise,
                                          )
        ## P4 -> P5
        self.bottom_up_layer_2 = RTCBlock(in_dim       = self.out_dim * 2,
                                          out_dim      = self.out_dim,
                                          num_blocks   = round(3*depth),
                                          shortcut     = False,
                                          act_type     = act_type,
                                          norm_type    = norm_type,
                                          depthwise    = depthwise,
                                          )

        self.init_weights()
        
    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()

    def forward(self, features):
        c3, c4, c5 = features

        # -------- Input projs --------
        p5 = self.reduce_layer_1(c5)
        p4 = self.reduce_layer_2(c4)
        p3 = self.reduce_layer_3(c3)

        # -------- Top down FPN --------
        p5_up = F.interpolate(p5, scale_factor=2.0)
        p4 = self.top_down_layer_1(torch.cat([p4, p5_up], dim=1))

        p4_up = F.interpolate(p4, scale_factor=2.0)
        p3 = self.top_down_layer_2(torch.cat([p3, p4_up], dim=1))

        # -------- Bottom up PAN --------
        p3_ds = self.dowmsample_layer_1(p3)
        p4 = self.bottom_up_layer_1(torch.cat([p4, p3_ds], dim=1))

        p4_ds = self.dowmsample_layer_2(p4)
        p5 = self.bottom_up_layer_2(torch.cat([p5, p4_ds], dim=1))

        out_feats = [p3, p4, p5]
        
        return out_feats
