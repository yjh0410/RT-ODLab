import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import Conv, RTCBlock


# Build PaFPN
def build_pafpn(cfg, in_dims, out_dim):
    return


# ----------------- Feature Pyramid Network -----------------
## Real-time Convolutional PaFPN
class RTCPaFPN(nn.Module):
    def __init__(self, 
                 in_dims   = [256, 512, 512],
                 width     = 1.0,
                 depth     = 1.0,
                 ratio     = 1.0,
                 act_type  = 'silu',
                 norm_type = 'BN',
                 depthwise = False):
        super(RTCPaFPN, self).__init__()
        print('==============================')
        print('FPN: {}'.format("RTC-PaFPN"))
        # ---------------- Basic parameters ----------------
        self.in_dims = in_dims
        self.width = width
        self.depth = depth
        self.out_dim = [round(256 * width), round(512 * width), round(512 * width * ratio)]
        c3, c4, c5 = in_dims

        # ---------------- Top dwon ----------------
        ## P5 -> P4
        self.top_down_layer_1 = RTCBlock(in_dim       = c5 + c4,
                                         out_dim      = round(512*width),
                                         num_blocks   = round(3*depth),
                                         shortcut     = False,
                                         act_type     = act_type,
                                         norm_type    = norm_type,
                                         depthwise    = depthwise,
                                         )
        ## P4 -> P3
        self.top_down_layer_2 = RTCBlock(in_dim       = round(512*width) + c3,
                                         out_dim      = round(256*width),
                                         num_blocks   = round(3*depth),
                                         shortcut     = False,
                                         act_type     = act_type,
                                         norm_type    = norm_type,
                                         depthwise    = depthwise,
                                         )
        # ---------------- Bottom up ----------------
        ## P3 -> P4
        self.dowmsample_layer_1 = Conv(round(256*width), round(256*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.bottom_up_layer_1  = RTCBlock(in_dim       = round(256*width) + round(512*width),
                                           out_dim      = round(512*width),
                                           num_blocks   = round(3*depth),
                                           shortcut     = False,
                                           act_type     = act_type,
                                           norm_type    = norm_type,
                                           depthwise    = depthwise,
                                           )
        ## P4 -> P5
        self.dowmsample_layer_2 = Conv(round(512*width), round(512*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.bottom_up_layer_2 = RTCBlock(in_dim       = round(512 * width) + c5,
                                          out_dim      = round(512 * width * ratio),
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

        # Top down
        ## P5 -> P4
        c6 = F.interpolate(c5, scale_factor=2.0)
        c7 = torch.cat([c6, c4], dim=1)
        c8 = self.top_down_layer_1(c7)
        ## P4 -> P3
        c9 = F.interpolate(c8, scale_factor=2.0)
        c10 = torch.cat([c9, c3], dim=1)
        c11 = self.top_down_layer_2(c10)

        # Bottom up
        # p3 -> P4
        c12 = self.dowmsample_layer_1(c11)
        c13 = torch.cat([c12, c8], dim=1)
        c14 = self.bottom_up_layer_1(c13)
        # P4 -> P5
        c15 = self.dowmsample_layer_2(c14)
        c16 = torch.cat([c15, c5], dim=1)
        c17 = self.bottom_up_layer_2(c16)

        out_feats = [c11, c14, c17] # [P3, P4, P5]
        
        return out_feats
