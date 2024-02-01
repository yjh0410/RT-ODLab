import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

try:
    from .basic import BasicConv, RTCBlock, CSPRepLayer
    from .transformer import TransformerEncoder
except:
    from  basic import BasicConv, RTCBlock, CSPRepLayer
    from  transformer import TransformerEncoder


# Build PaFPN
def build_fpn(cfg, in_dims, out_dim):
    if cfg['fpn'] == 'hybrid_encoder':
        return HybridEncoder(in_dims     = in_dims,
                             out_dim     = out_dim,
                             depth       = cfg['depth'],
                             act_type    = cfg['fpn_act'],
                             norm_type   = cfg['fpn_norm'],
                             depthwise   = cfg['fpn_depthwise'],
                             num_heads   = cfg['en_num_heads'],
                             num_layers  = cfg['en_num_layers'],
                             mlp_ratio   = cfg['en_mlp_ratio'],
                             dropout     = cfg['en_dropout'],
                             pe_temperature = cfg['pe_temperature'],
                             en_act_type    = cfg['en_act'],
                             )
    elif cfg['fpn'] == 'pp_hybrid_encoder':
        return PPHybridEncoder(in_dims     = in_dims,
                               out_dim     = out_dim,
                               depth       = cfg['depth'],
                               expansion   = cfg['expansion'],
                               act_type    = cfg['fpn_act'],
                               norm_type   = cfg['fpn_norm'],
                               depthwise   = cfg['fpn_depthwise'],
                               num_heads   = cfg['en_num_heads'],
                               num_layers  = cfg['en_num_layers'],
                               mlp_ratio   = cfg['en_mlp_ratio'],
                               dropout     = cfg['en_dropout'],
                               pe_temperature = cfg['pe_temperature'],
                               en_act_type    = cfg['en_act'],
                               )
    else:
        raise NotImplementedError("Unknown PaFPN: <{}>".format(cfg['fpn']))


# ----------------- Feature Pyramid Network -----------------
## Hybrid Encoder (Transformer encoder + Convolutional PaFPN)
class HybridEncoder(nn.Module):
    def __init__(self, 
                 in_dims     :List  = [256, 512, 1024],
                 out_dim     :int   = 256,
                 depth       :float = 1.0,
                 act_type    :str   = 'silu',
                 norm_type   :str   = 'BN',
                 depthwise   :bool  = False,
                 # Transformer's parameters
                 num_heads      :int   = 8,
                 num_layers     :int   = 1,
                 mlp_ratio      :float = 4.0,
                 dropout        :float = 0.1,
                 pe_temperature :float = 10000.,
                 en_act_type    :str   = 'gelu'
                 ) -> None:
        super(HybridEncoder, self).__init__()
        print('==============================')
        print('FPN: {}'.format("RTC-PaFPN"))
        # ---------------- Basic parameters ----------------
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.out_dims = [self.out_dim] * len(in_dims)
        self.depth = depth
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        c3, c4, c5 = in_dims

        # ---------------- Input projs ----------------
        self.reduce_layer_1 = BasicConv(c5, self.out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.reduce_layer_2 = BasicConv(c4, self.out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.reduce_layer_3 = BasicConv(c3, self.out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)

        # ---------------- Downsample ----------------
        self.dowmsample_layer_1 = BasicConv(self.out_dim, self.out_dim, kernel_size=3, padding=1, stride=2, act_type=act_type, norm_type=norm_type)
        self.dowmsample_layer_2 = BasicConv(self.out_dim, self.out_dim, kernel_size=3, padding=1, stride=2, act_type=act_type, norm_type=norm_type)

        # ---------------- Transformer Encoder ----------------
        self.transformer_encoder = TransformerEncoder(d_model        = self.out_dim,
                                                      num_heads      = num_heads,
                                                      num_layers     = num_layers,
                                                      mlp_ratio      = mlp_ratio,
                                                      pe_temperature = pe_temperature,
                                                      dropout        = dropout,
                                                      act_type       = en_act_type
                                                      )

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

        # -------- Transformer encoder --------
        p5 = self.transformer_encoder(p5)

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


## PaddlePaddle Hybrid Encoder (Transformer encoder + Convolutional PaFPN)
class PPHybridEncoder(nn.Module):
    def __init__(self, 
                 in_dims     :List  = [256, 512, 1024],
                 out_dim     :int   = 256,
                 depth       :float = 1.0,
                 expansion   :float = 1.0,
                 act_type    :str   = 'silu',
                 norm_type   :str   = 'BN',
                 depthwise   :bool  = False,
                 # Transformer's parameters
                 num_heads      :int   = 8,
                 num_layers     :int   = 1,
                 mlp_ratio      :float = 4.0,
                 dropout        :float = 0.1,
                 pe_temperature :float = 10000.,
                 en_act_type    :str   = 'gelu'
                 ) -> None:
        super(PPHybridEncoder, self).__init__()
        print('==============================')
        print('FPN: {}'.format("RTC-PaFPN"))
        # ---------------- Basic parameters ----------------
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.out_dims = [self.out_dim] * len(in_dims)
        self.depth = depth
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        c3, c4, c5 = in_dims

        # ---------------- Input projs ----------------
        self.reduce_layer_1 = BasicConv(c5, self.out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.reduce_layer_2 = BasicConv(c4, self.out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.reduce_layer_3 = BasicConv(c3, self.out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)

        # ---------------- Downsample ----------------
        self.dowmsample_layer_1 = BasicConv(self.out_dim, self.out_dim, kernel_size=3, padding=1, stride=2, act_type=act_type, norm_type=norm_type)
        self.dowmsample_layer_2 = BasicConv(self.out_dim, self.out_dim, kernel_size=3, padding=1, stride=2, act_type=act_type, norm_type=norm_type)

        # ---------------- Transformer Encoder ----------------
        self.transformer_encoder = TransformerEncoder(d_model        = self.out_dim,
                                                      num_heads      = num_heads,
                                                      num_layers     = num_layers,
                                                      mlp_ratio      = mlp_ratio,
                                                      pe_temperature = pe_temperature,
                                                      dropout        = dropout,
                                                      act_type       = en_act_type
                                                      )

        # ---------------- Top dwon FPN ----------------
        ## P5 -> P4
        self.top_down_layer_1 = CSPRepLayer(in_dim       = self.out_dim * 2,
                                            out_dim      = self.out_dim,
                                            num_blocks   = round(3*depth),
                                            expansion    = expansion,
                                            act_type     = act_type,
                                            norm_type    = norm_type,
                                            )
        ## P4 -> P3
        self.top_down_layer_2 = CSPRepLayer(in_dim       = self.out_dim * 2,
                                            out_dim      = self.out_dim,
                                            num_blocks   = round(3*depth),
                                            expansion    = expansion,
                                            act_type     = act_type,
                                            norm_type    = norm_type,
                                            )
        
        # ---------------- Bottom up PAN----------------
        ## P3 -> P4
        self.bottom_up_layer_1 = CSPRepLayer(in_dim       = self.out_dim * 2,
                                             out_dim      = self.out_dim,
                                             num_blocks   = round(3*depth),
                                             expansion    = expansion,
                                             act_type     = act_type,
                                             norm_type    = norm_type,
                                             )
        ## P4 -> P5
        self.bottom_up_layer_2 = CSPRepLayer(in_dim       = self.out_dim * 2,
                                             out_dim      = self.out_dim,
                                             num_blocks   = round(3*depth),
                                             expansion    = expansion,
                                             act_type     = act_type,
                                             norm_type    = norm_type,
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

        # -------- Transformer encoder --------
        p5 = self.transformer_encoder(p5)

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


if __name__ == '__main__':
    import time
    from thop import profile
    cfg = {
        'width': 1.0,
        'depth': 1.0,
        'fpn': 'hybrid_encoder',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        'expansion': 1.0,
        'en_num_heads': 8,
        'en_num_layers': 1,
        'en_mlp_ratio': 4.0,
        'en_dropout': 0.0,
        'pe_temperature': 10000.,
        'en_act': 'gelu',
    }
    fpn_dims = [256, 512, 1024]
    out_dim = 256
    pyramid_feats = [torch.randn(1, fpn_dims[0], 80, 80), torch.randn(1, fpn_dims[1], 40, 40), torch.randn(1, fpn_dims[2], 20, 20)]
    model = build_fpn(cfg, fpn_dims, out_dim)

    t0 = time.time()
    outputs = model(pyramid_feats)
    t1 = time.time()
    print('Time: ', t1 - t0)
    for out in outputs:
        print(out.shape)

    print('==============================')
    flops, params = profile(model, inputs=(pyramid_feats, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
