import torch
import torch.nn as nn

try:
    from .basic_modules.basic import BasicConv
except:
    from  basic_modules.basic import BasicConv


def build_decoder(cfg, in_dims, num_levels=3):
    if cfg['decoder'] == "det_decoder":
        decoder = MultiDetHead(cfg, in_dims, num_levels)
    elif cfg['decoder'] == "seg_decoder":
        decoder = MaskHead()
    elif cfg['decoder'] == "pos_decoder":
        decoder = PoseHead()

    return decoder


# ---------------------------- Detection Head ----------------------------
## Single-level Detection Head
class SingleDetHead(nn.Module):
    def __init__(self,
                 in_dim       :int  = 256,
                 cls_head_dim :int  = 256,
                 reg_head_dim :int  = 256,
                 num_cls_head :int  = 2,
                 num_reg_head :int  = 2,
                 act_type     :str  = "silu",
                 norm_type    :str  = "BN",
                 ):
        super().__init__()
        # --------- Basic Parameters ----------
        self.in_dim = in_dim
        self.num_cls_head = num_cls_head
        self.num_reg_head = num_reg_head
        self.act_type = act_type
        self.norm_type = norm_type
        
        # --------- Network Parameters ----------
        ## cls head
        cls_feats = []
        self.cls_head_dim = cls_head_dim
        for i in range(num_cls_head):
            if i == 0:
                cls_feats.append(
                    BasicConv(in_dim, self.cls_head_dim,
                              kernel_size=3, padding=1, stride=1, 
                              act_type=act_type, norm_type=norm_type)
                              )
            else:
                cls_feats.append(
                    BasicConv(self.cls_head_dim, self.cls_head_dim,
                              kernel_size=3, padding=1, stride=1, 
                              act_type=act_type, norm_type=norm_type)
                              )
        ## reg head
        reg_feats = []
        self.reg_head_dim = reg_head_dim
        for i in range(num_reg_head):
            if i == 0:
                cls_feats.append(
                    BasicConv(in_dim, self.reg_head_dim,
                              kernel_size=3, padding=1, stride=1, 
                              act_type=act_type, norm_type=norm_type)
                              )
            else:
                cls_feats.append(
                    BasicConv(self.reg_head_dim, self.reg_head_dim,
                              kernel_size=3, padding=1, stride=1, 
                              act_type=act_type, norm_type=norm_type)
                              )
        self.cls_feats = nn.Sequential(*cls_feats)
        self.reg_feats = nn.Sequential(*reg_feats)

        self.init_weights()
        
    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()

    def forward(self, x):
        """
            in_feats: (Tensor) [B, C, H, W]
        """
        cls_feats = self.cls_feats(x)
        reg_feats = self.reg_feats(x)

        return cls_feats, reg_feats
    
## Multi-level Detection Head
class MultiDetHead(nn.Module):
    def __init__(self, cfg, in_dims, num_levels=3):
        super().__init__()
        ## ----------- Network Parameters -----------
        self.multi_level_heads = nn.ModuleList(
            [SingleDetHead(in_dim       = in_dims[level],
                           cls_head_dim = cfg['hidden_dim'],
                           reg_head_dim = cfg['hidden_dim'],
                           num_cls_head = cfg['de_num_cls_layers'],
                           num_reg_head = cfg['de_num_reg_layers'],
                           act_type     = cfg['de_act'],
                           norm_type    = cfg['de_norm'],
                           )
                           for level in range(num_levels)
                           ])
        # --------- Basic Parameters ----------
        self.in_dims = in_dims
        self.cls_head_dim = self.multi_level_heads[0].cls_head_dim
        self.reg_head_dim = self.multi_level_heads[0].reg_head_dim

    def forward(self, feats):
        """
            feats: List[(Tensor)] [[B, C, H, W], ...]
        """
        cls_feats = []
        reg_feats = []
        for feat, head in zip(feats, self.multi_level_heads):
            # ---------------- Pred ----------------
            cls_feat, reg_feat = head(feat)

            cls_feats.append(cls_feat)
            reg_feats.append(reg_feat)

        outputs = {
            "cls_feat": cls_feats,
            "reg_feat": reg_feats
        }

        return outputs


# ---------------------------- Segmentation Head ----------------------------
class MaskHead(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return


# ---------------------------- Human-Pose Head ----------------------------
class PoseHead(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return


if __name__ == '__main__':
    import time
    from thop import profile
    cfg = {
        'width': 1.0,
        'depth': 1.0,
        # Decoder parameters
        'hidden_dim': 256,
        'decoder': 'det_decoder',
        'de_num_cls_layers': 2,
        'de_num_reg_layers': 2,
        'de_act': 'silu',
        'de_norm': 'BN',
    }
    fpn_dims = [256, 256, 256]
    out_dim = 256
    # Head-1
    model = build_decoder(cfg, fpn_dims, num_levels=3)
    print(model)
    fpn_feats = [torch.randn(1, fpn_dims[0], 80, 80), torch.randn(1, fpn_dims[1], 40, 40), torch.randn(1, fpn_dims[2], 20, 20)]
    t0 = time.time()
    outputs = model(fpn_feats)
    t1 = time.time()
    print('Time: ', t1 - t0)
    # for out in outputs:
    #     print(out.shape)

    print('==============================')
    flops, params = profile(model, inputs=(fpn_feats, ), verbose=False)
    print('==============================')
    print('Head-1: GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Head-1: Params : {:.2f} M'.format(params / 1e6))
