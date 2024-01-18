import torch
import torch.nn as nn

try:
    from .rtcdet_basic import Conv
except:
    from rtcdet_basic import Conv


def build_det_head(cfg, in_dims, cls_head_dim, reg_head_dim, num_levels=3):
    head = MDetHead(cfg, in_dims, cls_head_dim, reg_head_dim, num_levels)

    return head

def build_seg_head(cfg, in_dims, out_dim):
    return MaskHead()

def build_pose_head(cfg, in_dims, out_dim):
    return PoseHead()


# ---------------------------- Detection Head ----------------------------
## Single-level Detection Head
class SDetHead(nn.Module):
    def __init__(self,
                 in_dim       :int  = 256,
                 cls_head_dim :int  = 256,
                 reg_head_dim :int  = 256,
                 num_cls_head :int  = 2,
                 num_reg_head :int  = 2,
                 act_type     :str  = "silu",
                 norm_type    :str  = "BN",
                 depthwise    :bool = False):
        super().__init__()
        # --------- Basic Parameters ----------
        self.in_dim = in_dim
        self.num_cls_head = num_cls_head
        self.num_reg_head = num_reg_head
        self.act_type = act_type
        self.norm_type = norm_type
        self.depthwise = depthwise
        
        # --------- Network Parameters ----------
        ## cls head
        cls_feats = []
        self.cls_head_dim = cls_head_dim
        for i in range(num_cls_head):
            if i == 0:
                cls_feats.append(
                    Conv(in_dim, self.cls_head_dim, k=3, p=1, s=1, 
                         act_type=act_type,
                         norm_type=norm_type,
                         depthwise=depthwise)
                        )
            else:
                cls_feats.append(
                    Conv(self.cls_head_dim, self.cls_head_dim, k=3, p=1, s=1, 
                        act_type=act_type,
                        norm_type=norm_type,
                        depthwise=depthwise)
                        )      
        ## reg head
        reg_feats = []
        self.reg_head_dim = reg_head_dim
        for i in range(num_reg_head):
            if i == 0:
                reg_feats.append(
                    Conv(in_dim, self.reg_head_dim, k=3, p=1, s=1, 
                         act_type=act_type,
                         norm_type=norm_type,
                         depthwise=depthwise)
                        )
            else:
                reg_feats.append(
                    Conv(self.reg_head_dim, self.reg_head_dim, k=3, p=1, s=1, 
                         act_type=act_type,
                         norm_type=norm_type,
                         depthwise=depthwise)
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
class MDetHead(nn.Module):
    def __init__(self, cfg, in_dims, cls_head_dim, reg_head_dim, num_levels=3):
        super().__init__()
        ## ----------- Network Parameters -----------
        self.multi_level_heads = nn.ModuleList(
            [SDetHead(in_dim       = in_dims[level],
                      cls_head_dim = cls_head_dim,
                      reg_head_dim = reg_head_dim,
                      num_cls_head = cfg['num_cls_head'],
                      num_reg_head = cfg['num_reg_head'],
                      act_type     = cfg['head_act'],
                      norm_type    = cfg['head_norm'],
                      depthwise    = cfg['head_depthwise'])
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
        'head': 'decoupled_head',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_act': 'silu',
        'head_norm': 'BN',
        'head_depthwise': False,
        'reg_max': 16,
    }
    fpn_dims = [256, 256, 256]
    out_dim = 256
    # Head-1
    model = build_det_head(cfg, fpn_dims, out_dim, num_levels=3)
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
