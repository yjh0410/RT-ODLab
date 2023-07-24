import torch
import torch.nn as nn

from .rtcdet_v2_basic import Conv


# Single-level Head
class SingleLevelHead(nn.Module):
    def __init__(self, in_dim, cls_head_dim, reg_head_dim, num_cls_head, num_reg_head, act_type, norm_type, depthwise):
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


    def forward(self, x):
        """
            in_feats: (Tensor) [B, C, H, W]
        """
        cls_feats = self.cls_feats(x)
        reg_feats = self.reg_feats(x)

        return cls_feats, reg_feats
    

# Multi-level Head
class MultiLevelHead(nn.Module):
    def __init__(self, cfg, in_dims, out_dim, num_classes=80, reg_max=16, num_levels=3):
        super().__init__()
        ## ----------- Network Parameters -----------
        self.multi_level_heads = nn.ModuleList(
            [SingleLevelHead(
                in_dims[level],
                max(out_dim, num_classes),   # cls head dim
                max(out_dim//4, 4*reg_max),  # reg head dim
                cfg['num_cls_head'],
                cfg['num_reg_head'],
                cfg['head_act'],
                cfg['head_norm'],
                cfg['head_depthwise'])
                for level in range(num_levels)
            ])
        # --------- Basic Parameters ----------
        self.in_dims = in_dims
        self.num_classes = num_classes

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

        return cls_feats, reg_feats
    

# build detection head
def build_det_head(cfg, in_dim, out_dim, num_classes=80, reg_max=16, num_levels=3):
    if cfg['head'] == 'decoupled_head':
        head = MultiLevelHead(cfg, in_dim, out_dim, num_classes, reg_max, num_levels) 

    return head