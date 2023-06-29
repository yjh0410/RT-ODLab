import torch
import torch.nn as nn

from .e2eyolo_basic import Conv


class SingleLevelHead(nn.Module):
    def __init__(self, in_dim, out_dim, num_classes, num_cls_head, num_reg_head, act_type, norm_type, depthwise):
        super().__init__()
        # --------- Basic Parameters ----------
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.num_cls_head = num_cls_head
        self.num_reg_head = num_reg_head
        self.act_type = act_type
        self.norm_type = norm_type
        self.depthwise = depthwise
        
        # --------- Network Parameters ----------
        ## cls head
        cls_feats = []
        self.cls_out_dim = out_dim
        for i in range(num_cls_head):
            if i == 0:
                cls_feats.append(
                    Conv(in_dim, self.cls_out_dim, k=3, p=1, s=1, 
                         act_type=act_type,
                         norm_type=norm_type,
                         depthwise=depthwise)
                        )
            else:
                cls_feats.append(
                    Conv(self.cls_out_dim, self.cls_out_dim, k=3, p=1, s=1, 
                        act_type=act_type,
                        norm_type=norm_type,
                        depthwise=depthwise)
                        )      
        ## reg head
        reg_feats = []
        self.reg_out_dim = out_dim
        for i in range(num_reg_head):
            if i == 0:
                reg_feats.append(
                    Conv(in_dim, self.reg_out_dim, k=3, p=1, s=1, 
                         act_type=act_type,
                         norm_type=norm_type,
                         depthwise=depthwise)
                        )
            else:
                reg_feats.append(
                    Conv(self.reg_out_dim, self.reg_out_dim, k=3, p=1, s=1, 
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
    

class MultiLevelHead(nn.Module):
    def __init__(self, cfg, in_dims, out_dim, num_classes=80):
        super().__init__()
        # --------- Basic Parameters ----------
        self.in_dims = in_dims
        self.num_classes = num_classes

        ## ----------- Network Parameters -----------
        self.det_heads = nn.ModuleList(
            [SingleLevelHead(
                in_dim,
                out_dim,
                num_classes,
                cfg['num_cls_head'],
                cfg['num_reg_head'],
                cfg['head_act'],
                cfg['head_norm'],
                cfg['head_depthwise'])
                for in_dim in in_dims
            ])


    def forward(self, feats):
        """
            feats: List[(Tensor)] [[B, C, H, W], ...]
        """
        cls_feats = []
        reg_feats = []
        for feat, head in zip(feats, self.det_heads):
            # ---------------- Pred ----------------
            cls_feat, reg_feat = head(feat)

            cls_feats.append(cls_feat)
            reg_feats.append(reg_feat)

        return cls_feats, reg_feats
    

# build detection head
def build_head(cfg, in_dim, out_dim, num_classes=80):
    if cfg['head'] == 'decoupled_head':
        head = MultiLevelHead(cfg, in_dim, out_dim, num_classes) 

    return head
