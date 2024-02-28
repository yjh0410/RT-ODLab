import torch
import torch.nn as nn

from .yolov4_basic import Conv


class DecoupledHead(nn.Module):
    def __init__(self, cfg, in_dim, out_dim, num_classes=80):
        super().__init__()
        print('==============================')
        print('Head: Decoupled Head')
        self.in_dim = in_dim
        self.num_cls_head=cfg['num_cls_head']
        self.num_reg_head=cfg['num_reg_head']
        self.act_type=cfg['head_act']
        self.norm_type=cfg['head_norm']

        # cls head
        cls_feats = []
        self.cls_out_dim = max(out_dim, num_classes)
        for i in range(cfg['num_cls_head']):
            if i == 0:
                cls_feats.append(
                    Conv(in_dim, self.cls_out_dim, k=3, p=1, s=1, 
                        act_type=self.act_type,
                        norm_type=self.norm_type,
                        depthwise=cfg['head_depthwise'])
                        )
            else:
                cls_feats.append(
                    Conv(self.cls_out_dim, self.cls_out_dim, k=3, p=1, s=1, 
                        act_type=self.act_type,
                        norm_type=self.norm_type,
                        depthwise=cfg['head_depthwise'])
                        )
                
        # reg head
        reg_feats = []
        self.reg_out_dim = max(out_dim, 64)
        for i in range(cfg['num_reg_head']):
            if i == 0:
                reg_feats.append(
                    Conv(in_dim, self.reg_out_dim, k=3, p=1, s=1, 
                        act_type=self.act_type,
                        norm_type=self.norm_type,
                        depthwise=cfg['head_depthwise'])
                        )
            else:
                reg_feats.append(
                    Conv(self.reg_out_dim, self.reg_out_dim, k=3, p=1, s=1, 
                        act_type=self.act_type,
                        norm_type=self.norm_type,
                        depthwise=cfg['head_depthwise'])
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
    

# build detection head
def build_head(cfg, in_dim, out_dim, num_classes=80):
    head = DecoupledHead(cfg, in_dim, out_dim, num_classes) 

    return head
