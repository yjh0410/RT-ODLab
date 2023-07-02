import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleLevelPredLayer(nn.Module):
    def __init__(self, cfg, cls_dim, reg_dim, num_classes):
        super().__init__()
        # --------- Basic Parameters ----------
        self.cfg = cfg
        self.cls_dim = cls_dim
        self.reg_dim = reg_dim
        self.num_classes = num_classes

        # --------- Network Parameters ----------
        ## pred_conv
        self.cls_pred = nn.Conv2d(cls_dim, num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(reg_dim, 4*cfg['reg_max'], kernel_size=1)                

        self.init_weight()
        

    def init_weight(self):
        # cls pred
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        b = self.cls_pred.bias.view(1, -1)
        b.data.fill_(bias_value.item())
        self.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        # reg pred
        b = self.reg_pred.bias.view(-1, )
        b.data.fill_(1.0)
        self.reg_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        w = self.reg_pred.weight
        w.data.fill_(0.)
        self.reg_pred.weight = torch.nn.Parameter(w, requires_grad=True)


    def forward(self, cls_feat, reg_feat):
        """
            in_feats: (Tensor) [B, C, H, W]
        """
        cls_pred = self.cls_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)

        return cls_pred, reg_pred
    

class MultiLevelPredLayer(nn.Module):
    def __init__(self, cfg, cls_dim, reg_dim, strides, num_classes, num_levels=3):
        super().__init__()
        # --------- Basic Parameters ----------
        self.cfg = cfg
        self.cls_dim = cls_dim
        self.reg_dim = reg_dim
        self.strides = strides
        self.num_classes = num_classes
        self.num_levels = num_levels

        # ----------- Network Parameters -----------
        ## proj_conv
        self.proj = nn.Parameter(torch.linspace(0, cfg['reg_max'], cfg['reg_max']), requires_grad=False)
        self.proj_conv = nn.Conv2d(cfg['reg_max'], 1, kernel_size=1, bias=False)
        self.proj_conv.weight = nn.Parameter(self.proj.view([1, cfg['reg_max'], 1, 1]).clone().detach(), requires_grad=False)
        ## pred layers
        self.multi_level_preds = nn.ModuleList(
            [SingleLevelPredLayer(
                cfg,
                cls_dim,
                reg_dim,
                num_classes)
                for _ in range(num_levels)
            ])

    def generate_anchors(self, level, fmp_size):
        """
            fmp_size: (List) [H, W]
        """
        # generate grid cells
        fmp_h, fmp_w = fmp_size
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        anchors = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2)
        anchors += 0.5  # add center offset
        anchors *= self.strides[level]

        return anchors
        

    def decode_bbox(self, reg_pred, anchors, stride):
        # ----------------------- Decode bbox -----------------------
        B, M = reg_pred.shape[:2]
        # [B, M, 4*(reg_max)] -> [B, M, 4, reg_max] -> [B, 4, M, reg_max]
        reg_pred = reg_pred.reshape([B, M, 4, self.reg_max])
        # [B, M, 4, reg_max] -> [B, reg_max, 4, M]
        reg_pred = reg_pred.permute(0, 3, 2, 1).contiguous()
        # [B, reg_max, 4, M] -> [B, 1, 4, M]
        reg_pred = self.proj_conv(F.softmax(reg_pred, dim=1))
        # [B, 1, 4, M] -> [B, 4, M] -> [B, M, 4]
        reg_pred = reg_pred.view(B, 4, M).permute(0, 2, 1).contiguous()
        ## tlbr -> xyxy
        x1y1_pred = anchors[None] - reg_pred[..., :2] * stride
        x2y2_pred = anchors[None] + reg_pred[..., 2:] * stride
        box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

        return box_pred
    

    def forward(self, cls_feats, reg_feats):
        """
            feats: List[(Tensor)] [[B, C, H, W], ...]
        """
        all_anchors = []
        all_strides = []
        all_cls_preds = []
        all_reg_preds = []
        all_box_preds = []
        for level in range(self.num_levels):
            cls_pred, reg_pred = self.multi_level_preds[level](cls_feats[level], reg_feats[level])

            B, _, H, W = cls_pred.size()
            fmp_size = [H, W]
            # generate anchor boxes: [M, 4]
            anchors = self.generate_anchors(level, fmp_size)
            anchors = anchors.to(cls_pred.device)

            # stride tensor: [M, 1]
            stride_tensor = torch.ones_like(anchors[..., :1]) * self.stride[level]

            # process preds
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4*self.cfg['reg_max'])
            box_pred = self.decode_bbox(reg_pred, anchors, self.strides[level])

            # collect preds
            all_cls_preds.append(cls_pred)
            all_reg_preds.append(reg_pred)
            all_box_preds.append(box_pred)
            all_anchors.append(anchors)
            all_strides.append(stride_tensor)

            # output dict
            outputs = {"pred_cls": all_cls_preds,        # List(Tensor) [B, M, C]
                       "pred_reg": all_reg_preds,        # List(Tensor) [B, M, 4*(reg_max)]
                       "pred_box": all_box_preds,        # List(Tensor) [B, M, 4]
                       "anchors": all_anchors,           # List(Tensor) [M, 2]
                       "strides": self.strides,           # List(Int) = [8, 16, 32]
                       "stride_tensor": all_strides      # List(Tensor) [M, 1]
                       }
            
            return outputs 
    

# build detection head
def build_pred_layer(cfg, cls_dim, reg_dim, strides, num_classes, num_levels=3):
    pred_layers = MultiLevelPredLayer(cfg, cls_dim, reg_dim, strides, num_classes, num_levels) 

    return pred_layers
