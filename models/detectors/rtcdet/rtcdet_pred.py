import torch
import torch.nn as nn
import torch.nn.functional as F


# Single-level pred layer
class SingleLevelPredLayer(nn.Module):
    def __init__(self, cls_dim, reg_dim, num_classes, num_coords=4):
        super().__init__()
        # --------- Basic Parameters ----------
        self.cls_dim = cls_dim
        self.reg_dim = reg_dim
        self.num_classes = num_classes
        self.num_coords = num_coords

        # --------- Network Parameters ----------
        self.cls_pred = nn.Conv2d(cls_dim, num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(reg_dim, num_coords, kernel_size=1)                

        self.init_bias()
        

    def init_bias(self):
        # Init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        # cls pred
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
    

# Multi-level pred layer
class MultiLevelPredLayer(nn.Module):
    def __init__(self, cls_dim, reg_dim, strides, num_classes, num_coords=4, num_levels=3, reg_max=16):
        super().__init__()
        # --------- Basic Parameters ----------
        self.cls_dim = cls_dim
        self.reg_dim = reg_dim
        self.strides = strides
        self.num_classes = num_classes
        self.num_coords = num_coords
        self.num_levels = num_levels
        self.reg_max = reg_max

        # ----------- Network Parameters -----------
        ## pred layers
        self.multi_level_preds = nn.ModuleList(
            [SingleLevelPredLayer(
                cls_dim,
                reg_dim,
                num_classes,
                num_coords * self.reg_max)
                for _ in range(num_levels)
            ])
        ## proj conv
        self.proj = nn.Parameter(torch.linspace(0, reg_max, reg_max), requires_grad=False)
        self.proj_conv = nn.Conv2d(self.reg_max, 1, kernel_size=1, bias=False)
        self.proj_conv.weight = nn.Parameter(self.proj.view([1, reg_max, 1, 1]).clone().detach(), requires_grad=False)


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
        

    def forward(self, cls_feats, reg_feats):
        all_anchors = []
        all_strides = []
        all_cls_preds = []
        all_reg_preds = []
        all_box_preds = []
        all_delta_preds = []
        for level in range(self.num_levels):
            # pred
            cls_pred, reg_pred = self.multi_level_preds[level](cls_feats[level], reg_feats[level])

            # generate anchor boxes: [M, 4]
            B, _, H, W = cls_pred.size()
            fmp_size = [H, W]
            anchors = self.generate_anchors(level, fmp_size)
            anchors = anchors.to(cls_pred.device)
            # stride tensor: [M, 1]
            stride_tensor = torch.ones_like(anchors[..., :1]) * self.strides[level]
            
            # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4*self.reg_max)

            # ----------------------- Decode bbox -----------------------
            B, M = reg_pred.shape[:2]
            # [B, M, 4*(reg_max)] -> [B, M, 4, reg_max] -> [B, 4, M, reg_max]
            delta_pred = reg_pred.reshape([B, M, 4, self.reg_max])
            # [B, M, 4, reg_max] -> [B, reg_max, 4, M]
            delta_pred = delta_pred.permute(0, 3, 2, 1).contiguous()
            # [B, reg_max, 4, M] -> [B, 1, 4, M]
            delta_pred = self.proj_conv(F.softmax(delta_pred, dim=1))
            # [B, 1, 4, M] -> [B, 4, M] -> [B, M, 4]
            delta_pred = delta_pred.view(B, 4, M).permute(0, 2, 1).contiguous()
            ## tlbr -> xyxy
            x1y1_pred = anchors[None] - delta_pred[..., :2] * self.strides[level]
            x2y2_pred = anchors[None] + delta_pred[..., 2:] * self.strides[level]
            box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

            all_cls_preds.append(cls_pred)
            all_reg_preds.append(reg_pred)
            all_box_preds.append(box_pred)
            all_delta_preds.append(delta_pred)
            all_anchors.append(anchors)
            all_strides.append(stride_tensor)
        
        # output dict
        outputs = {"pred_cls": all_cls_preds,        # List(Tensor) [B, M, C]
                   "pred_reg": all_reg_preds,        # List(Tensor) [B, M, 4*(reg_max)]
                   "pred_box": all_box_preds,        # List(Tensor) [B, M, 4]
                   "pred_delta": all_delta_preds,    # List(Tensor) [B, M, 4]
                   "anchors": all_anchors,           # List(Tensor) [M, 2]
                   "strides": self.strides,          # List(Int) = [8, 16, 32]
                   "stride_tensor": all_strides      # List(Tensor) [M, 1]
                   }

        return outputs
    

# build detection head
def build_pred_layer(cls_dim, reg_dim, strides, num_classes, num_coords=4, num_levels=3, reg_max=16):
    pred_layers = MultiLevelPredLayer(cls_dim, reg_dim, strides, num_classes, num_coords, num_levels, reg_max) 

    return pred_layers
