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
        self.obj_pred = nn.Conv2d(reg_dim, 1, kernel_size=1)
        self.cls_pred = nn.Conv2d(cls_dim, num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(reg_dim, num_coords, kernel_size=1)                

        self.init_bias()
        

    def init_bias(self):
        # Init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        # obj pred
        b = self.obj_pred.bias.view(1, -1)
        b.data.fill_(bias_value.item())
        self.obj_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
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
        obj_pred = self.obj_pred(reg_feat)
        cls_pred = self.cls_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)

        return obj_pred, cls_pred, reg_pred
    

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
        all_obj_preds = []
        all_cls_preds = []
        all_reg_preds = []
        all_box_preds = []
        for level in range(self.num_levels):
            # pred
            obj_pred, cls_pred, reg_pred = self.multi_level_preds[level](
                cls_feats[level], reg_feats[level])

            # generate anchor boxes: [M, 4]
            B, _, H, W = cls_pred.size()
            fmp_size = [H, W]
            anchors = self.generate_anchors(level, fmp_size)
            anchors = anchors.to(cls_pred.device)
            # stride tensor: [M, 1]
            stride_tensor = torch.ones_like(anchors[..., :1]) * self.strides[level]
            
            # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
            obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4*self.reg_max)

            # ----------------------- Decode bbox -----------------------
            B, M = reg_pred.shape[:2]
            # [B, M, 4*(reg_max)] -> [B, M, 4, reg_max] -> [B, 4, M, reg_max]
            reg_pred_ = reg_pred.reshape([B, M, 4, self.reg_max])
            # [B, M, 4, reg_max] -> [B, reg_max, 4, M]
            reg_pred_ = reg_pred_.permute(0, 3, 2, 1).contiguous()
            # [B, reg_max, 4, M] -> [B, 1, 4, M]
            reg_pred_ = self.proj_conv(F.softmax(reg_pred_, dim=1))
            # [B, 1, 4, M] -> [B, 4, M] -> [B, M, 4]
            reg_pred_ = reg_pred_.view(B, 4, M).permute(0, 2, 1).contiguous()
            ## tlbr -> xyxy
            x1y1_pred = anchors[None] - reg_pred_[..., :2] * self.strides[level]
            x2y2_pred = anchors[None] + reg_pred_[..., 2:] * self.strides[level]
            box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

            all_obj_preds.append(obj_pred)
            all_cls_preds.append(cls_pred)
            all_reg_preds.append(reg_pred)
            all_box_preds.append(box_pred)
            all_anchors.append(anchors)
            all_strides.append(stride_tensor)
        
        # output dict
        outputs = {"pred_obj": all_obj_preds,        # List(Tensor) [B, M, 1]
                   "pred_cls": all_cls_preds,        # List(Tensor) [B, M, C]
                   "pred_reg": all_reg_preds,        # List(Tensor) [B, M, 4*(reg_max)]
                   "pred_box": all_box_preds,        # List(Tensor) [B, M, 4]
                   "anchors": all_anchors,           # List(Tensor) [M, 2]
                   "strides": self.strides,          # List(Int) = [8, 16, 32]
                   "stride_tensor": all_strides      # List(Tensor) [M, 1]
                   }

        return outputs
    

# build detection head
def build_pred_layer(cls_dim, reg_dim, strides, num_classes, num_coords=4, num_levels=3):
    pred_layers = MultiLevelPredLayer(cls_dim, reg_dim, strides, num_classes, num_coords, num_levels) 

    return pred_layers