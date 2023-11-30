import math
import torch
import torch.nn as nn


# Single-level pred layer
class SingleLevelPredLayer(nn.Module):
    def __init__(self,
                 cls_dim     :int = 256,
                 reg_dim     :int = 256,
                 stride      :int = 32,
                 num_classes :int = 80,
                 num_coords  :int = 4):
        super().__init__()
        # --------- Basic Parameters ----------
        self.stride = stride
        self.cls_dim = cls_dim
        self.reg_dim = reg_dim
        self.num_classes = num_classes
        self.num_coords = num_coords

        # --------- Network Parameters ----------
        self.cls_pred = nn.Conv2d(cls_dim, num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(reg_dim, num_coords, kernel_size=1)                

        self.init_bias()
        
    def init_bias(self):
        # cls pred bias
        b = self.cls_pred.bias.view(1, -1)
        b.data.fill_(math.log(5 / self.num_classes / (640. / self.stride) ** 2))
        self.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        # reg pred bias
        b = self.reg_pred.bias.view(-1, )
        b.data.fill_(1.0)
        self.reg_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def generate_anchors(self, fmp_size):
        """
            fmp_size: (List) [H, W]
        """
        # generate grid cells
        fmp_h, fmp_w = fmp_size
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        anchors = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2)
        anchors += 0.5  # add center offset
        anchors *= self.stride

        return anchors
        
    def forward(self, cls_feat, reg_feat):
        # pred
        cls_pred = self.cls_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)

        # generate anchor boxes: [M, 4]
        B, _, H, W = cls_pred.size()
        fmp_size = [H, W]
        anchors = self.generate_anchors(fmp_size)
        anchors = anchors.to(cls_pred.device)
        # stride tensor: [M, 1]
        stride_tensor = torch.ones_like(anchors[..., :1]) * self.stride
        
        # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)

        # ---------------- Decode bbox ----------------
        ctr_pred = reg_pred[..., :2] * self.stride + anchors[..., :2]
        wh_pred = torch.exp(reg_pred[..., 2:]) * self.stride
        pred_x1y1 = ctr_pred - wh_pred * 0.5
        pred_x2y2 = ctr_pred + wh_pred * 0.5
        box_pred = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        # output dict
        outputs = {"pred_cls": cls_pred,             # (Tensor) [B, M, C]
                   "pred_reg": reg_pred,             # (Tensor) [B, M, 4]
                   "pred_box": box_pred,             # (Tensor) [B, M, 4] 
                   "anchors": anchors,               # (Tensor) [M, 2]
                   "stride": self.stride,            # (Int)
                   "stride_tensors": stride_tensor   # List(Tensor) [M, 1]
                   }

        return outputs

# Multi-level pred layer
class MultiLevelPredLayer(nn.Module):
    def __init__(self,
                 cls_dim,
                 reg_dim,
                 strides,
                 num_classes :int = 80,
                 num_coords  :int = 4,
                 num_levels  :int = 3):
        super().__init__()
        # --------- Basic Parameters ----------
        self.cls_dim = cls_dim
        self.reg_dim = reg_dim
        self.strides = strides
        self.num_classes = num_classes
        self.num_coords = num_coords
        self.num_levels = num_levels

        # ----------- Network Parameters -----------
        ## multi-level pred layers
        self.multi_level_preds = nn.ModuleList(
            [SingleLevelPredLayer(cls_dim     = cls_dim,
                                  reg_dim     = reg_dim,
                                  stride      = strides[level],
                                  num_classes = num_classes,
                                  num_coords  = num_coords)
                                  for level in range(num_levels)
                                  ])
        
    def forward(self, cls_feats, reg_feats):
        all_anchors = []
        all_strides = []
        all_cls_preds = []
        all_box_preds = []
        all_reg_preds = []
        for level in range(self.num_levels):
            # ---------------- Single level prediction ----------------
            outputs = self.multi_level_preds[level](cls_feats[level], reg_feats[level])

            # collect results
            all_cls_preds.append(outputs["pred_cls"])
            all_box_preds.append(outputs["pred_box"])
            all_reg_preds.append(outputs["pred_reg"])
            all_anchors.append(outputs["anchors"])
            all_strides.append(outputs["stride_tensors"])
        
        # output dict
        outputs = {"pred_cls": all_cls_preds,      # List(Tensor) [B, M, C]
                   "pred_box": all_box_preds,      # List(Tensor) [B, M, 4]
                   "pred_reg": all_reg_preds,      # List(Tensor) [B, M, 4]
                   "anchors": all_anchors,         # List(Tensor) [M, 2]
                   "strides": self.strides,        # List(Int) [8, 16, 32]
                   "stride_tensors": all_strides   # List(Tensor) [M, 1]
                   }

        return outputs
    

# build detection head
def build_pred_layer(cls_dim, reg_dim, strides, num_classes, num_coords=4, num_levels=3):
    pred_layers = MultiLevelPredLayer(cls_dim     = cls_dim,
                                      reg_dim     = reg_dim,
                                      strides     = strides,
                                      num_classes = num_classes,
                                      num_coords  = num_coords,
                                      num_levels  = num_levels) 

    return pred_layers
