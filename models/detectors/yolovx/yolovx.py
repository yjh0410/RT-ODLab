# --------------- Torch components ---------------
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------- Model components ---------------
from .yolovx_backbone import build_backbone
from .yolovx_neck import build_neck
from .yolovx_pafpn import build_fpn
from .yolovx_head import build_head

# --------------- External components ---------------
from utils.misc import multiclass_nms


# YOLOvx
class YOLOvx(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes = 20, 
                 conf_thresh = 0.05,
                 nms_thresh = 0.6,
                 trainable = False, 
                 topk = 1000,
                 deploy = False):
        super(YOLOvx, self).__init__()
        # ---------------------- Basic Parameters ----------------------
        self.cfg = cfg
        self.device = device
        self.stride = cfg['stride']
        self.reg_max = cfg['reg_max']
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk
        self.deploy = deploy
        self.head_dim = round(256*cfg['width'])
        
        # ---------------------- Network Parameters ----------------------
        ## ----------- proj_conv ------------
        self.proj = nn.Parameter(torch.linspace(0, cfg['reg_max'], cfg['reg_max']), requires_grad=False)
        self.proj_conv = nn.Conv2d(self.reg_max, 1, kernel_size=1, bias=False)
        self.proj_conv.weight = nn.Parameter(self.proj.view([1, cfg['reg_max'], 1, 1]).clone().detach(), requires_grad=False)

        ## ----------- Backbone -----------
        self.backbone, feats_dim = build_backbone(cfg, trainable&cfg['pretrained'])

        ## ----------- Neck: SPP -----------
        self.neck = build_neck(cfg=cfg, in_dim=feats_dim[-1], out_dim=feats_dim[-1])
        feats_dim[-1] = self.neck.out_dim
        
        ## ----------- Neck: FPN -----------
        self.fpn = build_fpn(cfg=cfg, in_dims=feats_dim, out_dim=round(256*cfg['width']))
        self.fpn_dims = self.fpn.out_dim

        ## ----------- Heads -----------
        self.heads = build_head(cfg, self.fpn_dims, self.head_dim, num_classes) 

        ## ----------- Preds -----------
        self.cls_preds = nn.ModuleList(
                            [nn.Conv2d(self.head_dim, num_classes, kernel_size=1) 
                                for _ in range(len(self.stride))
                              ]) 
        self.reg_preds = nn.ModuleList(
                            [nn.Conv2d(self.head_dim, 4*cfg['reg_max'], kernel_size=1) 
                                for _ in range(len(self.stride))
                              ])                 


    # ---------------------- Basic Functions ----------------------
    ## generate anchor points
    def generate_anchors(self, level, fmp_size):
        """
            fmp_size: (List) [H, W]
        """
        # generate grid cells
        fmp_h, fmp_w = fmp_size
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2)
        anchor_xy += 0.5  # add center offset
        anchor_xy *= self.stride[level]
        anchors = anchor_xy.to(self.device)

        return anchors
        
    ## decode bbox
    def decode_bbox(self, reg_pred, anchors, stride):
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
    
    ## post-process
    def post_process(self, cls_preds, box_preds):
        """
        Input:
            cls_preds: List(Tensor) [[H x W, C], ...]
            box_preds: List(Tensor) [[H x W, 4], ...]
        """
        all_scores = []
        all_labels = []
        all_bboxes = []
        
        for cls_pred_i, box_pred_i in zip(cls_preds, box_preds):
            # (H x W x C,)
            scores_i = cls_pred_i.sigmoid().flatten()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk, box_pred_i.size(0))

            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = scores_i.sort(descending=True)
            topk_scores = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = topk_scores > self.conf_thresh
            scores = topk_scores[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
            labels = topk_idxs % self.num_classes

            bboxes = box_pred_i[anchor_idxs]

            all_scores.append(scores)
            all_labels.append(labels)
            all_bboxes.append(bboxes)

        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        bboxes = torch.cat(all_bboxes)

        # to cpu & numpy
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, False)

        return bboxes, scores, labels

    
    # ---------------------- Main Process for Inference ----------------------
    @torch.no_grad()
    def inference_single_image(self, x):
        # ---------------- Backbone ----------------
        pyramid_feats = self.backbone(x)

        # ---------------- Neck: SPP ----------------
        pyramid_feats[-1] = self.neck(pyramid_feats[-1])

        # ---------------- Neck: PaFPN ----------------
        pyramid_feats = self.fpn(pyramid_feats)

        # ---------------- Heads ----------------
        cls_feats, reg_feats = self.heads(pyramid_feats)

        # ---------------- Preds ----------------
        all_cls_preds = []
        all_box_preds = []
        for level, (cls_feat, reg_feat) in enumerate(zip(cls_feats, reg_feats)):
            # prediction
            cls_pred = self.cls_preds[level](cls_feat)
            reg_pred = self.reg_preds[level](reg_feat)
            
            # anchors: [M, 2]
            B, _, H, W = cls_feat.size()
            anchors = self.generate_anchors(level, [H, W])
            
            # process preds
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4*self.reg_max)
            box_pred = self.decode_bbox(reg_pred, anchors, self.stride[level])

            # collect preds
            all_cls_preds.append(cls_pred[0])
            all_box_preds.append(box_pred[0])

        if self.deploy:
            # no post process
            cls_preds = torch.cat(all_cls_preds, dim=0)
            box_pred = torch.cat(all_box_preds, dim=0)
            # [n_anchors_all, 4 + C]
            outputs = torch.cat([box_pred, cls_preds.sigmoid()], dim=-1)

            return outputs

        else:
            # post process
            bboxes, scores, labels = self.post_process(all_cls_preds, all_box_preds)
            
            return bboxes, scores, labels


    # ---------------------- Main Process for Training ----------------------
    def forward(self, x):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            # ---------------- Backbone ----------------
            pyramid_feats = self.backbone(x)

            # ---------------- Neck: SPP ----------------
            pyramid_feats[-1] = self.neck(pyramid_feats[-1])

            # ---------------- Neck: PaFPN ----------------
            pyramid_feats = self.fpn(pyramid_feats)

            # ---------------- Heads ----------------
            cls_feats, reg_feats = self.heads(pyramid_feats)

            # ---------------- Preds ----------------
            all_anchors = []
            all_strides = []
            all_cls_preds = []
            all_reg_preds = []
            all_box_preds = []
            for level, (cls_feat, reg_feat) in enumerate(zip(cls_feats, reg_feats)):
                # anchors & stride tensor
                B, _, H, W = cls_feat.size()
                anchors = self.generate_anchors(level, [H, W])                         # [M, 4]
                stride_tensor = torch.ones_like(anchors[..., :1]) * self.stride[level] # [M, 1]
                
                # prediction
                cls_pred = self.cls_preds[level](cls_feat)
                reg_pred = self.reg_preds[level](reg_feat)

                # process preds
                cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
                reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4*self.reg_max)
                box_pred = self.decode_bbox(reg_pred, anchors, self.stride[level])

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
                       "strides": self.stride,           # List(Int) = [8, 16, 32]
                       "stride_tensor": all_strides      # List(Tensor) [M, 1]
                       }
            
            return outputs 
