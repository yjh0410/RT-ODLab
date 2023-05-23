import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolov7_plus_backbone import build_backbone
from .yolov7_plus_neck import build_neck
from .yolov7_plus_pafpn import build_fpn
from .yolov7_plus_head import build_head

from utils.misc import multiclass_nms


# Anchor-free YOLO
class YOLOv7_Plus(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes = 20, 
                 conf_thresh = 0.05,
                 nms_thresh = 0.6,
                 trainable = False, 
                 topk = 1000):
        super(YOLOv7_Plus, self).__init__()
        # --------- Basic Parameters ----------
        self.cfg = cfg
        self.device = device
        self.stride = cfg['stride']
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk
        
        # ------------------- Anchor box -------------------
        self.num_levels = 3
        self.num_anchors = len(cfg['anchor_size']) // self.num_levels
        self.anchor_size = torch.as_tensor(
            cfg['anchor_size']
            ).view(self.num_levels, self.num_anchors, 2) # [S, A, 2]
        
        # ------------------- Network Structure -------------------
        ## backbone
        self.backbone, feats_dim = build_backbone(cfg, cfg['pretrained']*trainable)

        ## neck
        self.neck = build_neck(cfg=cfg, in_dim=feats_dim[-1], out_dim=feats_dim[-1])
        feats_dim[-1] = self.neck.out_dim
        
        ## fpn
        self.fpn = build_fpn(cfg=cfg, in_dims=feats_dim, out_dim=round(256*cfg['width']))
        self.head_dim = self.fpn.out_dim

        ## non-shared heads
        self.non_shared_heads = nn.ModuleList(
            [build_head(cfg, head_dim, head_dim, num_classes) 
            for head_dim in self.head_dim
            ])

        ## pred
        self.cls_preds = nn.ModuleList(
                            [nn.Conv2d(head.cls_out_dim, self.num_classes * self.num_anchors, kernel_size=1) 
                                for head in self.non_shared_heads
                              ]) 
        self.reg_preds = nn.ModuleList(
                            [nn.Conv2d(head.reg_out_dim, 4 * self.num_anchors, kernel_size=1) 
                                for head in self.non_shared_heads
                              ])                 


    # ---------------------- Basic Functions ----------------------
    ## generate anchor points
    def generate_anchors(self, level, fmp_size):
        fmp_h, fmp_w = fmp_size
        # [KA, 2]
        anchor_size = self.anchor_size[level]

        # generate grid cells
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2)
        # [HW, 2] -> [HW, KA, 2] -> [M, 2]
        anchor_xy = anchor_xy.unsqueeze(1).repeat(1, self.num_anchors, 1)
        anchor_xy = anchor_xy.view(-1, 2).to(self.device)
        anchor_xy += 0.5
        anchor_xy *= self.stride[level]

        # [KA, 2] -> [1, KA, 2] -> [HW, KA, 2] -> [M, 2]
        anchor_wh = anchor_size.unsqueeze(0).repeat(fmp_h*fmp_w, 1, 1)
        anchor_wh = anchor_wh.view(-1, 2).to(self.device)

        anchors = torch.cat([anchor_xy, anchor_wh], dim=-1)

        return anchors
        
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
            topk_scores = topk_scores[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
            topk_labels = topk_idxs % self.num_classes
            topk_bboxes = box_pred_i[anchor_idxs]

            all_scores.append(topk_scores)
            all_labels.append(topk_labels)
            all_bboxes.append(topk_bboxes)

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
        # backbone
        pyramid_feats = self.backbone(x)

        # neck
        pyramid_feats[-1] = self.neck(pyramid_feats[-1])

        # fpn
        pyramid_feats = self.fpn(pyramid_feats)

        # non-shared heads
        all_cls_preds = []
        all_box_preds = []
        all_anchors = []
        for level, (feat, head) in enumerate(zip(pyramid_feats, self.non_shared_heads)):
            cls_feat, reg_feat = head(feat)

            # pred
            cls_pred = self.cls_preds[level](cls_feat)  # [B, C*A, H, W]
            reg_pred = self.reg_preds[level](reg_feat)  # [B, 4*A, H, W]

            B, _, H, W = cls_pred.size()
            fmp_size = [H, W]
            # [M, 2]
            anchors = self.generate_anchors(level, fmp_size)

            # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
            cls_pred = cls_pred[0].permute(1, 2, 0).contiguous().view(-1, self.num_classes)
            reg_pred = reg_pred[0].permute(1, 2, 0).contiguous().view(-1, 4)

            # decode bbox
            ctr_pred = anchors[..., :2] + reg_pred[..., :2] * self.stride[level]
            wh_pred = torch.exp(reg_pred[..., 2:]) * anchors[..., 2:]
            pred_x1y1 = ctr_pred - wh_pred * 0.5
            pred_x2y2 = ctr_pred + wh_pred * 0.5
            box_pred = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

            all_cls_preds.append(cls_pred)
            all_box_preds.append(box_pred)
            all_anchors.append(anchors)

        # post process
        bboxes, scores, labels = self.post_process(
            all_cls_preds, all_box_preds)
        
        return bboxes, scores, labels


    # ---------------------- Main Process for Training ----------------------
    def forward(self, x):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            # backbone
            pyramid_feats = self.backbone(x)

            # neck
            pyramid_feats[-1] = self.neck(pyramid_feats[-1])

            # fpn
            pyramid_feats = self.fpn(pyramid_feats)

            # non-shared heads
            all_fmp_sizes = []
            all_cls_preds = []
            all_box_preds = []
            all_anchors = []
            for level, (feat, head) in enumerate(zip(pyramid_feats, self.non_shared_heads)):
                cls_feat, reg_feat = head(feat)

                # pred
                cls_pred = self.cls_preds[level](cls_feat)  # [B, C, H, W]
                reg_pred = self.reg_preds[level](reg_feat)  # [B, 4*(reg_max), H, W]

                B, _, H, W = cls_pred.size()
                fmp_size = [H, W]
                # generate anchor boxes: [M, 2]
                anchors = self.generate_anchors(level, fmp_size)
                
                # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
                cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
                reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)

                # decode bbox
                ctr_pred = anchors[..., :2] + reg_pred[..., :2] * self.stride[level]
                wh_pred = torch.exp(reg_pred[..., 2:]) * anchors[..., 2:]
                pred_x1y1 = ctr_pred - wh_pred * 0.5
                pred_x2y2 = ctr_pred + wh_pred * 0.5
                box_pred = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

                all_fmp_sizes.append(fmp_size)
                all_cls_preds.append(cls_pred)
                all_box_preds.append(box_pred)
                all_anchors.append(anchors)
            
            # output dict
            outputs = {"pred_cls": all_cls_preds,        # List(Tensor) [B, M, C]
                       "pred_box": all_box_preds,        # List(Tensor) [B, M, 4]
                       "anchors": all_anchors,           # List(Tensor) [B, M, 4]
                       'fmp_sizes': all_fmp_sizes,       # List
                       'strides': self.stride,           # List
                       }

            return outputs 
