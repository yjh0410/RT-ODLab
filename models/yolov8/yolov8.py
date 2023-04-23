import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolov8_backbone import build_backbone
from .yolov8_neck import build_neck
from .yolov8_pafpn import build_fpn
from .yolov8_head import build_head

from utils.nms import multiclass_nms


# Anchor-free YOLO
class YOLOv8(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes = 20, 
                 conf_thresh = 0.05,
                 nms_thresh = 0.6,
                 trainable = False, 
                 topk = 1000):
        super(YOLOv8, self).__init__()
        # --------- Basic Parameters ----------
        self.cfg = cfg
        self.device = device
        self.stride = cfg['stride']
        self.reg_max = cfg['reg_max']
        self.use_dfl = cfg['reg_max'] > 1
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk
        
        # --------- Network Parameters ----------
        self.proj_conv = nn.Conv2d(self.reg_max, 1, kernel_size=1, bias=False)

        ## backbone
        self.backbone, feats_dim = build_backbone(cfg, cfg['pretrained']*trainable)

        ## neck
        self.neck = build_neck(cfg=cfg, in_dim=feats_dim[-1], out_dim=feats_dim[-1])
        feats_dim[-1] = self.neck.out_dim
        
        ## fpn
        self.fpn = build_fpn(cfg=cfg, in_dims=feats_dim)
        fpn_dims = self.fpn.out_dim

        ## non-shared heads
        self.non_shared_heads = nn.ModuleList(
            [build_head(cfg, feat_dim, fpn_dims, num_classes) 
            for feat_dim in fpn_dims
            ])

        ## pred
        self.cls_preds = nn.ModuleList(
                            [nn.Conv2d(head.cls_out_dim, self.num_classes, kernel_size=1) 
                                for head in self.non_shared_heads
                              ]) 
        self.reg_preds = nn.ModuleList(
                            [nn.Conv2d(head.reg_out_dim, 4*(cfg['reg_max']), kernel_size=1) 
                                for head in self.non_shared_heads
                              ])                 

        # --------- Network Initialization ----------
        # init bias
        self.init_yolo()


    def init_yolo(self): 
        # Init yolo
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03    
        # Init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        # cls pred
        for cls_pred in self.cls_preds:
            b = cls_pred.bias.view(1, -1)
            b.data.fill_(bias_value.item())
            cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        for reg_pred in self.reg_preds:
            b = reg_pred.bias.view(-1, )
            b.data.fill_(1.0)
            reg_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = reg_pred.weight
            w.data.fill_(0.)
            reg_pred.weight = torch.nn.Parameter(w, requires_grad=True)

        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max), requires_grad=False)
        self.proj_conv.weight = nn.Parameter(self.proj.view([1, self.reg_max, 1, 1]).clone().detach(),
                                                   requires_grad=False)


    def generate_anchors(self, level, fmp_size):
        """
            fmp_size: (List) [H, W]
        """
        # generate grid cells
        fmp_h, fmp_w = fmp_size
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2) + 0.5
        anchor_xy *= self.stride[level]
        anchors = anchor_xy.to(self.device)

        return anchors
        

    def decode_boxes(self, anchors, pred_regs, stride):
        """
        Input:
            anchors:  (List[Tensor]) [1, M, 2]
            pred_reg: (List[Tensor]) [B, M, 4*(reg_max)]
        Output:
            pred_box: (Tensor) [B, M, 4]
        """
        if self.use_dfl:
            B, M = pred_regs.shape[:2]
            # [B, M, 4*(reg_max)] -> [B, M, 4, reg_max] -> [B, 4, M, reg_max]
            pred_regs = pred_regs.reshape([B, M, 4, self.reg_max])
            # [B, M, 4, reg_max] -> [B, reg_max, 4, M]
            pred_regs = pred_regs.permute(0, 3, 2, 1).contiguous()
            # [B, reg_max, 4, M] -> [B, 1, 4, M]
            pred_regs = self.proj_conv(F.softmax(pred_regs, dim=1))
            # [B, 1, 4, M] -> [B, 4, M] -> [B, M, 4]
            pred_regs = pred_regs.view(B, 4, M).permute(0, 2, 1).contiguous()

        # tlbr -> xyxy
        pred_x1y1 = anchors - pred_regs[..., :2] * stride
        pred_x2y2 = anchors + pred_regs[..., 2:] * stride
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box


    def post_process(self, cls_preds, reg_preds, anchors):
        """
        Input:
            cls_preds: List(Tensor) [[B, H x W, C], ...]
            reg_preds: List(Tensor) [[B, H x W, 4*(reg_max)], ...]
            anchors:   List(Tensor) [[H x W, 2], ...]
        """
        all_scores = []
        all_labels = []
        all_bboxes = []
        
        for level, (cls_pred_i, reg_pred_i, anchors_i) in enumerate(zip(cls_preds, reg_preds, anchors)):
            # [B, M, C] -> [M, C]
            cur_cls_pred_i = cls_pred_i[0]
            cur_reg_pred_i = reg_pred_i[0]
            # [MC,]
            scores_i = cur_cls_pred_i.sigmoid().flatten()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk, cur_reg_pred_i.size(0))

            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = scores_i.sort(descending=True)
            scores = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
            labels = topk_idxs % self.num_classes

            cur_reg_pred_i = cur_reg_pred_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]

            # decode box: [M, 4]
            box_pred_i = self.decode_boxes(
                anchors_i[None], cur_reg_pred_i[None], self.stride[level])
            bboxes = box_pred_i[0]

            all_scores.append(scores)
            all_labels.append(labels)
            all_bboxes.append(bboxes)

        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        bboxes = torch.cat(all_bboxes)

        # threshold
        keep_idxs = scores.gt(self.conf_thresh)
        scores = scores[keep_idxs]
        labels = labels[keep_idxs]
        bboxes = bboxes[keep_idxs]

        # to cpu & numpy
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, False)

        return bboxes, scores, labels


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
        all_reg_preds = []
        all_anchors = []
        for level, (feat, head) in enumerate(zip(pyramid_feats, self.non_shared_heads)):
            cls_feat, reg_feat = head(feat)

            # pred
            cls_pred = self.cls_preds[level](cls_feat)  # [B, C, H, W]
            reg_pred = self.reg_preds[level](reg_feat)  # [B, 4*(reg_max), H, W]

            B, _, H, W = cls_pred.size()
            fmp_size = [H, W]
            # [M, 2]
            anchors = self.generate_anchors(level, fmp_size)

            # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4*self.reg_max)

            all_cls_preds.append(cls_pred)
            all_reg_preds.append(reg_pred)
            all_anchors.append(anchors)

        # post process
        bboxes, scores, labels = self.post_process(
            all_cls_preds, all_reg_preds, all_anchors)
        
        return bboxes, scores, labels


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
            all_anchors = []
            all_cls_preds = []
            all_reg_preds = []
            all_box_preds = []
            all_strides = []
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
                reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4*self.reg_max)

                # decode box: [B, M, 4]
                box_pred = self.decode_boxes(anchors, reg_pred, self.stride[level])

                # stride tensor: [M, 1]
                stride_tensor = torch.ones_like(anchors[..., :1]) * self.stride[level]

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