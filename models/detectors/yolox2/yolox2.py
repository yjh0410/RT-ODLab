# --------------- Torch components ---------------
import torch
import torch.nn as nn

# --------------- Model components ---------------
from .yolox2_backbone import build_backbone
from .yolox2_neck import build_neck
from .yolox2_pafpn import build_fpn
from .yolox2_head import build_head

# --------------- External components ---------------
from utils.misc import multiclass_nms


# YOLOX-2
class YOLOX2(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes = 20, 
                 conf_thresh = 0.05,
                 nms_thresh = 0.6,
                 trainable = False, 
                 topk = 1000,
                 deploy = False):
        super(YOLOX2, self).__init__()
        # ---------------------- Basic Parameters ----------------------
        self.cfg = cfg
        self.device = device
        self.stride = cfg['stride']
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk
        self.deploy = deploy
        
        # ---------------------- Network Parameters ----------------------
        ## ----------- Backbone -----------
        self.backbone, feats_dim = build_backbone(cfg, trainable&cfg['pretrained'])

        ## ----------- Neck: SPP -----------
        self.neck = build_neck(cfg=cfg, in_dim=feats_dim[-1], out_dim=feats_dim[-1])
        feats_dim[-1] = self.neck.out_dim
        
        ## ----------- Neck: FPN -----------
        self.fpn = build_fpn(cfg=cfg, in_dims=feats_dim, out_dim=round(256*cfg['width']))
        self.head_dim = self.fpn.out_dim

        ## ----------- Heads -----------
        self.det_heads = nn.ModuleList(
            [build_head(cfg, head_dim, head_dim, num_classes) 
            for head_dim in self.head_dim
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
        
    ## post-process
    def post_process(self, obj_preds, cls_preds, box_preds):
        """
        Input:
            obj_preds: List(Tensor) [[H x W, 1], ...]
            cls_preds: List(Tensor) [[H x W, C], ...]
            box_preds: List(Tensor) [[H x W, 4], ...]
            anchors:   List(Tensor) [[H x W, 2], ...]
        """
        all_scores = []
        all_labels = []
        all_bboxes = []
        
        for obj_pred_i, cls_pred_i, box_pred_i in zip(obj_preds, cls_preds, box_preds):
            # (H x W x KA x C,)
            scores_i = (torch.sqrt(obj_pred_i.sigmoid() * cls_pred_i.sigmoid())).flatten()

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
        all_obj_preds = []
        all_cls_preds = []
        all_box_preds = []
        for level, (feat, head) in enumerate(zip(pyramid_feats, self.det_heads)):
            # ---------------- Pred ----------------
            obj_pred, cls_pred, reg_pred = head(feat)

            # anchors: [M, 2]
            fmp_size = cls_pred.shape[-2:]
            anchors = self.generate_anchors(level, fmp_size)

            # [1, C, H, W] -> [H, W, C] -> [M, C]
            obj_pred = obj_pred[0].permute(1, 2, 0).contiguous().view(-1, 1)
            cls_pred = cls_pred[0].permute(1, 2, 0).contiguous().view(-1, self.num_classes)
            reg_pred = reg_pred[0].permute(1, 2, 0).contiguous().view(-1, 4)

            # decode bbox
            ctr_pred = reg_pred[..., :2] * self.stride[level] + anchors[..., :2]
            wh_pred = torch.exp(reg_pred[..., 2:]) * self.stride[level]
            pred_x1y1 = ctr_pred - wh_pred * 0.5
            pred_x2y2 = ctr_pred + wh_pred * 0.5
            box_pred = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

            all_obj_preds.append(obj_pred)
            all_cls_preds.append(cls_pred)
            all_box_preds.append(box_pred)

        if self.deploy:
            obj_preds = torch.cat(all_obj_preds, dim=0)
            cls_preds = torch.cat(all_cls_preds, dim=0)
            box_preds = torch.cat(all_box_preds, dim=0)
            scores = torch.sqrt(obj_preds.sigmoid() * cls_preds.sigmoid())
            bboxes = box_preds
            # [n_anchors_all, 4 + C]
            outputs = torch.cat([bboxes, scores], dim=-1)

            return outputs
        else:
            # post process
            bboxes, scores, labels = self.post_process(
                all_obj_preds, all_cls_preds, all_box_preds)
        
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
            all_anchors = []
            all_obj_preds = []
            all_cls_preds = []
            all_box_preds = []
            all_strides = []
            for level, (feat, head) in enumerate(zip(pyramid_feats, self.det_heads)):
                # ---------------- Pred ----------------
                obj_pred, cls_pred, reg_pred = head(feat)

                B, _, H, W = cls_pred.size()
                fmp_size = [H, W]
                # generate anchor boxes: [M, 4]
                anchors = self.generate_anchors(level, fmp_size)
                
                # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
                obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
                cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
                reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)

                # decode bbox
                ctr_pred = reg_pred[..., :2] * self.stride[level] + anchors[..., :2]
                wh_pred = torch.exp(reg_pred[..., 2:]) * self.stride[level]
                pred_x1y1 = ctr_pred - wh_pred * 0.5
                pred_x2y2 = ctr_pred + wh_pred * 0.5
                box_pred = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

                all_obj_preds.append(obj_pred)
                all_cls_preds.append(cls_pred)
                all_box_preds.append(box_pred)
                all_anchors.append(anchors)
            
            # output dict
            outputs = {"pred_obj": all_obj_preds,        # List(Tensor) [B, M, 1]
                       "pred_cls": all_cls_preds,        # List(Tensor) [B, M, C]
                       "pred_box": all_box_preds,        # List(Tensor) [B, M, 4]
                       "anchors": all_anchors,           # List(Tensor) [B, M, 2]
                       'strides': self.stride}           # List(Int) [8, 16, 32]

            return outputs 
