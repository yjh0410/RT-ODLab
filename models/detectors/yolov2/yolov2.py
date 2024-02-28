import torch
import torch.nn as nn

from utils.misc import multiclass_nms

from .yolov2_backbone import build_backbone
from .yolov2_neck import build_neck
from .yolov2_head import build_head


# YOLOv2
class YOLOv2(nn.Module):
    def __init__(self,
                 cfg,
                 device,
                 num_classes=20,
                 conf_thresh=0.01,
                 nms_thresh=0.5,
                 topk=100,
                 trainable=False,
                 deploy=False,
                 no_multi_labels=False,
                 nms_class_agnostic=False):
        super(YOLOv2, self).__init__()
        # ------------------- Basic parameters -------------------
        self.cfg = cfg                                 # 模型配置文件
        self.device = device                           # cuda或者是cpu
        self.num_classes = num_classes                 # 类别的数量
        self.trainable = trainable                     # 训练的标记
        self.conf_thresh = conf_thresh                 # 得分阈值
        self.nms_thresh = nms_thresh                   # NMS阈值
        self.topk_candidates = topk                    # topk
        self.stride = 32                               # 网络的最大步长
        self.deploy = deploy
        self.no_multi_labels = no_multi_labels
        self.nms_class_agnostic = nms_class_agnostic
        # ------------------- Anchor box -------------------
        self.anchor_size = torch.as_tensor(cfg['anchor_size']).float().view(-1, 2) # [A, 2]
        self.num_anchors = self.anchor_size.shape[0]
        
        # ------------------- Network Structure -------------------
        ## 主干网络
        self.backbone, feat_dim = build_backbone(
            cfg['backbone'], trainable&cfg['pretrained'])

        ## 颈部网络
        self.neck = build_neck(cfg, feat_dim, out_dim=512)
        head_dim = self.neck.out_dim

        ## 检测头
        self.head = build_head(cfg, head_dim, head_dim, num_classes)

        ## 预测层
        self.obj_pred = nn.Conv2d(head_dim, 1*self.num_anchors, kernel_size=1)
        self.cls_pred = nn.Conv2d(head_dim, num_classes*self.num_anchors, kernel_size=1)
        self.reg_pred = nn.Conv2d(head_dim, 4*self.num_anchors, kernel_size=1)
    

        if self.trainable:
            self.init_bias()


    def init_bias(self):
        # init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.obj_pred.bias, bias_value)
        nn.init.constant_(self.cls_pred.bias, bias_value)


    def generate_anchors(self, fmp_size):
        """
            fmp_size: (List) [H, W]
        """
        fmp_h, fmp_w = fmp_size

        # generate grid cells
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2)
        # [HW, 2] -> [HW, A, 2] -> [M, 2]
        anchor_xy = anchor_xy.unsqueeze(1).repeat(1, self.num_anchors, 1)
        anchor_xy = anchor_xy.view(-1, 2).to(self.device)

        # [A, 2] -> [1, A, 2] -> [HW, A, 2] -> [M, 2]
        anchor_wh = self.anchor_size.unsqueeze(0).repeat(fmp_h*fmp_w, 1, 1)
        anchor_wh = anchor_wh.view(-1, 2).to(self.device)

        anchors = torch.cat([anchor_xy, anchor_wh], dim=-1)

        return anchors
        

    def decode_boxes(self, anchors, reg_pred):
        """
            将txtytwth转换为常用的x1y1x2y2形式。
        """

        # 计算预测边界框的中心点坐标和宽高
        pred_ctr = (torch.sigmoid(reg_pred[..., :2]) + anchors[..., :2]) * self.stride
        pred_wh = torch.exp(reg_pred[..., 2:]) * anchors[..., 2:]

        # 将所有bbox的中心带你坐标和宽高换算成x1y1x2y2形式
        pred_x1y1 = pred_ctr - pred_wh * 0.5
        pred_x2y2 = pred_ctr + pred_wh * 0.5
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box


    def postprocess(self, obj_pred, cls_pred, reg_pred, anchors):
        """
        Input:
            obj_pred: (Tensor) [H*W*A, 1]
            cls_pred: (Tensor) [H*W*A, C]
            reg_pred: (Tensor) [H*W*A, 4]
        """
        if self.no_multi_labels:
            # [M,]
            scores, labels = torch.max(torch.sqrt(obj_pred.sigmoid() * cls_pred.sigmoid()), dim=1)

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, reg_pred.size(0))

            # topk candidates
            predicted_prob, topk_idxs = scores.sort(descending=True)
            topk_scores = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = topk_scores > self.conf_thresh
            scores = topk_scores[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            labels = labels[topk_idxs]
            bboxes = self.decode_boxes(anchors[topk_idxs], reg_pred[topk_idxs])
        else:
        # (H x W x A x C,)
            scores = torch.sqrt(obj_pred.sigmoid() * cls_pred.sigmoid()).flatten()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, reg_pred.size(0))

            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = scores.sort(descending=True)
            topk_scores = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = topk_scores > self.conf_thresh
            scores = topk_scores[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
            labels = topk_idxs % self.num_classes

            reg_pred = reg_pred[anchor_idxs]
            anchors = anchors[anchor_idxs]

            # 解算边界框, 并归一化边界框: [H*W*A, 4]
            bboxes = self.decode_boxes(anchors, reg_pred)

        # to cpu & numpy
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, self.nms_class_agnostic)

        return bboxes, scores, labels


    @torch.no_grad()
    def inference(self, x):
        bs = x.shape[0]
        # 主干网络
        feat = self.backbone(x)

        # 颈部网络
        feat = self.neck(feat)

        # 检测头
        cls_feat, reg_feat = self.head(feat)

        # 预测层
        obj_pred = self.obj_pred(reg_feat)
        cls_pred = self.cls_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)
        fmp_size = obj_pred.shape[-2:]

        # anchors: [M, 2]
        anchors = self.generate_anchors(fmp_size)

        # 对 pred 的size做一些view调整，便于后续的处理
        # [B, A*C, H, W] -> [B, H, W, A*C] -> [B, H*W*A, C]
        obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, 1)
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, self.num_classes)
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, 4)

        # 测试时，笔者默认batch是1，
        # 因此，我们不需要用batch这个维度，用[0]将其取走。
        obj_pred = obj_pred[0]       # [H*W*A, 1]
        cls_pred = cls_pred[0]       # [H*W*A, NC]
        reg_pred = reg_pred[0]       # [H*W*A, 4]

        if self.deploy:
            scores = torch.sqrt(obj_pred.sigmoid() * cls_pred.sigmoid())
            bboxes = self.decode_boxes(anchors, reg_pred)
            # [n_anchors_all, 4 + C]
            outputs = torch.cat([bboxes, scores], dim=-1)

        else:
            # post process
            bboxes, scores, labels = self.postprocess(
                obj_pred, cls_pred, reg_pred, anchors)

            outputs = {
                "scores": scores,
                "labels": labels,
                "bboxes": bboxes
            }

        return outputs


    def forward(self, x):
        if not self.trainable:
            return self.inference(x)
        else:
            bs = x.shape[0]
            # 主干网络
            feat = self.backbone(x)

            # 颈部网络
            feat = self.neck(feat)

            # 检测头
            cls_feat, reg_feat = self.head(feat)

            # 预测层
            obj_pred = self.obj_pred(reg_feat)
            cls_pred = self.cls_pred(cls_feat)
            reg_pred = self.reg_pred(reg_feat)
            fmp_size = obj_pred.shape[-2:]

            # anchors: [M, 2]
            anchors = self.generate_anchors(fmp_size)

            # 对 pred 的size做一些view调整，便于后续的处理
            # [B, A*C, H, W] -> [B, H, W, A*C] -> [B, H*W*A, C]
            obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, 1)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, self.num_classes)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, 4)

            # decode bbox
            box_pred = self.decode_boxes(anchors, reg_pred)

            # 网络输出
            outputs = {"pred_obj": obj_pred,                   # (Tensor) [B, M, 1]
                       "pred_cls": cls_pred,                   # (Tensor) [B, M, C]
                       "pred_box": box_pred,                   # (Tensor) [B, M, 4]
                       "stride": self.stride,                  # (Int)
                       "fmp_size": fmp_size                    # (List) [fmp_h, fmp_w]
                       }           
            return outputs
        