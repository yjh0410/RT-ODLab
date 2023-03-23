import torch
import torch.nn as nn
import numpy as np

from .yolov1_basic import Conv
from .yolov1_neck import SPP
from .yolov1_backbone import build_resnet


# YOLOv1
class YOLOv1(nn.Module):
    def __init__(self,
                 device,
                 img_size=None,
                 num_classes=20,
                 conf_thresh=0.01,
                 nms_thresh=0.5,
                 trainable=False):
        super(YOLOv1, self).__init__()
        # ------------------- Basic parameters -------------------
        self.img_size = img_size                       # 输入图像大小
        self.device = device                           # cuda或者是cpu
        self.num_classes = num_classes                 # 类别的数量
        self.trainable = trainable                     # 训练的标记
        self.conf_thresh = conf_thresh                 # 得分阈值
        self.nms_thresh = nms_thresh                   # NMS阈值
        self.stride = 32                               # 网络的最大步长
        
        # ------------------- Network Structure -------------------
        ## backbone: resnet18
        self.backbone, feat_dim = build_resnet('resnet18', pretrained=trainable)

        ## neck: SPP
        self.neck = nn.Sequential(
            SPP(),
            Conv(feat_dim*4, feat_dim, k=1),
        )

        ## head
        self.convsets = nn.Sequential(
            Conv(feat_dim, feat_dim//2, k=1),
            Conv(feat_dim//2, feat_dim, k=3, p=1),
            Conv(feat_dim, feat_dim//2, k=1),
            Conv(feat_dim//2, feat_dim, k=3, p=1)
        )

        ## pred
        self.pred = nn.Conv2d(feat_dim, 1 + self.num_classes + 4, 1)
    

        if self.trainable:
            self.init_bias()


    def init_bias(self):
        # init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.pred.bias[..., :1], bias_value)
        nn.init.constant_(self.pred.bias[..., 1:1+self.num_classes], bias_value)


    def create_grid(self, fmp_size):
        """ 
            用于生成G矩阵，其中每个元素都是特征图上的像素坐标。
        """
        # 特征图的宽和高
        ws, hs = fmp_size

        # 生成网格的x坐标和y坐标
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])

        # 将xy两部分的坐标拼起来：[H, W, 2]
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()

        # [H, W, 2] -> [HW, 2] -> [HW, 2]
        grid_xy = grid_xy.view(-1, 2).to(self.device)
        
        return grid_xy


    def set_grid(self, img_size):
        """
            用于重置G矩阵。
        """
        self.img_size = img_size
        self.grid_cell = self.create_grid(img_size)


    def decode_boxes(self, pred, fmp_size):
        """
            将txtytwth转换为常用的x1y1x2y2形式。
        """
        # 生成网格坐标矩阵
        grid_cell = self.create_grid(fmp_size)

        # 计算预测边界框的中心点坐标和宽高
        pred[..., :2] = torch.sigmoid(pred[..., :2]) + grid_cell
        pred[..., 2:] = torch.exp(pred[..., 2:])

        # 将所有bbox的中心带你坐标和宽高换算成x1y1x2y2形式
        output = torch.zeros_like(pred)
        output[..., :2] = pred[..., :2] * self.stride - pred[..., 2:] * 0.5
        output[..., 2:] = pred[..., :2] * self.stride + pred[..., 2:] * 0.5
        
        return output


    def nms(self, bboxes, scores):
        """"Pure Python NMS baseline."""
        x1 = bboxes[:, 0]  #xmin
        y1 = bboxes[:, 1]  #ymin
        x2 = bboxes[:, 2]  #xmax
        y2 = bboxes[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []                                             
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # 计算交集的左上角点和右下角点的坐标
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # 计算交集的宽高
            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            # 计算交集的面积
            inter = w * h

            # 计算交并比
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # 滤除超过nms阈值的检测框
            inds = np.where(iou <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def postprocess(self, bboxes, scores):
        """
        Input:
            bboxes: [HxW, 4]
            scores: [HxW, num_classes]
        Output:
            bboxes: [N, 4]
            score:  [N,]
            labels: [N,]
        """

        labels = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), labels)]
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # NMS
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        return bboxes, scores, labels


    @torch.no_grad()
    def inference(self, x):
        # backbone主干网络
        feat = self.backbone(x)

        # neck网络
        feat = self.neck(feat)

        # detection head网络
        feat = self.convsets(feat)

        # 预测层
        pred = self.pred(feat)
        fmp_size = pred.shape[-2:]

        # 对pred 的size做一些view调整，便于后续的处理
        # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
        pred = pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

        # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测  
        # [B, H*W, 1]
        conf_pred = pred[..., :1]
        # [B, H*W, num_cls]
        cls_pred = pred[..., 1:1+self.num_classes]
        # [B, H*W, 4]
        txtytwth_pred = pred[..., 1+self.num_classes:]

        # 测试时，笔者默认batch是1，
        # 因此，我们不需要用batch这个维度，用[0]将其取走。
        conf_pred = conf_pred[0]            #[H*W, 1]
        cls_pred = cls_pred[0]              #[H*W, NC]
        txtytwth_pred = txtytwth_pred[0]    #[H*W, 4]

        # 每个边界框的得分
        scores = torch.sigmoid(conf_pred) * torch.softmax(cls_pred, dim=-1)
        
        # 解算边界框, 并归一化边界框: [H*W, 4]
        bboxes = self.decode_boxes(txtytwth_pred, fmp_size)
        
        # 将预测放在cpu处理上，以便进行后处理
        scores = scores.to('cpu').numpy()
        bboxes = bboxes.to('cpu').numpy()
        
        # 后处理
        bboxes, scores, labels = self.postprocess(bboxes, scores)

        return bboxes, scores, labels


    def forward(self, x):
        if not self.trainable:
            return self.inference(x)
        else:
            # backbone主干网络
            feat = self.backbone(x)

            # neck网络
            feat = self.neck(feat)

            # detection head网络
            feat = self.convsets(feat)

            # 预测层
            pred = self.pred(feat)

            # 对pred 的size做一些view调整，便于后续的处理
            # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
            pred = pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

            # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测  
            # [B, H*W, 1]
            conf_pred = pred[..., :1]
            # [B, H*W, num_cls]
            cls_pred = pred[..., 1:1+self.num_classes]
            # [B, H*W, 4]
            txtytwth_pred = pred[..., 1+self.num_classes:]

            # 网络输出
            outputs = {"pred_obj": conf_pred,                  # (Tensor) [B, M, 1]
                       "pred_cls": cls_pred,                   # (Tensor) [B, M, C]
                       "pred_txty": txtytwth_pred[..., :2],    # (Tensor) [B, M, 2]
                       "pred_twth": txtytwth_pred[..., 2:],    # (Tensor) [B, M, 2]
                       "stride": self.stride,                  # (Int)
                       "img_size": x.shape[-2:]                # (List) [img_h, img_w]
                       }           
            return outputs
        