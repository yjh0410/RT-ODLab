#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

from .loss import build_criterion
from .yolov3 import YOLOv3


# build object detector
def build_yolov3(args, cfg, device, num_classes=80, trainable=False, deploy=False):
    print('==============================')
    print('Build {} ...'.format(args.model.upper()))
    
    print('==============================')
    print('Model Configuration: \n', cfg)
    
    # -------------- Build YOLO --------------
    model = YOLOv3(cfg                = cfg,
                   device             = device, 
                   num_classes        = num_classes,
                   trainable          = trainable,
                   conf_thresh        = args.conf_thresh,
                   nms_thresh         = args.nms_thresh,
                   topk               = args.topk,
                   deploy             = deploy,
                   no_multi_labels    = args.no_multi_labels,
                   nms_class_agnostic = args.nms_class_agnostic
                   )

    # -------------- Initialize YOLO --------------
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03    
    # Init bias
    init_prob = 0.01
    bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
    # obj pred
    for obj_pred in model.obj_preds:
        b = obj_pred.bias.view(1, -1)
        b.data.fill_(bias_value.item())
        obj_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    # cls pred
    for cls_pred in model.cls_preds:
        b = cls_pred.bias.view(1, -1)
        b.data.fill_(bias_value.item())
        cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    # reg pred
    for reg_pred in model.reg_preds:
        b = reg_pred.bias.view(-1, )
        b.data.fill_(1.0)
        reg_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        w = reg_pred.weight
        w.data.fill_(0.)
        reg_pred.weight = torch.nn.Parameter(w, requires_grad=True)


    # -------------- Build criterion --------------
    criterion = None
    if trainable:
        # build criterion for training
        criterion = build_criterion(cfg, device, num_classes)
    return model, criterion
