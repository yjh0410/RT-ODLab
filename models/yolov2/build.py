#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from .loss import build_criterion
from .yolov2 import YOLOv2


# build object detector
def build_yolov2(args, cfg, device, num_classes=80, trainable=False):
    print('==============================')
    print('Build {} ...'.format(args.model.upper()))
    
    print('==============================')
    print('Model Configuration: \n', cfg)
    
    model = YOLOv2(
        cfg = cfg,
        device = device,
        img_size = args.img_size,
        num_classes = num_classes,
        conf_thresh = args.conf_thresh,
        nms_thresh = args.nms_thresh,
        topk = args.topk,
        trainable = trainable
        )

    criterion = None
    if trainable:
        # build criterion for training
        criterion = build_criterion(cfg, device, num_classes)

    return model, criterion
