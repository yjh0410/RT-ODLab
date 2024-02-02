#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

from .loss import build_criterion
from .rtdetr import RT_DETR


# build object detector
def build_rtdetr(args, cfg, num_classes=80, trainable=False, deploy=False):
    print('==============================')
    print('Build {} ...'.format(args.model.upper()))
    
    print('==============================')
    print('Model Configuration: \n', cfg)
    
    # -------------- Build RT-DETR --------------
    model = RT_DETR(cfg             = cfg,
                    num_classes     = num_classes,
                    nms_thresh      = args.nms_thresh,
                    conf_thresh     = args.conf_thresh,
                    topk            = 300,
                    deploy          = deploy,
                    no_multi_labels = args.no_multi_labels,
                    use_nms         = True,
                    nms_class_agnostic = args.nms_class_agnostic
                    )
            
    # -------------- Build criterion --------------
    criterion = None
    if trainable:
        # build criterion for training
        criterion = build_criterion(cfg, num_classes)
        
    return model, criterion
