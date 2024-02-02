#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

from .loss import build_criterion
from .rtpdetr import RT_PDETR


# build object detector
def build_rtpdetr(args, cfg, num_classes=80, trainable=False, deploy=False):
    print('==============================')
    print('Build {} ...'.format(args.model.upper()))
    
    print('==============================')
    print('Model Configuration: \n', cfg)
    
    # -------------- Build RT-DETR --------------
    model = RT_PDETR(cfg             = cfg,
                     num_classes     = num_classes,
                     conf_thresh     = args.conf_thresh,
                     topk            = 300,
                     deploy          = deploy,
                     no_multi_labels = args.no_multi_labels,
                     )
            
    # -------------- Build criterion --------------
    criterion = None
    if trainable:
        # build criterion for training
        criterion = build_criterion(cfg, num_classes, aux_loss=True)
        
    return model, criterion
