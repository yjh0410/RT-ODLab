#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

from .loss import build_criterion
from .rtrdet import RTRDet


# build object detector
def build_rtrdet(args, cfg, device, num_classes=80, trainable=False, deploy=False):
    print('==============================')
    print('Build {} ...'.format(args.model.upper()))
        
    # -------------- Build RTRDet --------------
    model = RTRDet(cfg         = cfg,
                   device      = device, 
                   num_classes = num_classes,
                   trainable   = trainable,
                   aux_loss    = True if trainable else False,
                   deploy      = deploy
                   )
            
    # -------------- Build criterion --------------
    criterion = None
    if trainable:
        # build criterion for training
        criterion = build_criterion(cfg, num_classes, aux_loss=True)

    return model, criterion