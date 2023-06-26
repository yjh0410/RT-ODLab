#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from .loss import build_criterion
from .rtdetr import RTDETR


# build object detector
def build_rtdetr(args, cfg, device, num_classes=80, trainable=False, deploy=False):
    print('==============================')
    print('Build {} ...'.format(args.model.upper()))
    
    print('==============================')
    print('Model Configuration: \n', cfg)
    
    # -------------- Build rtdetr --------------
    model = RTDETR(
        cfg=cfg,
        device=device, 
        num_classes=num_classes,
        trainable=trainable,
        aux_loss=trainable,
        with_box_refine=True,
        deploy=deploy
        )

    # -------------- Build criterion --------------
    criterion = None
    if trainable:
        # build criterion for training
        criterion = build_criterion(cfg, num_classes, aux_loss=True)
        
    return model, criterion
