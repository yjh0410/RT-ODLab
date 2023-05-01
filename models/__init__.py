#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
from .yolov1.build import build_yolov1
from .yolov2.build import build_yolov2
from .yolov3.build import build_yolov3
from .yolov4.build import build_yolov4
from .yolov5.build import build_yolov5
from .yolov7.build import build_yolov7
from .yolov8.build import build_yolov8
from .yolox.build import build_yolox


# build object detector
def build_model(args, 
                model_cfg,
                device, 
                num_classes=80, 
                trainable=False):
    # YOLOv1    
    if args.model == 'yolov1':
        model, criterion = build_yolov1(
            args, model_cfg, device, num_classes, trainable)
    # YOLOv2   
    elif args.model == 'yolov2':
        model, criterion = build_yolov2(
            args, model_cfg, device, num_classes, trainable)
    # YOLOv3   
    elif args.model == 'yolov3':
        model, criterion = build_yolov3(
            args, model_cfg, device, num_classes, trainable)
    # YOLOv4   
    elif args.model == 'yolov4':
        model, criterion = build_yolov4(
            args, model_cfg, device, num_classes, trainable)
    # YOLOv5   
    elif args.model in ['yolov5_n', 'yolov5_s', 'yolov5_m', 'yolov5_l', 'yolov5_x']:
        model, criterion = build_yolov5(
            args, model_cfg, device, num_classes, trainable)
    # YOLOv7
    elif args.model in ['yolov7_t', 'yolov7_l', 'yolov7_x']:
        model, criterion = build_yolov7(
            args, model_cfg, device, num_classes, trainable)
    # YOLOv8
    elif args.model in ['yolov8_n', 'yolov8_s', 'yolov8_m', 'yolov8_l', 'yolov8_x']:
        model, criterion = build_yolov8(
            args, model_cfg, device, num_classes, trainable)
    # YOLOX   
    elif args.model == 'yolox':
        model, criterion = build_yolox(
            args, model_cfg, device, num_classes, trainable)

    if trainable:
        # Load pretrained weight
        if args.pretrained is not None:
            print('Loading COCO pretrained weight ...')
            checkpoint = torch.load(args.pretrained, map_location='cpu')
            # checkpoint state dict
            checkpoint_state_dict = checkpoint.pop("model")
            # model state dict
            model_state_dict = model.state_dict()
            # check
            for k in list(checkpoint_state_dict.keys()):
                if k in model_state_dict:
                    shape_model = tuple(model_state_dict[k].shape)
                    shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                    if shape_model != shape_checkpoint:
                        checkpoint_state_dict.pop(k)
                        print(k)
                else:
                    checkpoint_state_dict.pop(k)
                    print(k)

            model.load_state_dict(checkpoint_state_dict, strict=False)

        # keep training
        if args.resume is not None:
            print('keep training: ', args.resume)
            checkpoint = torch.load(args.resume, map_location='cpu')
            # checkpoint state dict
            checkpoint_state_dict = checkpoint.pop("model")
            # check
            new_checkpoint_state_dict = {}

            for k in list(checkpoint_state_dict.keys()):
                v = checkpoint_state_dict[k]
                if 'reduce_layer_3' in k:
                    k_new = k.split('.')
                    k_new[1] = 'downsample_layer_1'
                    k = k_new[0] + '.' + k_new[1] + '.' + k_new[2] + '.' + k_new[3] + '.' + k_new[4]
                elif 'reduce_layer_4' in k:
                    k_new = k.split('.')
                    k_new[1] = 'downsample_layer_2'
                    k = k_new[0] + '.' + k_new[1] + '.' + k_new[2] + '.' + k_new[3] + '.' + k_new[4]
                new_checkpoint_state_dict[k] = v
            model.load_state_dict(new_checkpoint_state_dict)

        return model, criterion

    else:      
        return model