#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
# YOLO series
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
                trainable=False,
                deploy=False):
    # YOLOv1    
    if args.model == 'yolov1':
        model, criterion = build_yolov1(
            args, model_cfg, device, num_classes, trainable, deploy)
    # YOLOv2   
    elif args.model == 'yolov2':
        model, criterion = build_yolov2(
            args, model_cfg, device, num_classes, trainable, deploy)
    # YOLOv3   
    elif args.model in ['yolov3', 'yolov3_tiny']:
        model, criterion = build_yolov3(
            args, model_cfg, device, num_classes, trainable, deploy)
    # YOLOv4   
    elif args.model in ['yolov4', 'yolov4_tiny']:
        model, criterion = build_yolov4(
            args, model_cfg, device, num_classes, trainable, deploy)
    # YOLOv5   
    elif args.model in ['yolov5_n', 'yolov5_s', 'yolov5_m', 'yolov5_l', 'yolov5_x']:
        model, criterion = build_yolov5(
            args, model_cfg, device, num_classes, trainable, deploy)
    # YOLOv5-AdamW
    elif args.model in ['yolov5_n_adamw', 'yolov5_s_adamw', 'yolov5_m_adamw', 'yolov5_l_adamw', 'yolov5_x_adamw']:
        model, criterion = build_yolov5(
            args, model_cfg, device, num_classes, trainable, deploy)
    # YOLOv7
    elif args.model in ['yolov7_tiny', 'yolov7', 'yolov7_x']:
        model, criterion = build_yolov7(
            args, model_cfg, device, num_classes, trainable, deploy)
    # YOLOv8
    elif args.model in ['yolov8_n', 'yolov8_s', 'yolov8_m', 'yolov8_l', 'yolov8_x']:
        model, criterion = build_yolov8(
            args, model_cfg, device, num_classes, trainable, deploy)
    # YOLOX
    elif args.model in ['yolox_n', 'yolox_s', 'yolox_m', 'yolox_l', 'yolox_x']:
        model, criterion = build_yolox(
            args, model_cfg, device, num_classes, trainable, deploy)
    # YOLOX-AdamW
    elif args.model in ['yolox_n_adamw', 'yolox_s_adamw', 'yolox_m_adamw', 'yolox_l_adamw', 'yolox_x_adamw']:
        model, criterion = build_yolox(
            args, model_cfg, device, num_classes, trainable, deploy)

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
        if args.resume and args.resume != "None":
            checkpoint = torch.load(args.resume, map_location='cpu')
            # checkpoint state dict
            try:
                checkpoint_state_dict = checkpoint.pop("model")
                print('Load model from the checkpoint: ', args.resume)
                model.load_state_dict(checkpoint_state_dict)
                del checkpoint, checkpoint_state_dict
            except:
                print("No model in the given checkpoint.")

        return model, criterion

    else:      
        return model