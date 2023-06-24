# YOLOv4 Config

yolov4_cfg = {
    'yolov4':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'cspdarknet53',
        'pretrained': True,
        'stride': [8, 16, 32],  # P3, P4, P5
        'width': 1.0,
        'depth': 1.0,
        'max_stride': 32,
        ## Neck
        'neck': 'csp_sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        ## FPN
        'fpn': 'yolov4_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        ## Head
        'head': 'decoupled_head',
        'head_act': 'silu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],     # P3
                        [30, 61],   [62, 45],   [59, 119],    # P4
                        [116, 90],  [156, 198], [373, 326]],  # P5
        # ---------------- Train config ----------------
        ## input
        'trans_type': 'yolov5_large',
        'multi_scale': [0.5, 1.0],
        # ---------------- Assignment config ----------------
        ## matcher
        'iou_thresh': 0.5,
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        # ---------------- Train config ----------------
        ## close strong augmentation
        'no_aug_epoch': 10,
        ## optimizer
        'optimizer': 'sgd',        # optional: sgd, AdamW
        'momentum': 0.937,         # SGD: 0.937;    AdamW: None
        'weight_decay': 5e-4,      # SGD: 5e-4;     AdamW: 5e-2
        'clip_grad': 10,           # SGD: 10.0;     AdamW: -1
        ## model EMA
        'ema_decay': 0.9999,       # SGD: 0.9999;   AdamW: 0.9998
        'ema_tau': 2000,
        ## lr schedule
        'scheduler': 'linear',
        'lr0': 0.01,              # SGD: 0.01;     AdamW: 0.001
        'lrf': 0.01,               # SGD: 0.01;     AdamW: 0.01
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
    },

    'yolov4_t':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'cspdarknet_tiny',
        'pretrained': True,
        'stride': [8, 16, 32],  # P3, P4, P5
        'width': 0.25,
        'depth': 0.34,
        'max_stride': 32,
        ## Neck
        'neck': 'csp_sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        ## FPN
        'fpn': 'yolov4_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        ## Head
        'head': 'decoupled_head',
        'head_act': 'silu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],     # P3
                        [30, 61],   [62, 45],   [59, 119],    # P4
                        [116, 90],  [156, 198], [373, 326]],  # P5
        # ---------------- Train config ----------------
        ## input
        'trans_type': 'yolov5_nano',
        'multi_scale': [0.5, 1.0],
        # ---------------- Assignment config ----------------
        ## matcher
        'iou_thresh': 0.5,
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        # ---------------- Train config ----------------
        ## close strong augmentation
        'no_aug_epoch': 10,
        ## optimizer
        'optimizer': 'sgd',        # optional: sgd, AdamW
        'momentum': 0.937,         # SGD: 0.937;    AdamW: None
        'weight_decay': 5e-4,      # SGD: 5e-4;     AdamW: 5e-2
        'clip_grad': 10,           # SGD: 10.0;     AdamW: -1
        ## model EMA
        'ema_decay': 0.9999,       # SGD: 0.9999;   AdamW: 0.9998
        'ema_tau': 2000,
        ## lr schedule
        'scheduler': 'linear',
        'lr0': 0.01,              # SGD: 0.01;     AdamW: 0.001
        'lrf': 0.01,               # SGD: 0.01;     AdamW: 0.01
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
    },

}