# yolov8 config

yolov5_plus_cfg = {
    'yolov5_plus_n':{
        # input
        'trans_type': 'yolov5_tiny',
        'multi_scale': [0.5, 1.0],   # 320 -> 640
        # ----------------- Model config -----------------
        ## Backbone
        'backbone': 'elan_cspnet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 0.25,
        'depth': 0.34,
        'ratio': 2.0,
        'stride': [8, 16, 32],  # P3, P4, P5
        ## Neck: SPP
        'neck': 'sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        ## Neck: FPN
        'fpn': 'yolov5_plus_pafpn',
        'fpn_reduce_layer': 'Conv',
        'fpn_downsample_layer': 'Conv',
        'fpn_core_block': 'ELAN_CSPBlock',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],     # P3
                        [30, 61],   [62, 45],   [59, 119],    # P4
                        [116, 90],  [156, 198], [373, 326]],  # P5
        ## Head
        'head': 'decoupled_head',
        'head_act': 'silu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        # ----------------- Label Assignment config -----------------
        'matcher': {
            ## For fixed assigner
            'anchor_thresh': 4.0,
            ## For dynamic assigner
            'topk': 10,
            'alpha': 0.5,
            'beta': 6.0},
        # ----------------- Loss config -----------------
        'cls_loss': 'bce',
        'loss_cls_weight': 0.5,
        'loss_iou_weight': 7.5,
        # ----------------- Train config -----------------
        ## stop strong augment
        'no_aug_epoch': 20,
        ## optimizer
        'optimizer': 'sgd',        # optional: sgd, adamw
        'momentum': 0.937,         # SGD: 0.937;    AdamW: invalid
        'weight_decay': 5e-4,      # SGD: 5e-4;     AdamW: 5e-2
        'clip_grad': 10,           # SGD: 10.0;     AdamW: -1
        ## Model EMA
        'ema_decay': 0.9999,       # SGD: 0.9999;   AdamW: 0.9998
        'ema_tau': 2000,
        ## LR schedule
        'scheduler': 'linear',
        'lr0': 0.01,               # SGD: 0.01;     AdamW: 0.004
        'lrf': 0.01,               # SGD: 0.01;     AdamW: 0.05
        ## WarmUpLR schedule
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
    },

    'yolov5_plus_s':{
        # input
        'trans_type': 'yolov5_small',
        'multi_scale': [0.5, 1.0],   # 320 -> 640
        # ----------------- Model config 
        # Backbone
        'backbone': 'elan_cspnet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 0.5,
        'depth': 0.34,
        'ratio': 2.0,
        'stride': [8, 16, 32],  # P3, P4, P5
        # Neck: SPP
        'neck': 'sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # Neck: FPN
        'fpn': 'yolov5_plus_pafpn',
        'fpn_reduce_layer': 'Conv',
        'fpn_downsample_layer': 'Conv',
        'fpn_core_block': 'ELAN_CSPBlock',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],     # P3
                        [30, 61],   [62, 45],   [59, 119],    # P4
                        [116, 90],  [156, 198], [373, 326]],  # P5
        # Head
        'head': 'decoupled_head',
        'head_act': 'silu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        # ----------------- Label Assignment config -----------------
        'matcher': {
            ## For fixed assigner
            'anchor_thresh': 4.0,
            ## For dynamic assigner
            'topk': 10,
            'alpha': 0.5,
            'beta': 6.0},
        # ----------------- Loss config -----------------
        'cls_loss': 'bce',
        'loss_cls_weight': 0.5,
        'loss_iou_weight': 7.5,
        # ----------------- Train config -----------------
        # stop strong augment
        'no_aug_epoch': 20,
        ## optimizer
        'optimizer': 'sgd',        # optional: sgd, adamw
        'momentum': 0.937,         # SGD: 0.937;    AdamW: invalid
        'weight_decay': 5e-4,      # SGD: 5e-4;     AdamW: 5e-2
        'clip_grad': 10,           # SGD: 10.0;     AdamW: -1
        ## Model EMA
        'ema_decay': 0.9999,       # SGD: 0.9999;   AdamW: 0.9998
        'ema_tau': 2000,
        ## LR schedule
        'scheduler': 'linear',
        'lr0': 0.01,               # SGD: 0.01;     AdamW: 0.004
        'lrf': 0.01,               # SGD: 0.01;     AdamW: 0.05
        ## WarmUpLR schedule
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
    },

    'yolov5_plus_m':{
        # input
        'trans_type': 'yolov5_medium',
        'multi_scale': [0.5, 1.0],   # 320 -> 640
        # ----------------- Model config 
        # Backbone
        'backbone': 'elan_cspnet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 0.75,
        'depth': 0.67,
        'ratio': 1.5,
        'stride': [8, 16, 32],  # P3, P4, P5
        # Neck: SPP
        'neck': 'sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # Neck: FPN
        'fpn': 'yolov5_plus_pafpn',
        'fpn_reduce_layer': 'Conv',
        'fpn_downsample_layer': 'Conv',
        'fpn_core_block': 'ELAN_CSPBlock',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],     # P3
                        [30, 61],   [62, 45],   [59, 119],    # P4
                        [116, 90],  [156, 198], [373, 326]],  # P5
        # Head
        'head': 'decoupled_head',
        'head_act': 'silu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        # ----------------- Label Assignment config -----------------
        'matcher': {
            ## For fixed assigner
            'anchor_thresh': 4.0,
            ## For dynamic assigner
            'topk': 10,
            'alpha': 0.5,
            'beta': 6.0},
        # ----------------- Loss config -----------------
        'cls_loss': 'bce',
        'loss_cls_weight': 0.5,
        'loss_iou_weight': 7.5,
        # ----------------- Train config -----------------
        # stop strong augment
        'no_aug_epoch': 20,
        ## optimizer
        'optimizer': 'sgd',        # optional: sgd, adamw
        'momentum': 0.937,         # SGD: 0.937;    AdamW: invalid
        'weight_decay': 5e-4,      # SGD: 5e-4;     AdamW: 5e-2
        'clip_grad': 10,           # SGD: 10.0;     AdamW: -1
        ## Model EMA
        'ema_decay': 0.9999,       # SGD: 0.9999;   AdamW: 0.9998
        'ema_tau': 2000,
        ## LR schedule
        'scheduler': 'linear',
        'lr0': 0.01,               # SGD: 0.01;     AdamW: 0.004
        'lrf': 0.01,               # SGD: 0.01;     AdamW: 0.05
        ## WarmUpLR schedule
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
    },

    'yolov5_plus_l':{
        # input
        'trans_type': 'yolov5_large',
        'multi_scale': [0.5, 1.0],   # 320 -> 640
        # ----------------- Model config 
        # Backbone
        'backbone': 'elan_cspnet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 1.0,
        'depth': 1.0,
        'ratio': 1.0,
        'stride': [8, 16, 32],  # P3, P4, P5
        # Neck: SPP
        'neck': 'sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # Neck: FPN
        'fpn': 'yolov5_plus_pafpn',
        'fpn_reduce_layer': 'Conv',
        'fpn_downsample_layer': 'Conv',
        'fpn_core_block': 'ELAN_CSPBlock',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],     # P3
                        [30, 61],   [62, 45],   [59, 119],    # P4
                        [116, 90],  [156, 198], [373, 326]],  # P5
        # Head
        'head': 'decoupled_head',
        'head_act': 'silu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        # ----------------- Label Assignment config -----------------
        'matcher': {
            ## For fixed assigner
            'anchor_thresh': 4.0,
            ## For dynamic assigner
            'topk': 10,
            'alpha': 0.5,
            'beta': 6.0},
        # ----------------- Loss config -----------------
        'cls_loss': 'bce',
        'loss_cls_weight': 0.5,
        'loss_iou_weight': 7.5,
        # ----------------- Train config -----------------
        # stop strong augment
        'no_aug_epoch': 20,
        ## optimizer
        'optimizer': 'sgd',        # optional: sgd, adamw
        'momentum': 0.937,         # SGD: 0.937;    AdamW: invalid
        'weight_decay': 5e-4,      # SGD: 5e-4;     AdamW: 5e-2
        'clip_grad': 10,           # SGD: 10.0;     AdamW: -1
        ## Model EMA
        'ema_decay': 0.9999,       # SGD: 0.9999;   AdamW: 0.9998
        'ema_tau': 2000,
        ## LR schedule
        'scheduler': 'linear',
        'lr0': 0.01,               # SGD: 0.01;     AdamW: 0.004
        'lrf': 0.01,               # SGD: 0.01;     AdamW: 0.05
        ## WarmUpLR schedule
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
    },

    'yolov5_plus_x':{
        # input
        'trans_type': 'yolov5_huge',
        'multi_scale': [0.5, 1.0],   # 320 -> 640
        # ----------------- Model config 
        # Backbone
        'backbone': 'elan_cspnet',
        'pretrained': False,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 1.25,
        'depth': 1.0,
        'ratio': 1.0,
        'stride': [8, 16, 32],  # P3, P4, P5
        # Neck: SPP
        'neck': 'sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # Neck: FPN
        'fpn': 'yolov5_plus_pafpn',
        'fpn_reduce_layer': 'Conv',
        'fpn_downsample_layer': 'Conv',
        'fpn_core_block': 'ELAN_CSPBlock',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],     # P3
                        [30, 61],   [62, 45],   [59, 119],    # P4
                        [116, 90],  [156, 198], [373, 326]],  # P5
        # Head
        'head': 'decoupled_head',
        'head_act': 'silu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        # ----------------- Label Assignment config -----------------
        'matcher': {
            ## For fixed assigner
            'anchor_thresh': 4.0,
            ## For dynamic assigner
            'topk': 10,
            'alpha': 0.5,
            'beta': 6.0},
        # ----------------- Loss config -----------------
        'cls_loss': 'bce',
        'loss_cls_weight': 0.5,
        'loss_iou_weight': 7.5,
        # ----------------- Train config -----------------
        # stop strong augment
        'no_aug_epoch': 20,
        ## optimizer
        'optimizer': 'sgd',        # optional: sgd, adamw
        'momentum': 0.937,         # SGD: 0.937;    AdamW: invalid
        'weight_decay': 5e-4,      # SGD: 5e-4;     AdamW: 5e-2
        'clip_grad': 10,           # SGD: 10.0;     AdamW: -1
        ## Model EMA
        'ema_decay': 0.9999,       # SGD: 0.9999;   AdamW: 0.9998
        'ema_tau': 2000,
        ## LR schedule
        'scheduler': 'linear',
        'lr0': 0.01,               # SGD: 0.01;     AdamW: 0.004
        'lrf': 0.01,               # SGD: 0.01;     AdamW: 0.05
        ## WarmUpLR schedule
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
    },

}