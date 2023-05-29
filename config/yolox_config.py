# YOLOx Config


yolox_cfg = {
    'yolox_n':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'cspdarknet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 0.25,
        'depth': 0.34,
        'stride': [8, 16, 32],  # P3, P4, P5
        ## FPN
        'fpn': 'yolov5_pafpn',
        'fpn_reduce_layer': 'Conv',
        'fpn_downsample_layer': 'Conv',
        'fpn_core_block': 'CSPBlock',
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
        # ---------------- Train config ----------------
        ## input
        'multi_scale': [0.5, 1.5],   # 320 -> 960
        'trans_type': 'yolox_tiny',
        # ---------------- Assignment config ----------------
        ## matcher
        'matcher': {'center_sampling_radius': 2.5,
                    'topk_candicate': 10},
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        # ---------------- Train config ----------------
        ## close strong augmentation
        'no_aug_epoch': 20,
        ## optimizer
        'optimizer': 'sgd',        # optional: sgd, AdamW
        'momentum': 0.9,           # SGD: 0.9;    AdamW: None
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

    'yolox_s':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'cspdarknet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 0.50,
        'depth': 0.34,
        'stride': [8, 16, 32],  # P3, P4, P5
        ## FPN
        'fpn': 'yolov5_pafpn',
        'fpn_reduce_layer': 'Conv',
        'fpn_downsample_layer': 'Conv',
        'fpn_core_block': 'CSPBlock',
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
        # ---------------- Train config ----------------
        ## input
        'multi_scale': [0.5, 1.5],   # 320 -> 960
        'trans_type': 'yolox_small',
        # ---------------- Assignment config ----------------
        ## matcher
        'matcher': {'center_sampling_radius': 2.5,
                    'topk_candicate': 10},
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        # ---------------- Train config ----------------
        ## close strong augmentation
        'no_aug_epoch': 20,
        ## optimizer
        'optimizer': 'sgd',        # optional: sgd, AdamW
        'momentum': 0.9,           # SGD: 0.9;    AdamW: None
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

    'yolox_m':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'cspdarknet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 0.75,
        'depth': 0.67,
        'stride': [8, 16, 32],  # P3, P4, P5
        ## FPN
        'fpn': 'yolov5_pafpn',
        'fpn_reduce_layer': 'Conv',
        'fpn_downsample_layer': 'Conv',
        'fpn_core_block': 'CSPBlock',
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
        # ---------------- Train config ----------------
        ## input
        'multi_scale': [0.5, 1.5],   # 320 -> 960
        'trans_type': 'yolox_medium',
        # ---------------- Assignment config ----------------
        ## matcher
        'matcher': {'center_sampling_radius': 2.5,
                    'topk_candicate': 10},
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        # ---------------- Train config ----------------
        ## close strong augmentation
        'no_aug_epoch': 20,
        ## optimizer
        'optimizer': 'sgd',        # optional: sgd, AdamW
        'momentum': 0.9,           # SGD: 0.9;    AdamW: None
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

    'yolox_l':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'cspdarknet',
        'pretrained': False,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 1.0,
        'depth': 1.0,
        'stride': [8, 16, 32],  # P3, P4, P5
        ## FPN
        'fpn': 'yolov5_pafpn',
        'fpn_reduce_layer': 'Conv',
        'fpn_downsample_layer': 'Conv',
        'fpn_core_block': 'CSPBlock',
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
        # ---------------- Train config ----------------
        ## input
        'multi_scale': [0.5, 1.25],   # 320 -> 800
        'trans_type': 'yolox_large',
        # ---------------- Assignment config ----------------
        ## matcher
        'matcher': {'center_sampling_radius': 2.5,
                    'topk_candicate': 10},
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        # ---------------- Train config ----------------
        ## close strong augmentation
        'no_aug_epoch': 20,
        ## optimizer
        'optimizer': 'sgd',        # optional: sgd, AdamW
        'momentum': 0.9,           # SGD: 0.9;    AdamW: None
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

    'yolox_x':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'cspdarknet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 1.25,
        'depth': 1.34,
        'stride': [8, 16, 32],  # P3, P4, P5
        ## FPN
        'fpn': 'yolov5_pafpn',
        'fpn_reduce_layer': 'Conv',
        'fpn_downsample_layer': 'Conv',
        'fpn_core_block': 'CSPBlock',
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
        # ---------------- Train config ----------------
        ## input
        'multi_scale': [0.5, 1.25],   # 320 -> 800
        'trans_type': 'yolox_huge',
        # ---------------- Assignment config ----------------
        ## matcher
        'matcher': {'center_sampling_radius': 2.5,
                    'topk_candicate': 10},
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        # ---------------- Train config ----------------
        ## close strong augmentation
        'no_aug_epoch': 20,
        ## optimizer
        'optimizer': 'sgd',        # optional: sgd, AdamW
        'momentum': 0.9,           # SGD: 0.9;    AdamW: None
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