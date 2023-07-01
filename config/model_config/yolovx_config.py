# YOLOvx Config


yolovx_cfg = {
    'yolovx_n':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elannet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 0.25,
        'depth': 0.34,
        'stride': [8, 16, 32],  # P3, P4, P5
        'max_stride': 32,
        ## Neck: SPP
        'neck': 'sppf',
        'neck_expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        ## Neck: PaFPN
        'fpn': 'yolo_pafpn',
        'fpn_reduce_layer': 'Conv',
        'fpn_downsample_layer': 'Conv',
        'fpn_core_block': 'elanblock',
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
        ## Input
        'multi_scale': [0.5, 1.5],   # 320 -> 960
        'trans_type': 'yolox_nano',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': {'center_sampling_radius': 2.5,
                    'topk_candicate': 10},
        # ---------------- Loss config ----------------
        ## Loss weight
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        # ---------------- Train config ----------------
        ## Training configuration
        'no_aug_epoch': 20,
        'trainer_type': 'rtmdet',
        ## Optimizer
        'optimizer': 'adamw',      # optional: sgd, adam, adamw
        'momentum': None,          # SGD: 0.937;    AdamW: invalid
        'weight_decay': 5e-2,      # SGD: 5e-4;     AdamW: 5e-2
        'clip_grad': 35,           # SGD: 10.0;     AdamW: -1
        ## Model EMA
        'ema_decay': 0.9998,       # SGD: 0.9999;   AdamW: 0.9998
        'ema_tau': 2000,
        ## LR schedule
        'scheduler': 'cos_linear',
        'lr0': 0.001,              # SGD: 0.01;     AdamW: 0.001
        'lrf': 0.01,               # SGD: 0.01;     AdamW: 0.01
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
    },

    'yolovx_l':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elannet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 1.0,
        'depth': 1.0,
        'stride': [8, 16, 32],  # P3, P4, P5
        'max_stride': 32,
        ## Neck: SPP
        'neck': 'sppf',
        'neck_expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        ## Neck: PaFPN
        'fpn': 'yolo_pafpn',
        'fpn_reduce_layer': 'Conv',
        'fpn_downsample_layer': 'Conv',
        'fpn_core_block': 'elanblock',
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
        ## Input
        'multi_scale': [0.5, 1.25],   # 320 -> 800
        'trans_type': 'yolox_large',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': {'center_sampling_radius': 2.5,
                    'topk_candicate': 10},
        # ---------------- Loss config ----------------
        ## Loss weight
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        # ---------------- Train config ----------------
        ## Training configuration
        'no_aug_epoch': 20,
        'trainer_type': 'rtmdet',
        ## Optimizer
        'optimizer': 'adamw',      # optional: sgd, adam, adamw
        'momentum': None,          # SGD: 0.937;    AdamW: invalid
        'weight_decay': 5e-2,      # SGD: 5e-4;     AdamW: 5e-2
        'clip_grad': 35,           # SGD: 10.0;     AdamW: -1
        ## Model EMA
        'ema_decay': 0.9998,       # SGD: 0.9999;   AdamW: 0.9998
        'ema_tau': 2000,
        ## LR schedule
        'scheduler': 'cos_linear',
        'lr0': 0.001,              # SGD: 0.01;     AdamW: 0.001
        'lrf': 0.01,               # SGD: 0.01;     AdamW: 0.01
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
    },

}