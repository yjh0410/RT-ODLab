# FreeYOLOv2 Config


yolo_free_v2_cfg = {
    'yolo_free_v2_n':{
        # ---------------- Model config ----------------
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
        'max_stride': 32,
        ## Neck: SPP
        'neck': 'csp_sppf',
        'neck_expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        ## Neck: PaFPN
        'fpn': 'yolo_pafpn',
        'fpn_reduce_layer': 'Conv',
        'fpn_downsample_layer': 'Conv',
        'fpn_core_block': 'elan_cspblock',
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
        'reg_max': 16,
        # ---------------- Train config ----------------
        ## Input
        'multi_scale': [0.5, 1.5],   # 320 -> 960
        'trans_type': 'yolox_nano',
        # ---------------- Assignment config ----------------
        'matcher': {'topk': 10,
                    'alpha': 0.5,
                    'beta': 6.0},
        # ---------------- Loss config ----------------
        ## Loss weight
        'cls_loss': 'bce',   # optional: vfl
        'loss_cls_weight': 0.5,
        'loss_iou_weight': 7.5,
        'loss_dfl_weight': 1.5,
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

    'yolo_free_v2_l':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elan_cspnet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 1.0,
        'depth': 1.0,
        'ratio': 1.0,
        'stride': [8, 16, 32],  # P3, P4, P5
        'max_stride': 32,
        ## Neck: SPP
        'neck': 'csp_sppf',
        'neck_expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        ## Neck: PaFPN
        'fpn': 'yolo_pafpn',
        'fpn_reduce_layer': 'Conv',
        'fpn_downsample_layer': 'Conv',
        'fpn_core_block': 'elan_cspblock',
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
        'reg_max': 16,
        # ---------------- Train config ----------------
        ## Input
        'multi_scale': [0.5, 1.25],   # 320 -> 800
        'trans_type': 'yolox_large',
        # ---------------- Assignment config ----------------
        'matcher': {'topk': 10,
                    'alpha': 0.5,
                    'beta': 6.0},
        # ---------------- Loss config ----------------
        ## Loss weight
        'cls_loss': 'bce',   # optional: vfl
        'loss_cls_weight': 0.5,
        'loss_iou_weight': 7.5,
        'loss_dfl_weight': 1.5,
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