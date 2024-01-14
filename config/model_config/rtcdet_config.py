# Real-time Convolution Object Detector


# ------------------- Det task --------------------
rtcdet_cfg = {
    'rtcdet_n':{
        # ---------------- Model config ----------------
        ## Backbone
        'bk_pretrained': True,
        'bk_pretrained_mae': False,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'width': 0.25,
        'depth': 0.34,
        'ratio': 2.0,
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
        'fpn': 'rtcdet_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        ## Head
        'det_head': {'name': 'decoupled_head',
                     'num_cls_head': 2,
                     'num_reg_head': 2,
                     'head_act': 'silu',
                     'head_norm': 'BN',
                     'head_depthwise': False,  
                     },
        'seg_head': {'name': None,
                     },
        'pos_head': {'name': None,
                     },
        # ---------------- Train config ----------------
        ## input
        'multi_scale': [0.5, 1.25],   # 320 -> 800
        'trans_type': 'yolox_n',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': "aligned_simota",
        'matcher_hpy': {'soft_center_radius': 3.0,
                        'topk_candidates': 13},
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_cls_weight': 1.0,
        'loss_box_weight': 2.0,
        # ---------------- Train config ----------------
        'trainer_type': 'rtcdet',
    },

    'rtcdet_s':{
        # ---------------- Model config ----------------
        ## Backbone
        'bk_pretrained': True,
        'bk_pretrained_mae': False,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'width': 0.50,
        'depth': 0.34,
        'ratio': 2.0,
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
        'fpn': 'rtcdet_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        ## Head
        'det_head': {'name': 'decoupled_head',
                     'num_cls_head': 2,
                     'num_reg_head': 2,
                     'head_act': 'silu',
                     'head_norm': 'BN',
                     'head_depthwise': False,  
                     },
        'seg_head': {'name': None,
                     },
        'pos_head': {'name': None,
                     },
        # ---------------- Train config ----------------
        ## input
        'multi_scale': [0.5, 1.25],   # 320 -> 800
        'trans_type': 'yolox_s',
        # ---------------- Assignment config ----------------
        ## matcher
        'matcher': "aligned_simota",
        'matcher_hpy': {'soft_center_radius': 3.0,
                        'topk_candidates': 13},
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_cls_weight': 1.0,
        'loss_box_weight': 2.0,
        # ---------------- Train config ----------------
        'trainer_type': 'rtcdet',
    },

    'rtcdet_m':{
        # ---------------- Model config ----------------
        ## Backbone
        'bk_pretrained': True,
        'bk_pretrained_mae': False,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'width': 0.75,
        'depth': 0.67,
        'ratio': 1.5,
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
        'fpn': 'rtcdet_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        ## Head
        'det_head': {'name': 'decoupled_head',
                     'num_cls_head': 2,
                     'num_reg_head': 2,
                     'head_act': 'silu',
                     'head_norm': 'BN',
                     'head_depthwise': False,  
                     },
        'seg_head': {'name': None,
                     },
        'pos_head': {'name': None,
                     },
        # ---------------- Train config ----------------
        ## input
        'multi_scale': [0.5, 1.25],   # 320 -> 800
        'trans_type': 'yolox_m',
        # ---------------- Assignment config ----------------
        ## matcher
        'matcher': "aligned_simota",
        'matcher_hpy': {'soft_center_radius': 3.0,
                        'topk_candidates': 13},
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_cls_weight': 1.0,
        'loss_box_weight': 2.0,
        # ---------------- Train config ----------------
        'trainer_type': 'rtcdet',
    },

    'rtcdet_l':{
        # ---------------- Model config ----------------
        ## Backbone
        'bk_pretrained': True,
        'bk_pretrained_mae': False,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'width': 1.0,
        'depth': 1.0,
        'ratio': 1.0,
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
        'fpn': 'rtcdet_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        ## Head
        'det_head': {'name': 'decoupled_head',
                     'num_cls_head': 2,
                     'num_reg_head': 2,
                     'head_act': 'silu',
                     'head_norm': 'BN',
                     'head_depthwise': False,  
                     },
        'seg_head': {'name': None,
                     },
        'pos_head': {'name': None,
                     },
        # ---------------- Train config ----------------
        ## input
        'multi_scale': [0.5, 1.25],   # 320 -> 800
        'trans_type': 'yolox_l',
        # ---------------- Assignment config ----------------
        ## matcher
        'matcher': "aligned_simota",
        'matcher_hpy': {'soft_center_radius': 3.0,
                        'topk_candidates': 13},
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_cls_weight': 1.0,
        'loss_box_weight': 2.0,
        # ---------------- Train config ----------------
        'trainer_type': 'rtcdet',
    },

    'rtcdet_x':{
        # ---------------- Model config ----------------
        ## Backbone
        'bk_pretrained': True,
        'bk_pretrained_mae': False,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'width': 1.25,
        'depth': 1.34,
        'ratio': 1.0,
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
        'fpn': 'rtcdet_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        ## Head
        'det_head': {'name': 'decoupled_head',
                     'num_cls_head': 2,
                     'num_reg_head': 2,
                     'head_act': 'silu',
                     'head_norm': 'BN',
                     'head_depthwise': False,  
                     },
        'seg_head': {'name': None,
                     },
        'pos_head': {'name': None,
                     },
        # ---------------- Train config ----------------
        ## input
        'multi_scale': [0.5, 1.25],   # 320 -> 800
        'trans_type': 'yolox_x',
        # ---------------- Assignment config ----------------
        ## matcher
        'matcher': "aligned_simota",
        'matcher_hpy': {'soft_center_radius': 3.0,
                        'topk_candidates': 13},
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_cls_weight': 1.0,
        'loss_box_weight': 2.0,
        # ---------------- Train config ----------------
        'trainer_type': 'rtcdet',
    },

}


# ------------------- Det + Seg task -------------------
rtcdet_seg_cfg = {
    'rtcdet_seg_n':{
        # ---------------- Model config ----------------
        ## Backbone
        'bk_pretrained': True,
        'bk_pretrained_mae': False,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'width': 0.25,
        'depth': 0.34,
        'ratio': 2.0,
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
        'fpn': 'rtcdet_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        ## Head
        'det_head': {'name': 'decoupled_head',
                     'num_cls_head': 2,
                     'num_reg_head': 2,
                     'head_act': 'silu',
                     'head_norm': 'BN',
                     'head_depthwise': False,  
                     },
        'seg_head': {'name': None,
                     },
        'pos_head': {'name': None,
                     },
        # ---------------- Train config ----------------
        ## input
        'multi_scale': [0.5, 1.25],   # 320 -> 800
        'trans_type': 'yolox_n',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': "aligned_simota",
        'matcher_hpy': {'soft_center_radius': 3.0,
                        'topk_candidates': 13},
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_cls_weight': 1.0,
        'loss_box_weight': 2.0,
        # ---------------- Train config ----------------
        'trainer_type': 'rtcdet',
    },

}


# ------------------- Det + Pos task -------------------
rtcdet_pos_cfg = {
    'rtcdet_pos_n':{
        # ---------------- Model config ----------------
        ## Backbone
        'bk_pretrained': True,
        'bk_pretrained_mae': False,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'width': 0.25,
        'depth': 0.34,
        'ratio': 2.0,
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
        'fpn': 'rtcdet_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        ## Head
        'det_head': {'name': 'decoupled_head',
                     'num_cls_head': 2,
                     'num_reg_head': 2,
                     'head_act': 'silu',
                     'head_norm': 'BN',
                     'head_depthwise': False,  
                     },
        'seg_head': {'name': None,
                     },
        'pos_head': {'name': None,
                     },
        # ---------------- Train config ----------------
        ## input
        'multi_scale': [0.5, 1.25],   # 320 -> 800
        'trans_type': 'yolox_n',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': "aligned_simota",
        'matcher_hpy': {'soft_center_radius': 3.0,
                        'topk_candidates': 13},
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_cls_weight': 1.0,
        'loss_box_weight': 2.0,
        # ---------------- Train config ----------------
        'trainer_type': 'rtcdet',
    },

}


# ------------------- Det + Seg + Pos task -------------------
rtcdet_seg_pos_cfg = {
    'rtcdet_seg_pos_n':{
        # ---------------- Model config ----------------
        ## Backbone
        'bk_pretrained': True,
        'bk_pretrained_mae': False,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'width': 0.25,
        'depth': 0.34,
        'ratio': 2.0,
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
        'fpn': 'rtcdet_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        ## Head
        'det_head': {'name': 'decoupled_head',
                     'num_cls_head': 2,
                     'num_reg_head': 2,
                     'head_act': 'silu',
                     'head_norm': 'BN',
                     'head_depthwise': False,  
                     },
        'seg_head': {'name': None,
                     },
        'pos_head': {'name': None,
                     },
        # ---------------- Train config ----------------
        ## input
        'multi_scale': [0.5, 1.25],   # 320 -> 800
        'trans_type': 'yolox_n',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': "aligned_simota",
        'matcher_hpy': {'soft_center_radius': 3.0,
                        'topk_candidates': 13},
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_cls_weight': 1.0,
        'loss_box_weight': 2.0,
        # ---------------- Train config ----------------
        'trainer_type': 'rtcdet',
    },

}
