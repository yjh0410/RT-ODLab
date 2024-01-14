# Enhanced CenterNet


ctrnet_cfg = {
    'ctrnet_n':{
        # ---------------- Model config ----------------
        ## Backbone
        'bk_pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'width': 0.25,
        'depth': 0.34,
        'ratio': 2.0,
        'max_stride': 32,
        'out_stride': 4,
        ## Neck
        'neck': 'sppf',
        'neck_expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        ## Decoder
        'dec_act': 'silu',
        'dec_norm': 'BN',
        'dec_depthwise': False,
        ## Head
        'head': 'decoupled_head',
        'num_cls_head': 4,
        'num_reg_head': 4,
        'head_act': 'silu',
        'head_norm': 'BN',
        'head_depthwise': False,  
        # ---------------- Train config ----------------
        ## input
        'multi_scale': [0.5, 1.25],   # 320 -> 800
        'trans_type': 'yolox_n',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': "aligned_simota",
        'matcher_hpy': {'main' : {'soft_center_radius': 3.0,
                                  'topk_candidates': 1},   # one-to-one assignment
                        'aux'  : {'soft_center_radius': 3.0,
                                  'topk_candidates': 13},  # one-to-many assignment
                                  },
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_cls_weight': 1.0,
        'loss_box_weight': 2.0,
        # ---------------- Train config ----------------
        'trainer_type': 'rtcdet',
    },

}

