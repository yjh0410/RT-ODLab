# yolov8-v2 Config


rtcdet_cfg = {
    'rtcdet_s':{
        # ---------------- Model config ----------------
        ## Backbone
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'width': 0.50,
        'depth': 0.34,
        'ratio': 2.0,
        'stride': [8, 16, 32],  # P3, P4, P5
        'max_stride': 32,
        'reg_max': 16,
        ## Neck: SPP
        'neck': 'sppf',
        'neck_expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        ## Neck: PaFPN
        'fpn': 'rtc_pafpn',
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
        'multi_scale': [0.5, 1.5], # 320 -> 960
        'trans_type': 'yolov5_s',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': "simota",
        'matcher_hpy': {'center_sampling_radius': 2.5,
                        'topk_candidate': 10,
                        },
        # ---------------- Loss config ----------------
        'loss_cls_weight': 0.5,
        'loss_box_weight': 7.5,
        'loss_dfl_weight': 1.5,
        # ---------------- Train config ----------------
        'trainer_type': 'rtcdet',
    },

}