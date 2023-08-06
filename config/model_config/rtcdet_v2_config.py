# RTCDet-v2 Config


rtcdet_v2_cfg = {
    'rtcdet_v2_l':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elannet_v2',
        'pretrained': False,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'width': 1.0,
        'depth': 1.0,
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
        'fpn': 'rtcdet_pafpn',
        'fpn_reduce_layer': 'conv',
        'fpn_downsample_layer': 'conv',
        'fpn_core_block': 'elan_block',
        'fpn_squeeze_ratio': 0.25,
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
        'trans_type': 'rtcdet_v1_large',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': {'tal': {'topk': 10,
                            'alpha': 0.5,
                            'beta': 6.0},
                    'ota': {'center_sampling_radius': 2.5,
                             'topk_candidate': 10},
                    },
        # ---------------- Loss config ----------------
        ## Loss weight
        'ema_update': False,
        'loss_cls_weight': {'tal': 0.5, 'ota': 1.0},
        'loss_box_weight': {'tal': 7.0, 'ota': 5.0},
        'loss_dfl_weight': {'tal': 1.5, 'ota': 1.0},
        # ---------------- Train config ----------------
        'trainer_type': 'rtmdet',
    },

}