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
        'fpn': 'yolovx_pafpn',
        'fpn_reduce_layer': 'conv',
        'fpn_downsample_layer': 'conv',
        'fpn_core_block': 'elanblock',
        'fpn_expand_ratio': 0.5,
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
        'trans_type': 'yolovx_nano',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': {'center_sampling_radius': 2.5,
                    'topk_candicate': 10},
        # ---------------- Loss config ----------------
        ## Loss weight
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        'loss_dfl_weight': 1.0,
        # ---------------- Train config ----------------
        'trainer_type': 'rtmdet',
    },

    'yolovx_t':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elannet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 0.375,
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
        'fpn': 'yolovx_pafpn',
        'fpn_reduce_layer': 'conv',
        'fpn_downsample_layer': 'conv',
        'fpn_core_block': 'elanblock',
        'fpn_expand_ratio': 0.5,
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
        'trans_type': 'yolovx_nano',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': {'center_sampling_radius': 2.5,
                    'topk_candicate': 10},
        # ---------------- Loss config ----------------
        ## Loss weight
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        'loss_dfl_weight': 1.0,
        # ---------------- Train config ----------------
        'trainer_type': 'rtmdet',
    },

    'yolovx_s':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elannet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 0.50,
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
        'fpn': 'yolovx_pafpn',
        'fpn_reduce_layer': 'conv',
        'fpn_downsample_layer': 'conv',
        'fpn_core_block': 'elanblock',
        'fpn_expand_ratio': 0.5,
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
        'trans_type': 'yolovx_small',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': {'center_sampling_radius': 2.5,
                    'topk_candicate': 10},
        # ---------------- Loss config ----------------
        ## Loss weight
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        'loss_dfl_weight': 1.0,
        # ---------------- Train config ----------------
        'trainer_type': 'rtmdet',
    },

    'yolovx_m':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elannet',
        'pretrained': False,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 0.75,
        'depth': 0.67,
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
        'fpn': 'yolovx_pafpn',
        'fpn_reduce_layer': 'conv',
        'fpn_downsample_layer': 'conv',
        'fpn_core_block': 'elanblock',
        'fpn_expand_ratio': 0.5,
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
        'trans_type': 'yolovx_medium',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': {'center_sampling_radius': 2.5,
                    'topk_candicate': 10},
        # ---------------- Loss config ----------------
        ## Loss weight
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        'loss_dfl_weight': 1.0,
        # ---------------- Train config ----------------
        'trainer_type': 'rtmdet',
    },

    'yolovx_l':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elannet',
        'pretrained': False,
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
        'fpn': 'yolovx_pafpn',
        'fpn_reduce_layer': 'conv',
        'fpn_downsample_layer': 'conv',
        'fpn_core_block': 'elanblock',
        'fpn_expand_ratio': 0.5,
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
        'trans_type': 'yolovx_large',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': {'center_sampling_radius': 2.5,
                    'topk_candicate': 10},
        # ---------------- Loss config ----------------
        ## Loss weight
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        'loss_dfl_weight': 1.0,
        # ---------------- Train config ----------------
        'trainer_type': 'rtmdet',
    },

    'yolovx_x':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elannet',
        'pretrained': False,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 1.25,
        'depth': 1.34,
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
        'fpn': 'yolovx_pafpn',
        'fpn_reduce_layer': 'conv',
        'fpn_downsample_layer': 'conv',
        'fpn_core_block': 'elanblock',
        'fpn_expand_ratio': 0.5,
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
        'trans_type': 'yolovx_huge',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': {'center_sampling_radius': 2.5,
                    'topk_candicate': 10},
        # ---------------- Loss config ----------------
        ## Loss weight
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        'loss_dfl_weight': 1.0,
        # ---------------- Train config ----------------
        'trainer_type': 'rtmdet',
    },

}