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
        'max_stride': 32,
        ## FPN
        'fpn': 'yolox_pafpn',
        'fpn_reduce_layer': 'conv',
        'fpn_downsample_layer': 'conv',
        'fpn_core_block': 'cspblock',
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
        'multi_scale': [0.7, 1.25],   # 448 -> 800
        'trans_type': 'yolox_nano',
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
        'trainer_type': 'yolox',
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
        'max_stride': 32,
        ## FPN
        'fpn': 'yolox_pafpn',
        'fpn_reduce_layer': 'conv',
        'fpn_downsample_layer': 'conv',
        'fpn_core_block': 'cspblock',
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
        'multi_scale': [0.7, 1.25],   # 448 -> 800
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
        'trainer_type': 'yolox',
    },

    'yolox_m':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'cspdarknet',
        'pretrained': False,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 0.75,
        'depth': 0.67,
        'stride': [8, 16, 32],  # P3, P4, P5
        'max_stride': 32,
        ## FPN
        'fpn': 'yolox_pafpn',
        'fpn_reduce_layer': 'conv',
        'fpn_downsample_layer': 'conv',
        'fpn_core_block': 'cspblock',
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
        'multi_scale': [0.7, 1.25],   # 448 -> 800
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
        'trainer_type': 'yolox',
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
        'max_stride': 32,
        ## FPN
        'fpn': 'yolox_pafpn',
        'fpn_reduce_layer': 'conv',
        'fpn_downsample_layer': 'conv',
        'fpn_core_block': 'cspblock',
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
        'multi_scale': [0.7, 1.25],   # 448 -> 800
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
        'trainer_type': 'yolox',
    },

    'yolox_x':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'cspdarknet',
        'pretrained': False,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 1.25,
        'depth': 1.34,
        'stride': [8, 16, 32],  # P3, P4, P5
        'max_stride': 32,
        ## FPN
        'fpn': 'yolox_pafpn',
        'fpn_reduce_layer': 'conv',
        'fpn_downsample_layer': 'conv',
        'fpn_core_block': 'cspblock',
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
        'multi_scale': [0.7, 1.25],   # 448 -> 800
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
        'trainer_type': 'yolox',
    },

}