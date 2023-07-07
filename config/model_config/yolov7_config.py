# YOLOv7 Config

yolov7_cfg = {
    'yolov7_tiny':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elannet_tiny',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'stride': [8, 16, 32],  # P3, P4, P5
        'max_stride': 32,
        ## Neck
        'neck': 'csp_sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        ## FPN
        'fpn': 'yolov7_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        'nbranch': 2.0,       # number of branch in ELANBlockFPN
        'depth': 1.0,         # depth factor of each branch in ELANBlockFPN
        'width': 0.5,         # width factor of channel in FPN
        ## Head
        'head': 'decoupled_head',
        'head_act': 'silu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        # ---------------- Train config ----------------
        ## input
        'trans_type': 'yolov5_nano',
        'multi_scale': [0.5, 1.25], # 320 -> 800
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
        'trainer_type': 'yolo',
    },

    'yolov7':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elannet_large',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'stride': [8, 16, 32],  # P3, P4, P5
        'max_stride': 32,
        # neck
        'neck': 'csp_sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'yolov7_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        'nbranch': 4.0,       # number of branch in ELANBlockFPN
        'depth': 1.0,         # depth factor of each branch in ELANBlockFPN
        'width': 1.0,         # width factor of channel in FPN
        # head
        'head': 'decoupled_head',
        'head_act': 'silu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        # ---------------- Train config ----------------
        ## input
        'trans_type': 'yolov5_large',
        'multi_scale': [0.5, 1.25], # 320 -> 800
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
        'trainer_type': 'yolo',
    },

    'yolov7_x':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elannet_huge',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'stride': [8, 16, 32],  # P3, P4, P5
        'max_stride': 32,
        ## Neck
        'neck': 'csp_sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        ## FPN
        'fpn': 'yolov7_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        'nbranch': 4.0,        # number of branch in ELANBlockFPN
        'depth': 2.0,          # depth factor of each branch in ELANBlockFPN
        'width': 1.25,         # width factor of channel in FPN
        ## Head
        'head': 'decoupled_head',
        'head_act': 'silu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        # ---------------- Train config ----------------
        ## input
        'trans_type': 'yolov5_huge',
        'multi_scale': [0.5, 1.25], # 320 -> 640
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
        'trainer_type': 'yolo',
    },

}