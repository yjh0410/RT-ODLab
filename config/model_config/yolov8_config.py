# yolov8-v2 Config


yolov8_cfg = {
    'yolov8_n':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'yolov8',
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'width': 0.25,
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
        'fpn': 'yolov8_pafpn',
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
        'trans_type': 'yolo_n',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': "tal",
        'matcher_hpy': {'topk_candidates': 10,
                        'alpha': 0.5,
                        'beta':  6.0},
        # ---------------- Loss config ----------------
        'loss_cls_weight': 0.5,
        'loss_box_weight': 7.5,
        'loss_dfl_weight': 1.5,
        # ---------------- Train config ----------------
        'trainer_type': 'yolo',
    },

    'yolov8_s':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'yolov8',
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
        'fpn': 'yolov8_pafpn',
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
        'trans_type': 'yolo_s',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': "tal",
        'matcher_hpy': {'topk_candidates': 10,
                        'alpha': 0.5,
                        'beta':  6.0},
        # ---------------- Loss config ----------------
        'loss_cls_weight': 0.5,
        'loss_box_weight': 7.5,
        'loss_dfl_weight': 1.5,
        # ---------------- Train config ----------------
        'trainer_type': 'yolo',
    },

    'yolov8_m':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'yolov8',
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'width': 0.75,
        'depth': 0.67,
        'ratio': 1.5,
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
        'fpn': 'yolov8_pafpn',
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
        'trans_type': 'yolo_m',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': "tal",
        'matcher_hpy': {'topk_candidates': 10,
                        'alpha': 0.5,
                        'beta':  6.0},
        # ---------------- Loss config ----------------
        'loss_cls_weight': 0.5,
        'loss_box_weight': 7.5,
        'loss_dfl_weight': 1.5,
        # ---------------- Train config ----------------
        'trainer_type': 'yolo',
    },

    'yolov8_l':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'yolov8',
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'width': 1.0,
        'depth': 1.0,
        'ratio': 1.0,
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
        'fpn': 'yolov8_pafpn',
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
        'trans_type': 'yolo_l',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': "tal",
        'matcher_hpy': {'topk_candidates': 10,
                        'alpha': 0.5,
                        'beta':  6.0},
        # ---------------- Loss config ----------------
        'loss_cls_weight': 0.5,
        'loss_box_weight': 7.5,
        'loss_dfl_weight': 1.5,
        # ---------------- Train config ----------------
        'trainer_type': 'yolo',
    },

    'yolov8_x':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'yolov8',
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'width': 1.25,
        'depth': 1.0,
        'ratio': 1.0,
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
        'fpn': 'yolov8_pafpn',
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
        'trans_type': 'yolo_x',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': "tal",
        'matcher_hpy': {'topk_candidates': 10,
                        'alpha': 0.5,
                        'beta':  6.0},
        # ---------------- Loss config ----------------
        'loss_cls_weight': 0.5,
        'loss_box_weight': 7.5,
        'loss_dfl_weight': 1.5,
        # ---------------- Train config ----------------
        'trainer_type': 'yolo',
    },

}