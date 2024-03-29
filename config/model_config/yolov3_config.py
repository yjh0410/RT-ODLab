# YOLOv3 Config

yolov3_cfg = {
    'yolov3':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'darknet53',
        'pretrained': True,
        'stride': [8, 16, 32],  # P3, P4, P5
        'width': 1.0,
        'depth': 1.0,
        'max_stride': 32,
        ## Neck
        'neck': 'sppf',
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        'expand_ratio': 0.5,
        'pooling_size': 5,
        ## FPN
        'fpn': 'yolov3_fpn',
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
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],     # P3
                        [30, 61],   [62, 45],   [59, 119],    # P4
                        [116, 90],  [156, 198], [373, 326]],  # P5
        # ---------------- Data process config ----------------
        'trans_type': 'yolo_l',
        'multi_scale': [0.5, 1.25],  # 320 -> 800
        # ---------------- Assignment config ----------------
        ## matcher
        'iou_thresh': 0.5,
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        # ---------------- Train config ----------------
        'trainer_type': 'yolo',
    },

    'yolov3_tiny':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'darknet_tiny',
        'pretrained': True,
        'stride': [8, 16, 32],  # P3, P4, P5
        'width': 0.25,
        'depth': 0.34,
        'max_stride': 32,
        ## Neck
        'neck': 'sppf',
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        'expand_ratio': 0.5,
        'pooling_size': 5,
        ## FPN
        'fpn': 'yolov3_fpn',
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
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],     # P3
                        [30, 61],   [62, 45],   [59, 119],    # P4
                        [116, 90],  [156, 198], [373, 326]],  # P5
        # ---------------- Data process config ----------------
        ## input
        'trans_type': 'yolo_n',
        'multi_scale': [0.5, 1.25],  # 320 -> 800
        # ---------------- Assignment config ----------------
        ## matcher
        'iou_thresh': 0.5,
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        # ---------------- Train config ----------------
        'trainer_type': 'yolo',
    },

}