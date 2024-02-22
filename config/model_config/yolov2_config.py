# YOLOv2 Config

yolov2_cfg = {
    # ---------------- Model config ----------------
    ## Backbone
    'backbone': 'darknet19',
    'pretrained': True,
    'stride': 32,  # P5
    'max_stride': 32,
    ## Neck
    'neck': 'sppf',
    'neck_act': 'lrelu',
    'neck_norm': 'BN',
    'neck_depthwise': False,
    'expand_ratio': 0.5,
    'pooling_size': 5,
    ## Head
    'head': 'decoupled_head',
    'head_act': 'lrelu',
    'head_norm': 'BN',
    'num_cls_head': 2,
    'num_reg_head': 2,
    'head_depthwise': False,
    'anchor_size': [[17,  25],
                    [55,  75],
                    [92,  206],
                    [202, 21],
                    [289, 311]],  # 416 scale
    # ---------------- Data process config ----------------
    ## Input
    'multi_scale': [0.5, 1.5], # 320 -> 960
    'trans_type': 'ssd',
    # ---------------- Matcher config ----------------
    'iou_thresh': 0.5,
    # ---------------- Loss config ----------------
    'loss_obj_weight': 1.0,
    'loss_cls_weight': 1.0,
    'loss_box_weight': 5.0,
    # ---------------- Trainer config ----------------
    'trainer_type': 'yolo',
}