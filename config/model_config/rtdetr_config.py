# yolo-free config


rtdetr_cfg = {
    # P5
    'rtdetr_n': {
        # ---------------- Model config ----------------
        ## ------- Image Encoder -------
        ### CNN-Backbone
        'backbone': 'elannet',
        'pretrained': False,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 0.25,
        'depth': 0.34,
        'stride': [8, 16, 32],  # P3, P4, P5
        'max_stride': 32,
        ### CNN-Neck
        'neck': 'sppf',
        'neck_expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        ### CNN-CSFM
        'fpn': 'yolovx_pafpn',
        'fpn_reduce_layer': 'conv',
        'fpn_downsample_layer': 'conv',
        'fpn_core_block': 'elanblock',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        ## ------- Memory Decoder -------
        'num_compressed': 300,
        'com_dim_feedforward': 1024,
        'com_num_heads': 8,
        'com_dropout': 0.1,
        'com_act': 'silu',
        ## ------- Transformer Decoder -------
        'd_model': 256,
        'attn_type': 'mhsa',
        'num_decoder_layers': 6,
        'num_queries': 300,
        'de_dim_feedforward': 1024,
        'de_num_heads': 8,
        'de_dropout': 0.1,
        'de_act': 'silu',
        'de_norm': 'LN',
        # ---------------- Train config ----------------
        ## input
        'multi_scale': [0.5, 1.0],   # 320 -> 640
        'trans_type': 'yolov5_nano',
        # ---------------- Assignment config ----------------
        ## matcher
        'set_cost_class': 2.0,
        'set_cost_bbox': 5.0,
        'set_cost_giou': 2.0,
        # ---------------- Loss config ----------------
        ## loss weight
        'focal_alpha': 0.25,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        'loss_giou_weight': 2.0,
        # ---------------- Train config ----------------
        'trainer_type': 'detr',
        },

}