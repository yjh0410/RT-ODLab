# Low-computation Object Detector Config


lowdet_cfg = {
    # ---------------- Model config ----------------
    ## Backbone
    'backbone': 'smnet',
    'pretrained': True,
    'bk_act': 'silu',
    'bk_norm': 'BN',
    'bk_dpw': True,
    'bk_num_heads': 4,
    'stride': [8, 16, 32],  # P3, P4, P5
    'max_stride': 32,
    ## Neck: SPP
    'neck': 'sppf',
    'neck_expand_ratio': 0.5,
    'pooling_size': 5,
    'neck_act': 'silu',
    'neck_norm': 'BN',
    'neck_depthwise': True,
    ## Neck: PaFPN
    'fpn': 'lowdet_pafpn',
    'fpn_core_block': 'smblock',
    'fpn_reduce_layer': 'conv',
    'fpn_downsample_layer': 'conv',
    'fpn_nblocks': 1,
    'fpn_num_heads': 4,
    'fpn_shortcut': False,
    'fpn_act': 'silu',
    'fpn_norm': 'BN',
    'fpn_depthwise': True,
    ## Head
    'head': 'decoupled_head',
    'head_act': 'silu',
    'head_norm': 'BN',
    'head_dim': 96,
    'num_cls_head': 2,
    'num_reg_head': 2,
    'head_depthwise': True,
    'reg_max': 16,
    # ---------------- Train config ----------------
    ## Input
    'multi_scale': [0.5, 1.5],   # 320 -> 960
    'trans_type': 'yolovx_pico',
    # ---------------- Assignment config ----------------
    ## Matcher
    'matcher': {'center_sampling_radius': 2.5,
                'topk_candicate': 10},
    # ---------------- Loss config ----------------
    ## Loss weight
    'loss_cls_weight': 1.0,
    'loss_box_weight': 5.0,
    'loss_dfl_weight': 1.0,
    # ---------------- Train config ----------------
    'trainer_type': 'rtmdet',
}