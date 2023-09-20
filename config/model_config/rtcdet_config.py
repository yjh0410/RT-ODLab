# RTCDet-v2 Config


rtcdet_cfg = {
    'rtcdet_p':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elannet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': True,
        'width': 0.25,
        'depth': 0.34,
        'stride': [8, 16, 32],  # P3, P4, P5
        'max_stride': 32,
        'reg_max': 16,
        ## Neck: SPP
        'neck': 'sppf',
        'neck_expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': True,
        ## Neck: PaFPN
        'fpn': 'rtcdet_pafpn',
        'fpn_reduce_layer': 'conv',
        'fpn_downsample_layer': 'conv',
        'fpn_core_block': 'elanblock',
        'fpn_branch_depth': 3,
        'fpn_expand_ratio': 0.5,
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': True,
        ## Head
        'head': 'decoupled_head',
        'head_act': 'silu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': True,
        # ---------------- Train config ----------------
        ## Input
        'multi_scale': [0.4, 1.0], # 256 -> 640
        'trans_type': 'yolox_pico',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': "simota",
        'matcher_hpy': {"simota": {'center_sampling_radius': 2.5,
                                   'topk_candidate': 10},
                        "aligned_simota": {'soft_center_radius': 3.0,
                                           'topk_candicate': 10,
                                           'iou_weight': 3.0},
                                           },
        # ---------------- Loss config ----------------
        ## Loss weight
        'ema_update': False,
        'loss_box_aux': True,
        'loss_weights': {"simota": {'loss_cls_weight': 1.0,
                                    'loss_dfl_weight': 1.0,
                                    'loss_box_weight': 5.0},
                         "aligned_simota": {'loss_cls_weight': 1.0,
                                            'loss_dfl_weight': 1.0,
                                            'loss_box_weight': 2.0}
                                            },
        # ---------------- Train config ----------------
        'trainer_type': 'rtcdet',
    },

    'rtcdet_n':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elannet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'width': 0.25,
        'depth': 0.34,
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
        'fpn': 'rtcdet_pafpn',
        'fpn_reduce_layer': 'conv',
        'fpn_downsample_layer': 'conv',
        'fpn_core_block': 'elanblock',
        'fpn_branch_depth': 3,
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
        # ---------------- Train config ----------------
        ## Input
        'multi_scale': [0.5, 1.5], # 320 -> 960
        'trans_type': 'yolox_nano',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': "simota",
        'matcher_hpy': {"simota": {'center_sampling_radius': 2.5,
                                   'topk_candidate': 10},
                        "aligned_simota": {'soft_center_radius': 3.0,
                                           'topk_candicate': 10,
                                           'iou_weight': 3.0},
                                           },
        # ---------------- Loss config ----------------
        ## Loss weight
        'ema_update': False,
        'loss_box_aux': True,
        'loss_weights': {"simota": {'loss_cls_weight': 1.0,
                                    'loss_dfl_weight': 1.0,
                                    'loss_box_weight': 5.0},
                         "aligned_simota": {'loss_cls_weight': 1.0,
                                            'loss_dfl_weight': 1.0,
                                            'loss_box_weight': 2.0}
                                            },
        # ---------------- Train config ----------------
        'trainer_type': 'rtcdet',
    },

    'rtcdet_t':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elannet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'width': 0.375,
        'depth': 0.34,
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
        'fpn': 'rtcdet_pafpn',
        'fpn_reduce_layer': 'conv',
        'fpn_downsample_layer': 'conv',
        'fpn_core_block': 'elanblock',
        'fpn_branch_depth': 3,
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
        # ---------------- Train config ----------------
        ## Input
        'multi_scale': [0.5, 1.5], # 320 -> 960
        'trans_type': 'yolox_small',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': "simota",
        'matcher_hpy': {"simota": {'center_sampling_radius': 2.5,
                                   'topk_candidate': 10},
                        "aligned_simota": {'soft_center_radius': 3.0,
                                           'topk_candicate': 10,
                                           'iou_weight': 3.0},
                                           },
        # ---------------- Loss config ----------------
        ## Loss weight
        'ema_update': False,
        'loss_box_aux': True,
        'loss_weights': {"simota": {'loss_cls_weight': 1.0,
                                    'loss_dfl_weight': 1.0,
                                    'loss_box_weight': 5.0},
                         "aligned_simota": {'loss_cls_weight': 1.0,
                                            'loss_dfl_weight': 1.0,
                                            'loss_box_weight': 2.0}
                                            },
        # ---------------- Train config ----------------
        'trainer_type': 'rtcdet',
    },

    'rtcdet_s':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elannet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'width': 0.50,
        'depth': 0.34,
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
        'fpn': 'rtcdet_pafpn',
        'fpn_reduce_layer': 'conv',
        'fpn_downsample_layer': 'conv',
        'fpn_core_block': 'elanblock',
        'fpn_branch_depth': 3,
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
        # ---------------- Train config ----------------
        ## Input
        'multi_scale': [0.5, 1.5], # 320 -> 960
        'trans_type': 'yolox_small',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': "simota",
        'matcher_hpy': {"simota": {'center_sampling_radius': 2.5,
                                   'topk_candidate': 10},
                        "aligned_simota": {'soft_center_radius': 3.0,
                                           'topk_candicate': 10,
                                           'iou_weight': 3.0},
                                           },
        # ---------------- Loss config ----------------
        ## Loss weight
        'ema_update': False,
        'loss_box_aux': True,
        'loss_weights': {"simota": {'loss_cls_weight': 1.0,
                                    'loss_dfl_weight': 1.0,
                                    'loss_box_weight': 5.0},
                         "aligned_simota": {'loss_cls_weight': 1.0,
                                            'loss_dfl_weight': 1.0,
                                            'loss_box_weight': 2.0}
                                            },
        # ---------------- Train config ----------------
        'trainer_type': 'rtcdet',
    },

    'rtcdet_m':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elannet',
        'pretrained': False,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'width': 0.75,
        'depth': 0.67,
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
        'fpn': 'rtcdet_pafpn',
        'fpn_reduce_layer': 'conv',
        'fpn_downsample_layer': 'conv',
        'fpn_core_block': 'elanblock',
        'fpn_branch_depth': 3,
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
        # ---------------- Train config ----------------
        ## Input
        'multi_scale': [0.5, 1.5], # 320 -> 960
        'trans_type': 'yolox_medium',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': "simota",
        'matcher_hpy': {"simota": {'center_sampling_radius': 2.5,
                                   'topk_candidate': 10},
                        "aligned_simota": {'soft_center_radius': 3.0,
                                           'topk_candicate': 10,
                                           'iou_weight': 3.0},
                                           },
        # ---------------- Loss config ----------------
        ## Loss weight
        'ema_update': False,
        'loss_box_aux': True,
        'loss_weights': {"simota": {'loss_cls_weight': 1.0,
                                    'loss_dfl_weight': 1.0,
                                    'loss_box_weight': 5.0},
                         "aligned_simota": {'loss_cls_weight': 1.0,
                                            'loss_dfl_weight': 1.0,
                                            'loss_box_weight': 2.0}
                                            },
        # ---------------- Train config ----------------
        'trainer_type': 'rtcdet',
    },

    'rtcdet_l':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elannet',
        'pretrained': False,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'width': 1.0,
        'depth': 1.0,
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
        'fpn': 'rtcdet_pafpn',
        'fpn_reduce_layer': 'conv',
        'fpn_downsample_layer': 'conv',
        'fpn_core_block': 'elanblock',
        'fpn_branch_depth': 3,
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
        # ---------------- Train config ----------------
        ## Input
        'multi_scale': [0.5, 1.5], # 320 -> 960
        'trans_type': 'yolox_large',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': "simota",
        'matcher_hpy': {"simota": {'center_sampling_radius': 2.5,
                                   'topk_candidate': 10},
                        "aligned_simota": {'soft_center_radius': 3.0,
                                           'topk_candicate': 10,
                                           'iou_weight': 3.0},
                                           },
        # ---------------- Loss config ----------------
        ## Loss weight
        'ema_update': False,
        'loss_box_aux': True,
        'loss_weights': {"simota": {'loss_cls_weight': 1.0,
                                    'loss_dfl_weight': 1.0,
                                    'loss_box_weight': 5.0},
                         "aligned_simota": {'loss_cls_weight': 1.0,
                                            'loss_dfl_weight': 1.0,
                                            'loss_box_weight': 2.0}
                                            },
        # ---------------- Train config ----------------
        'trainer_type': 'rtcdet',
    },

    'rtcdet_x':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elannet',
        'pretrained': False,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'width': 1.25,
        'depth': 1.34,
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
        'fpn': 'rtcdet_pafpn',
        'fpn_reduce_layer': 'conv',
        'fpn_downsample_layer': 'conv',
        'fpn_core_block': 'elanblock',
        'fpn_branch_depth': 3,
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
        # ---------------- Train config ----------------
        ## Input
        'multi_scale': [0.5, 1.5], # 320 -> 960
        'trans_type': 'yolox_huge',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': "simota",
        'matcher_hpy': {"simota": {'center_sampling_radius': 2.5,
                                   'topk_candidate': 10},
                        "aligned_simota": {'soft_center_radius': 3.0,
                                           'topk_candicate': 10,
                                           'iou_weight': 3.0},
                                           },
        # ---------------- Loss config ----------------
        ## Loss weight
        'ema_update': False,
        'loss_box_aux': True,
        'loss_weights': {"simota": {'loss_cls_weight': 1.0,
                                    'loss_dfl_weight': 1.0,
                                    'loss_box_weight': 5.0},
                         "aligned_simota": {'loss_cls_weight': 1.0,
                                            'loss_dfl_weight': 1.0,
                                            'loss_box_weight': 2.0}
                                            },
        # ---------------- Train config ----------------
        'trainer_type': 'rtcdet',
    },

}