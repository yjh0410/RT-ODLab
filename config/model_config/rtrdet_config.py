# Real-time Detection with Transformer


rtrdet_cfg = {
    'rtrdet_l':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elannet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'width': 1.0,
        'depth': 1.0,
        'max_stride': 32,
        'out_stride': 16,
        'd_model': 512,
        ## Transformer Encoder
        'transformer': 'RTRDetTransformer',
        'num_encoder': 1,
        'encoder_num_head': 8,
        'encoder_mlp_ratio': 4.0,
        'encoder_dropout': 0.1,
        'neck_depthwise': False,
        'encoder_act': 'relu',
        ## Transformer Decoder
        'num_decoder': 6,
        'stop_layer_id': -1,
        'decoder_num_head': 8,
        'decoder_mlp_ratio': 4.0,
        'decoder_dropout': 0.1,
        'decoder_act': 'relu',
        'decoder_num_queries': 100,
        'decoder_num_pattern': 3,
        'spatial_prior': 'learned',  # 'learned', 'grid'
        'num_topk': 100,
        # ---------------- Train config ----------------
        ## Input
        'multi_scale': [0.5, 1.0], # 320 -> 640
        'trans_type': 'rtrdet_large',
        # ---------------- Assignment config ----------------
        ## Matcher
        'matcher': "hungarian_matcher",
        'matcher_hpy': {"hungarian_matcher": {'cost_cls_weight':  2.0,
                                              'cost_box_weight':  5.0,
                                              'cost_giou_weight': 2.0,
                                              },
                        },
        # ---------------- Loss config ----------------
        ## Loss weight
        'ema_update': False,
        'loss_weights': {"hungarian_matcher": {'loss_cls_weight':  1.0,
                                               'loss_box_weight':  5.0,
                                               'loss_giou_weight': 2.0},
                        },
        # ---------------- Train config ----------------
        'trainer_type': 'rtrdet',
    },

}
