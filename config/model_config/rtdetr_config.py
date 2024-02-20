# Real-time Transformer-based Object Detector


# ------------------- Det task --------------------
rtdetr_cfg = {
    'rtdetr_r18':{
        # ---------------- Model config ----------------
        ## Image Encoder - Backbone
        'backbone': 'resnet18',
        'backbone_norm': 'BN',
        'pretrained': True,
        'pretrained_weight': 'imagenet1k_v1',
        'freeze_at': 0,
        'freeze_stem_only': False,
        'out_stride': [8, 16, 32],
        'max_stride': 32,
        ## Image Encoder - FPN
        'fpn': 'hybrid_encoder',
        'fpn_num_blocks': 3,
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        'hidden_dim': 256,
        'en_num_heads': 8,
        'en_num_layers': 1,
        'en_ffn_dim': 1024,
        'en_dropout': 0.0,
        'pe_temperature': 10000.,
        'en_act': 'gelu',
        # Transformer Decoder
        'transformer': 'rtdetr_transformer',
        'de_num_heads': 8,
        'de_num_layers': 3,
        'de_ffn_dim': 1024,
        'de_dropout': 0.0,
        'de_act': 'relu',
        'de_num_points': 4,
        'num_queries': 300,
        'learnt_init_query': False,
        'pe_temperature': 10000.,
        'dn_num_denoising': 100,
        'dn_label_noise_ratio': 0.5,
        'dn_box_noise_scale': 1,
        # ---------------- Assignment config ----------------
        'matcher_hpy': {'cost_class': 2.0,
                        'cost_bbox': 5.0,
                        'cost_giou': 2.0,},
        # ---------------- Loss config ----------------
        'use_vfl': True,
        'loss_coeff': {'class': 1,
                       'bbox': 5,
                       'giou': 2,},
        # ---------------- Train config ----------------
        ## input
        'multi_scale': [0.5, 1.25],   # 320 -> 800
        'trans_type': 'rtdetr_base',
        # ---------------- Train config ----------------
        'trainer_type': 'rtdetr',
    },

    'rtdetr_r50':{
        # ---------------- Model config ----------------
        ## Image Encoder - Backbone
        'backbone': 'resnet50',
        'backbone_norm': 'FrozeBN',
        'pretrained': True,
        'pretrained_weight': 'imagenet1k_v2',
        'freeze_at': 0,
        'freeze_stem_only': False,
        'out_stride': [8, 16, 32],
        'max_stride': 32,
        ## Image Encoder - FPN
        'fpn': 'hybrid_encoder',
        'fpn_num_blocks': 3,
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        'hidden_dim': 256,
        'en_num_heads': 8,
        'en_num_layers': 1,
        'en_ffn_dim': 2048,
        'en_dropout': 0.0,
        'pe_temperature': 10000.,
        'en_act': 'gelu',
        # Transformer Decoder
        'transformer': 'rtdetr_transformer',
        'de_num_heads': 8,
        'de_num_layers': 6,
        'de_ffn_dim': 2048,
        'de_dropout': 0.0,
        'de_act': 'relu',
        'de_num_points': 4,
        'num_queries': 300,
        'learnt_init_query': False,
        'pe_temperature': 10000.,
        'dn_num_denoising': 100,
        'dn_label_noise_ratio': 0.5,
        'dn_box_noise_scale': 1,
        # Head
        'det_head': 'dino_head',
        # ---------------- Assignment config ----------------
        'matcher_hpy': {'cost_class': 2.0,
                        'cost_bbox': 5.0,
                        'cost_giou': 2.0,},
        # ---------------- Loss config ----------------
        'use_vfl': True,
        'loss_coeff': {'class': 1,
                       'bbox': 5,
                       'giou': 2,},
        # ---------------- Train config ----------------
        ## input
        'multi_scale': [0.5, 1.25],   # 320 -> 800
        'trans_type': 'rtdetr_base',
        # ---------------- Train config ----------------
        'trainer_type': 'rtdetr',
    },

    'rtdetr_r101':{
        # ---------------- Model config ----------------
        ## Image Encoder - Backbone
        'backbone': 'resnet101',
        'backbone_norm': 'FrozeBN',
        'pretrained': True,
        'pretrained_weight': 'imagenet1k_v2',
        'freeze_at': 0,
        'freeze_stem_only': False,
        'out_stride': [8, 16, 32],
        'max_stride': 32,
        ## Image Encoder - FPN
        'fpn': 'hybrid_encoder',
        'fpn_num_blocks': 4,
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        'hidden_dim': 384,
        'en_num_heads': 8,
        'en_num_layers': 1,
        'en_ffn_dim': 2048,
        'en_dropout': 0.0,
        'pe_temperature': 10000.,
        'en_act': 'gelu',
        # Transformer Decoder
        'transformer': 'rtdetr_transformer',
        'de_num_heads': 8,
        'de_num_layers': 6,
        'de_ffn_dim': 2048,
        'de_dropout': 0.0,
        'de_act': 'relu',
        'de_num_points': 4,
        'num_queries': 300,
        'learnt_init_query': False,
        'pe_temperature': 10000.,
        'dn_num_denoising': 100,
        'dn_label_noise_ratio': 0.5,
        'dn_box_noise_scale': 1,
        # Head
        'det_head': 'dino_head',
        # ---------------- Assignment config ----------------
        'matcher_hpy': {'cost_class': 2.0,
                        'cost_bbox': 5.0,
                        'cost_giou': 2.0,},
        # ---------------- Loss config ----------------
        'use_vfl': True,
        'loss_coeff': {'class': 1,
                       'bbox': 5,
                       'giou': 2,},
        # ---------------- Train config ----------------
        ## input
        'multi_scale': [0.5, 1.25],   # 320 -> 800
        'trans_type': 'rtdetr_base',
        # ---------------- Train config ----------------
        'trainer_type': 'rtdetr',
    },

}