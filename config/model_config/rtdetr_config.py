# Real-time Transformer-based Object Detector


# ------------------- Det task --------------------
rtdetr_cfg = {
    'rtdetr_r18':{
        # ---------------- Model config ----------------
        ## Model scale
        'width': 1.0,
        'depth': 1.0,
        ## Image Encoder - Backbone
        'backbone': 'resnet18',
        'backbone_norm': 'FrozeBN',
        'res5_dilation': False,
        'pretrained': True,
        'pretrained_weight': 'imagenet1k_v1',
        'freeze_stem_only': True,
        'out_stride': [8, 16, 32],
        ## Image Encoder - FPN
        'fpn': 'hybrid_encoder',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        'hidden_dim': 256,
        'en_num_heads': 8,
        'en_num_layers': 1,
        'en_mlp_ratio': 4.0,
        'en_dropout': 0.1,
        'pe_temperature': 10000.,
        'en_act': 'gelu',
        # Transformer Decoder
        'transformer': 'rtdetr_transformer',
        'hidden_dim': 256,
        'de_num_heads': 8,
        'de_num_layers': 3,
        'de_mlp_ratio': 4.0,
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