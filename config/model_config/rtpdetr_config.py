# Real-time Transformer-based Object Detector


# ------------------- Det task --------------------
rtpdetr_cfg = {
    'rtpdetr_r50':{
        # ---------------- Model config ----------------
        ## Model scale
        'width': 1.0,
        'depth': 1.0,
        'max_stride': 32,
        'out_stride': 16,
        # Image Encoder - Backbone
        'backbone': 'resnet50',
        'backbone_norm': 'FrozeBN',
        'pretrained': True,
        'freeze_at': 0,
        'freeze_stem_only': False,
        'hidden_dim': 256,
        'en_num_heads': 8,
        'en_num_layers': 1,
        'en_mlp_ratio': 4.0,
        'en_dropout': 0.0,
        'en_act': 'gelu',
        # Transformer Decoder
        'transformer': 'plain_detr_transformer',
        'de_num_heads': 8,
        'de_num_layers': 6,
        'de_mlp_ratio': 4.0,
        'de_dropout': 0.0,
        'de_act': 'gelu',
        'de_pre_norm': True,
        'rpe_hidden_dim': 512,
        'use_checkpoint': False,
        'proposal_feature_levels': 3,
        'proposal_tgt_strides': [8, 16, 32],
        'num_queries_one2one': 300,
        'num_queries_one2many': 10,
        # ---------------- Assignment config ----------------
        'matcher_hpy': {'cost_class': 2.0,
                        'cost_bbox': 1.0,
                        'cost_giou': 2.0,},
        # ---------------- Loss config ----------------
        'k_one2many': 6,
        'lambda_one2many': 1.0,
        'loss_coeff': {'class': 2,
                       'bbox': 1,
                       'giou': 2,},
        # ---------------- Train config ----------------
        ## input
        'multi_scale': [0.5, 1.25],   # 320 -> 800
        'trans_type': 'rtdetr_base',
        # ---------------- Train config ----------------
        'trainer_type': 'rtpdetr',
    },

}