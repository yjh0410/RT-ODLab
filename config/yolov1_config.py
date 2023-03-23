# YOLOv1 Config

yolov1_cfg = {
    # input
    'trans_type': 'ssd',
    # loss weight
    'loss_obj_weight': 1.0,
    'loss_cls_weight': 1.0,
    'loss_reg_weight': 5.0,
    # training configuration
    'no_aug_epoch': -1,
    # optimizer
    'optimizer': 'sgd',        # optional: sgd, adam, adamw
    'momentum': 0.937,         # SGD: 0.937;    AdamW: invalid
    'weight_decay': 5e-4,      # SGD: 5e-4;     AdamW: 5e-2
    'clip_grad': 10,           # SGD: 10.0;     AdamW: -1
    # model EMA
    'ema_decay': 0.9999,       # SGD: 0.9999;   AdamW: 0.9998
    'ema_tau': 2000,
    # lr schedule
    'scheduler': 'linear',
    'lr0': 0.01,              # SGD: 0.01;     AdamW: 0.004
    'lrf': 0.01,               # SGD: 0.01;     AdamW: 0.05
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
}