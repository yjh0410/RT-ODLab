# YOLOv1 Config

yolov1_cfg = {
    # input
    'trans_type': 'ssd',
    # loss weight
    'loss_obj_weight': 1.0,
    'loss_cls_weight': 1.0,
    'loss_txty_weight': 1.0,
    'loss_twth_weight': 1.0,
    # training configuration
    'no_aug_epoch': -1,
    # optimizer
    'optimizer': 'sgd',        # optional: sgd, yolov5_sgd
    'momentum': 0.9,           # SGD: 0.937;    AdamW: invalid
    'weight_decay': 5e-4,      # SGD: 5e-4;     AdamW: 5e-2
    'clip_grad': 10,           # SGD: 10.0;     AdamW: -1
    # model EMA
    'ema_decay': 0.9999,       # SGD: 0.9999;   AdamW: 0.9998
    # lr schedule
    'scheduler': 'linear',
    'lr0': 0.001,              # SGD: 0.01;     AdamW: 0.004
    'lrf': 0.01,               # SGD: 0.01;     AdamW: 0.05
    # warmup strategy
    'warmup': 'linear',
    'warmup_factor': 0.00066667,
}