# ------------------ Model Config ----------------------
from .yolov1_config import yolov1_cfg
from .yolov2_config import yolov2_cfg
from .yolov3_config import yolov3_cfg
from .yolov4_config import yolov4_cfg
from .yolov5_config import yolov5_cfg
from .yolov7_config import yolov7_cfg
from .yolox_config import yolox_cfg


def build_model_config(args):
    print('==============================')
    print('Model: {} ...'.format(args.model.upper()))
    # YOLOv1
    if args.model == 'yolov1':
        cfg = yolov1_cfg
    # YOLOv2
    elif args.model == 'yolov2':
        cfg = yolov2_cfg
    # YOLOv3
    elif args.model in ['yolov3', 'yolov3_t']:
        cfg = yolov3_cfg[args.model]
    # YOLOv4
    elif args.model in ['yolov4', 'yolov4_t']:
        cfg = yolov4_cfg[args.model]
    # YOLOv5
    elif args.model in ['yolov5_n', 'yolov5_s', 'yolov5_m', 'yolov5_l', 'yolov5_x']:
        cfg = yolov5_cfg[args.model]
    # YOLOv7
    elif args.model in ['yolov7_t', 'yolov7_l', 'yolov7_x']:
        cfg = yolov7_cfg[args.model]
    # YOLOX
    elif args.model in ['yolox_n', 'yolox_s', 'yolox_m', 'yolox_l', 'yolox_x']:
        cfg = yolox_cfg[args.model]

    return cfg


# ------------------ Transform Config ----------------------
from .transform_config import (
    yolov5_nano_trans_config,
    yolov5_tiny_trans_config,
    yolov5_small_trans_config,
    yolov5_medium_trans_config,
    yolov5_large_trans_config,
    yolov5_huge_trans_config,
    ssd_trans_config
)

def build_trans_config(trans_config='ssd'):
    print('==============================')
    print('Transform: {}-Style ...'.format(trans_config))
    # SSD-style transform 
    if trans_config == 'ssd':
        cfg = ssd_trans_config

    # YOLOv5-style transform 
    elif trans_config == 'yolov5_nano':
        cfg = yolov5_nano_trans_config
    elif trans_config == 'yolov5_tiny':
        cfg = yolov5_tiny_trans_config
    elif trans_config == 'yolov5_small':
        cfg = yolov5_small_trans_config
    elif trans_config == 'yolov5_medium':
        cfg = yolov5_medium_trans_config
    elif trans_config == 'yolov5_large':
        cfg = yolov5_large_trans_config
    elif trans_config == 'yolov5_huge':
        cfg = yolov5_huge_trans_config
        
    return cfg