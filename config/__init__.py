# ------------------------ Dataset Config ------------------------
from .data_config.dataset_config import dataset_cfg


def build_dataset_config(args):
    if args.dataset in ['coco', 'coco-val', 'coco-test']:
        cfg = dataset_cfg['coco']
    else:
        cfg = dataset_cfg[args.dataset]

    print('==============================')
    print('Dataset Config: {} \n'.format(cfg))

    return cfg


# ------------------ Transform Config ----------------------
from .data_config.transform_config import (
    # YOLOv5-Style
    yolov5_pico_trans_config,
    yolov5_nano_trans_config,
    yolov5_small_trans_config,
    yolov5_medium_trans_config,
    yolov5_large_trans_config,
    yolov5_huge_trans_config,
    # YOLOX-Style
    yolox_pico_trans_config,
    yolox_nano_trans_config,
    yolox_small_trans_config,
    yolox_medium_trans_config,
    yolox_large_trans_config,
    yolox_huge_trans_config,
    # SSD-Style
    ssd_trans_config,
    # RTMDet-v1-Style
    rtcdet_v1_pico_trans_config,
    rtcdet_v1_nano_trans_config,
    rtcdet_v1_small_trans_config,
    rtcdet_v1_medium_trans_config,
    rtcdet_v1_large_trans_config,
    rtcdet_v1_huge_trans_config,
)

def build_trans_config(trans_config='ssd'):
    print('==============================')
    print('Transform: {}-Style ...'.format(trans_config))
   
    # SSD-style transform 
    if trans_config == 'ssd':
        cfg = ssd_trans_config

    # YOLOv5-style transform 
    elif trans_config == 'yolov5_pico':
        cfg = yolov5_pico_trans_config
    elif trans_config == 'yolov5_nano':
        cfg = yolov5_nano_trans_config
    elif trans_config == 'yolov5_small':
        cfg = yolov5_small_trans_config
    elif trans_config == 'yolov5_medium':
        cfg = yolov5_medium_trans_config
    elif trans_config == 'yolov5_large':
        cfg = yolov5_large_trans_config
    elif trans_config == 'yolov5_huge':
        cfg = yolov5_huge_trans_config
        
    # YOLOX-style transform 
    elif trans_config == 'yolox_pico':
        cfg = yolox_pico_trans_config
    elif trans_config == 'yolox_nano':
        cfg = yolox_nano_trans_config
    elif trans_config == 'yolox_small':
        cfg = yolox_small_trans_config
    elif trans_config == 'yolox_medium':
        cfg = yolox_medium_trans_config
    elif trans_config == 'yolox_large':
        cfg = yolox_large_trans_config
    elif trans_config == 'yolox_huge':
        cfg = yolox_huge_trans_config

    # RTMDetv1-style transform 
    elif trans_config == 'rtcdet_v1_pico':
        cfg = rtcdet_v1_pico_trans_config
    elif trans_config == 'rtcdet_v1_nano':
        cfg = rtcdet_v1_nano_trans_config
    elif trans_config == 'rtcdet_v1_small':
        cfg = rtcdet_v1_small_trans_config
    elif trans_config == 'rtcdet_v1_medium':
        cfg = rtcdet_v1_medium_trans_config
    elif trans_config == 'rtcdet_v1_large':
        cfg = rtcdet_v1_large_trans_config
    elif trans_config == 'rtcdet_v1_huge':
        cfg = rtcdet_v1_huge_trans_config

    print('Transform Config: {} \n'.format(cfg))

    return cfg


# ------------------ Model Config ----------------------
## YOLO series
from .model_config.yolov1_config import yolov1_cfg
from .model_config.yolov2_config import yolov2_cfg
from .model_config.yolov3_config import yolov3_cfg
from .model_config.yolov4_config import yolov4_cfg
from .model_config.yolov5_config import yolov5_cfg
from .model_config.yolov7_config import yolov7_cfg
from .model_config.yolox_config import yolox_cfg
## My RTMDet series
from .model_config.rtcdet_v1_config import rtcdet_v1_cfg
from .model_config.rtcdet_v2_config import rtcdet_v2_cfg


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
    elif args.model in ['yolov3', 'yolov3_tiny']:
        cfg = yolov3_cfg[args.model]
    # YOLOv4
    elif args.model in ['yolov4', 'yolov4_tiny']:
        cfg = yolov4_cfg[args.model]
    # YOLOv5
    elif args.model in ['yolov5_n', 'yolov5_s', 'yolov5_m', 'yolov5_l', 'yolov5_x']:
        cfg = yolov5_cfg[args.model]
    # YOLOv7
    elif args.model in ['yolov7_tiny', 'yolov7', 'yolov7_x']:
        cfg = yolov7_cfg[args.model]
    # YOLOX
    elif args.model in ['yolox_n', 'yolox_s', 'yolox_m', 'yolox_l', 'yolox_x']:
        cfg = yolox_cfg[args.model]
    # My RTMDet-v1
    elif args.model in ['rtcdet_v1_p', 'rtcdet_v1_n', 'rtcdet_v1_t', 'rtcdet_v1_s', 'rtcdet_v1_m', 'rtcdet_v1_l', 'rtcdet_v1_x']:
        cfg = rtcdet_v1_cfg[args.model]
    # My RTMDet-v2
    elif args.model in ['rtcdet_v2_p', 'rtcdet_v2_n', 'rtcdet_v2_t', 'rtcdet_v2_s', 'rtcdet_v2_m', 'rtcdet_v2_l', 'rtcdet_v2_x']:
        cfg = rtcdet_v2_cfg[args.model]

    return cfg

