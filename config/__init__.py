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
    # YOLOvx-Style
    yolovx_pico_trans_config,
    yolovx_nano_trans_config,
    yolovx_small_trans_config,
    yolovx_medium_trans_config,
    yolovx_large_trans_config,
    yolovx_huge_trans_config,
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

    # YOLOvx-style transform 
    elif trans_config == 'yolovx_pico':
        cfg = yolovx_pico_trans_config
    elif trans_config == 'yolovx_nano':
        cfg = yolovx_nano_trans_config
    elif trans_config == 'yolovx_small':
        cfg = yolovx_small_trans_config
    elif trans_config == 'yolovx_medium':
        cfg = yolovx_medium_trans_config
    elif trans_config == 'yolovx_large':
        cfg = yolovx_large_trans_config
    elif trans_config == 'yolovx_huge':
        cfg = yolovx_huge_trans_config

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
from .model_config.yolovx_config import yolovx_cfg
from .model_config.yolox_config import yolox_cfg
## Real-Time DETR
from .model_config.rtdetr_config import rtdetr_cfg


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
    # YOLOvx
    elif args.model in ['yolovx_n', 'yolovx_t', 'yolovx_s', 'yolovx_m', 'yolovx_l', 'yolovx_x']:
        cfg = yolovx_cfg[args.model]
    # RT-DETR
    elif args.model in ['rtdetr_n', 'rtdetr_s', 'rtdetr_m', 'rtdetr_l', 'rtdetr_x']:
        cfg = rtdetr_cfg[args.model]

    return cfg

