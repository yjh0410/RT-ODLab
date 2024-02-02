# ------------------ Dataset Config ------------------
from .data_config.dataset_config import dataset_cfg


def build_dataset_config(args):
    if args.dataset in ['coco', 'coco-val', 'coco-test']:
        cfg = dataset_cfg['coco']
    else:
        cfg = dataset_cfg[args.dataset]

    print('==============================')
    print('Dataset Config: {} \n'.format(cfg))

    return cfg


# ------------------ Transform Config ------------------
from .data_config.transform_config import (
    # YOLOv5-Style
    yolov5_p_trans_config,
    yolov5_n_trans_config,
    yolov5_s_trans_config,
    yolov5_m_trans_config,
    yolov5_l_trans_config,
    yolov5_x_trans_config,
    # YOLOX-Style
    yolox_p_trans_config,
    yolox_n_trans_config,
    yolox_s_trans_config,
    yolox_m_trans_config,
    yolox_l_trans_config,
    yolox_x_trans_config,
    # SSD-Style
    ssd_trans_config,
    # RT-DETR style
    rtdetr_base_trans_config,
    rtdetr_l_trans_config,
    rtdetr_x_trans_config
)

def build_trans_config(trans_config='ssd'):
    print('==============================')
    print('Transform: {}-Style ...'.format(trans_config))
   
    # SSD-style transform 
    if trans_config == 'ssd':
        cfg = ssd_trans_config

    # YOLOv5-style transform 
    elif trans_config == 'yolov5_p':
        cfg = yolov5_p_trans_config
    elif trans_config == 'yolov5_n':
        cfg = yolov5_n_trans_config
    elif trans_config == 'yolov5_s':
        cfg = yolov5_s_trans_config
    elif trans_config == 'yolov5_m':
        cfg = yolov5_m_trans_config
    elif trans_config == 'yolov5_l':
        cfg = yolov5_l_trans_config
    elif trans_config == 'yolov5_x':
        cfg = yolov5_x_trans_config
        
    # YOLOX-style transform 
    elif trans_config == 'yolox_p':
        cfg = yolox_p_trans_config
    elif trans_config == 'yolox_n':
        cfg = yolox_n_trans_config
    elif trans_config == 'yolox_s':
        cfg = yolox_s_trans_config
    elif trans_config == 'yolox_m':
        cfg = yolox_m_trans_config
    elif trans_config == 'yolox_l':
        cfg = yolox_l_trans_config
    elif trans_config == 'yolox_x':
        cfg = yolox_x_trans_config

    # RT-DETR style
    elif trans_config == 'rtdetr_base':
        cfg = rtdetr_base_trans_config
    elif trans_config == 'rtdetr_l':
        cfg = rtdetr_l_trans_config
    elif trans_config == 'rtdetr_x':
        cfg = rtdetr_x_trans_config

    print('Transform Config: {} \n'.format(cfg))

    return cfg


# ------------------ Model Config ------------------
## YOLO series
from .model_config.yolov1_config import yolov1_cfg
from .model_config.yolov2_config import yolov2_cfg
from .model_config.yolov3_config import yolov3_cfg
from .model_config.yolov4_config import yolov4_cfg
from .model_config.yolov5_config import yolov5_cfg, yolov5_adamw_cfg
from .model_config.yolov7_config import yolov7_cfg
from .model_config.yolov8_config import yolov8_cfg
from .model_config.yolox_config import yolox_cfg, yolox_adamw_cfg
## My RTCDet series
from .model_config.rtcdet_config import rtcdet_cfg, rtcdet_seg_cfg, rtcdet_pos_cfg, rtcdet_seg_pos_cfg
from .model_config.rtdetr_config import rtdetr_cfg
from .model_config.rtpdetr_config import rtpdetr_cfg

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
    # YOLOv5-AdamW
    elif args.model in ['yolov5_n_adamw', 'yolov5_s_adamw', 'yolov5_m_adamw', 'yolov5_l_adamw', 'yolov5_x_adamw']:
        cfg = yolov5_adamw_cfg[args.model]
    # YOLOv7
    elif args.model in ['yolov7_tiny', 'yolov7', 'yolov7_x']:
        cfg = yolov7_cfg[args.model]
    # YOLOv8
    elif args.model in ['yolov8_n', 'yolov8_s', 'yolov8_m', 'yolov8_l', 'yolov8_x']:
        cfg = yolov8_cfg[args.model]
    # YOLOX
    elif args.model in ['yolox_n', 'yolox_s', 'yolox_m', 'yolox_l', 'yolox_x']:
        cfg = yolox_cfg[args.model]
    # YOLOX-AdamW
    elif args.model in ['yolox_n_adamw', 'yolox_s_adamw', 'yolox_m_adamw', 'yolox_l_adamw', 'yolox_x_adamw']:
        cfg = yolox_adamw_cfg[args.model]
    # RTCDet
    elif args.model in ['rtcdet_n', 'rtcdet_t', 'rtcdet_s', 'rtcdet_m', 'rtcdet_l', 'rtcdet_x']:
        cfg = rtcdet_cfg[args.model]
    # RT-DETR
    elif args.model in ['rtdetr_r18', 'rtdetr_r34', 'rtdetr_r50', 'rtdetr_r101']:
        cfg = rtdetr_cfg[args.model]
    # RT-PlainDETR
    elif args.model in ['rtpdetr_r18', 'rtpdetr_r34', 'rtpdetr_r50', 'rtpdetr_r101']:
        cfg = rtpdetr_cfg[args.model]

    return cfg

