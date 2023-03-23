# ------------------ Model Config ----------------------
from .yolov1_config import yolov1_cfg


def build_model_config(args):
    print('==============================')
    print('Model: {} ...'.format(args.model.upper()))
    # YOLOv1
    if args.model == 'yolov1':
        cfg = yolov1_cfg
    # # YOLOv2
    # elif args.model == 'yolov2':
    #     cfg = yolov2_cfg
    # # YOLOv3
    # elif args.model == 'yolov3':
    #     cfg = yolov3_cfg
    # # YOLOv4
    # elif args.model == 'yolov4':
    #     cfg = yolov4_cfg

    return cfg


# ------------------ Transform Config ----------------------
from .transform_config import yolov5_trans_config, ssd_trans_config

def build_trans_config(trans_config='ssd'):
    print('==============================')
    print('Transform: {}-Style ...'.format(trans_config))
    # SSD-style transform 
    if trans_config == 'ssd':
        cfg = ssd_trans_config
    # YOLOv5-style transform 
    elif trans_config == 'yolov5':
        cfg = yolov5_trans_config

    return cfg
