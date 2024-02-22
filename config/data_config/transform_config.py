# Transform config


# ----------------------- SSD-Style Transform -----------------------
ssd_trans_config = {
    'aug_type': 'ssd',
    'use_ablu': False,
    # Mosaic & Mixup are not used for SSD-style augmentation
    'mosaic_prob': 0.0,
    'mixup_prob':  0.0,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolov5',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}


# ----------------------- YOLOv5-Style Transform -----------------------
yolo_x_trans_config = {
    'aug_type': 'yolo',
    'use_ablu': True,
    # Basic Augment
    'affine_params': {
        'degrees': 0.0,
        'translate': 0.2,
        'scale': [0.1, 2.0],
        'shear': 0.0,
        'perspective': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
    },
    # Mosaic & Mixup
    'mosaic_prob': 1.0,
    'mixup_prob':  0.2,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolov5',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}

yolo_l_trans_config = {
    'aug_type': 'yolo',
    'use_ablu': True,
    # Basic Augment
    'affine_params': {
        'degrees': 0.0,
        'translate': 0.2,
        'scale': [0.1, 2.0],
        'shear': 0.0,
        'perspective': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
    },
    # Mosaic & Mixup
    'mosaic_prob': 1.0,
    'mixup_prob':  0.15,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolov5',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}

yolo_m_trans_config = {
    'aug_type': 'yolo',
    'use_ablu': True,
    # Basic Augment
    'affine_params': {
        'degrees': 0.0,
        'translate': 0.2,
        'scale': [0.1, 2.0],
        'shear': 0.0,
        'perspective': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
    },
    # Mosaic & Mixup
    'mosaic_prob': 1.0,
    'mixup_prob':  0.10,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolov5',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}

yolo_s_trans_config = {
    'aug_type': 'yolo',
    'use_ablu': True,
    # Basic Augment
    'affine_params': {
        'degrees': 0.0,
        'translate': 0.2,
        'scale': [0.1, 2.0],
        'shear': 0.0,
        'perspective': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
    },
    # Mosaic & Mixup
    'mosaic_prob': 1.0,
    'mixup_prob':  0.0,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolov5',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}

yolo_n_trans_config = {
    'aug_type': 'yolo',
    'use_ablu': True,
    # Basic Augment
    'affine_params': {
        'degrees': 0.0,
        'translate': 0.1,
        'scale': [0.5, 1.5],
        'shear': 0.0,
        'perspective': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
    },
    # Mosaic & Mixup
    'mosaic_prob': 1.0,
    'mixup_prob':  0.0,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolov5',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}

yolo_p_trans_config = {
    'aug_type': 'yolo',
    'use_ablu': True,
    # Basic Augment
    'affine_params': {
        'degrees': 0.0,
        'translate': 0.1,
        'scale': [0.5, 1.5],
        'shear': 0.0,
        'perspective': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
    },
    # Mosaic & Mixup
    'mosaic_prob': 0.5,
    'mixup_prob':  0.0,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolov5',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}


# ----------------------- YOLOX-Style Transform -----------------------
yolox_x_trans_config = {
    'aug_type': 'yolo',
    'use_ablu': False,
    # Basic Augment
    'affine_params': {
        'degrees': 10.0,
        'translate': 0.1,
        'scale': [0.1, 2.0],
        'shear': 2.0,
        'perspective': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
    },
    # Mosaic & Mixup
    'mosaic_prob': 1.0,
    'mixup_prob':  1.0,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolox',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}

yolox_l_trans_config = {
    'aug_type': 'yolo',
    'use_ablu': False,
    # Basic Augment
    'affine_params': {
        'degrees': 10.0,
        'translate': 0.1,
        'scale': [0.1, 2.0],
        'shear': 2.0,
        'perspective': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
    },
    # Mosaic & Mixup
    'mosaic_prob': 1.0,
    'mixup_prob':  1.0,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolox',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}

yolox_m_trans_config = {
    'aug_type': 'yolo',
    'use_ablu': False,
    # Basic Augment
    'affine_params': {
        'degrees': 10.0,
        'translate': 0.1,
        'scale': [0.1, 2.0],
        'shear': 2.0,
        'perspective': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
    },
    # Mosaic & Mixup
    'mosaic_prob': 1.0,
    'mixup_prob':  1.0,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolox',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}

yolox_s_trans_config = {
    'aug_type': 'yolo',
    'use_ablu': False,
    # Basic Augment
    'affine_params': {
        'degrees': 10.0,
        'translate': 0.1,
        'scale': [0.1, 2.0],
        'shear': 2.0,
        'perspective': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
    },
    # Mosaic & Mixup
    'mosaic_prob': 1.0,
    'mixup_prob':  1.0,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolox',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}

yolox_n_trans_config = {
    'aug_type': 'yolo',
    'use_ablu': False,
    # Basic Augment
    'affine_params': {
        'degrees': 10.0,
        'translate': 0.1,
        'scale': [0.1, 2.0],
        'shear': 2.0,
        'perspective': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
    },
    # Mosaic & Mixup
    'mosaic_prob': 1.0,
    'mixup_prob':  0.5,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolox',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}

yolox_p_trans_config = {
    'aug_type': 'yolo',
    'use_ablu': False,
    # Basic Augment
    'affine_params': {
        'degrees': 10.0,
        'translate': 0.1,
        'scale': [0.1, 2.0],
        'shear': 2.0,
        'perspective': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
    },
    # Mosaic & Mixup
    'mosaic_prob': 0.5,
    'mixup_prob':  0.0,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolox',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}
