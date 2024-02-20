# Transform config


# ----------------------- YOLOv5-Style Transform -----------------------
yolov5_x_trans_config = {
    'aug_type': 'yolov5',
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
    'mosaic_keep_ratio': True,
    'mosaic_prob': 1.0,
    'mixup_prob':  0.2,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolov5',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}

yolov5_l_trans_config = {
    'aug_type': 'yolov5',
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
    'mosaic_keep_ratio': True,
    'mosaic_prob': 1.0,
    'mixup_prob':  0.15,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolov5',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}

yolov5_m_trans_config = {
    'aug_type': 'yolov5',
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
    'mosaic_keep_ratio': True,
    'mosaic_prob': 1.0,
    'mixup_prob':  0.10,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolov5',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}

yolov5_s_trans_config = {
    'aug_type': 'yolov5',
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
    'mosaic_keep_ratio': True,
    'mosaic_prob': 1.0,
    'mixup_prob':  0.0,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolov5',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}

yolov5_n_trans_config = {
    'aug_type': 'yolov5',
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
    'mosaic_keep_ratio': True,
    'mosaic_prob': 1.0,
    'mixup_prob':  0.0,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolov5',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}

yolov5_p_trans_config = {
    'aug_type': 'yolov5',
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
    'mosaic_keep_ratio': True,
    'mosaic_prob': 0.5,
    'mixup_prob':  0.0,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolov5',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}


# ----------------------- YOLOX-Style Transform -----------------------
yolox_x_trans_config = {
    'aug_type': 'yolov5',
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
    'mosaic_keep_ratio': True,
    'mosaic_prob': 1.0,
    'mixup_prob':  1.0,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolox',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}

yolox_l_trans_config = {
    'aug_type': 'yolov5',
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
    'mosaic_keep_ratio': True,
    'mosaic_prob': 1.0,
    'mixup_prob':  1.0,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolox',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}

yolox_m_trans_config = {
    'aug_type': 'yolov5',
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
    'mosaic_keep_ratio': True,
    'mosaic_prob': 1.0,
    'mixup_prob':  1.0,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolox',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}

yolox_s_trans_config = {
    'aug_type': 'yolov5',
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
    'mosaic_keep_ratio': True,
    'mosaic_prob': 1.0,
    'mixup_prob':  1.0,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolox',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}

yolox_n_trans_config = {
    'aug_type': 'yolov5',
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
    'mosaic_keep_ratio': True,
    'mosaic_prob': 1.0,
    'mixup_prob':  0.5,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolox',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}

yolox_p_trans_config = {
    'aug_type': 'yolov5',
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
    'mosaic_keep_ratio': True,
    'mosaic_prob': 0.5,
    'mixup_prob':  0.0,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolox',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}


# ----------------------- SSD-Style Transform -----------------------
ssd_trans_config = {
    'aug_type': 'ssd',
    'use_ablu': False,
    # Mosaic & Mixup are not used for SSD-style augmentation
    'mosaic_keep_ratio': False,
    'mosaic_prob': 0.0,
    'mixup_prob':  0.0,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolov5',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}


# ----------------------- SSD-Style Transform -----------------------
rtdetr_base_trans_config = {
    'aug_type': 'rtdetr',
    'use_ablu': True,
    'pixel_mean': [123.675, 116.28, 103.53],  # IN-1K statistics
    'pixel_std':  [58.395, 57.12, 57.375],    # IN-1K statistics
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
    'mosaic_keep_ratio': False,
    'mosaic_prob': 0.0,
    'mixup_prob':  0.0,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolov5',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}
