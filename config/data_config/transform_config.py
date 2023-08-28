# Transform config


# ----------------------- YOLOv5-Style Transform -----------------------
yolov5_huge_trans_config = {
    'aug_type': 'yolov5',
    # Basic Augment
    'degrees': 0.0,
    'translate': 0.2,
    'scale': [0.1, 2.0],
    'shear': 0.0,
    'perspective': 0.0,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    # Mosaic & Mixup
    'mosaic_prob': 1.0,
    'mixup_prob': 0.2,
    'mosaic_type': 'yolov5_mosaic',
    'mixup_type': 'yolov5_mixup',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp
}

yolov5_large_trans_config = {
    'aug_type': 'yolov5',
    # Basic Augment
    'degrees': 0.0,
    'translate': 0.2,
    'scale': [0.1, 2.0],
    'shear': 0.0,
    'perspective': 0.0,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    # Mosaic & Mixup
    'mosaic_prob': 1.0,
    'mixup_prob': 0.15,
    'mosaic_type': 'yolov5_mosaic',
    'mixup_type': 'yolov5_mixup',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp
}

yolov5_medium_trans_config = {
    'aug_type': 'yolov5',
    # Basic Augment
    'degrees': 0.0,
    'translate': 0.2,
    'scale': [0.1, 2.0],
    'shear': 0.0,
    'perspective': 0.0,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    # Mosaic & Mixup
    'mosaic_prob': 1.0,
    'mixup_prob': 0.10,
    'mosaic_type': 'yolov5_mosaic',
    'mixup_type': 'yolov5_mixup',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp
}

yolov5_small_trans_config = {
    'aug_type': 'yolov5',
    # Basic Augment
    'degrees': 0.0,
    'translate': 0.2,
    'scale': [0.1, 2.0],
    'shear': 0.0,
    'perspective': 0.0,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    # Mosaic & Mixup
    'mosaic_prob': 1.0,
    'mixup_prob': 0.05,
    'mosaic_type': 'yolov5_mosaic',
    'mixup_type': 'yolov5_mixup',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp
}

yolov5_nano_trans_config = {
    'aug_type': 'yolov5',
    # Basic Augment
    'degrees': 0.0,
    'translate': 0.1,
    'scale': [0.5, 1.5],
    'shear': 0.0,
    'perspective': 0.0,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    # Mosaic & Mixup
    'mosaic_prob': 1.0,
    'mixup_prob': 0.05,
    'mosaic_type': 'yolov5_mosaic',
    'mixup_type': 'yolov5_mixup',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp
}

yolov5_pico_trans_config = {
    'aug_type': 'yolov5',
    # Basic Augment
    'degrees': 0.0,
    'translate': 0.1,
    'scale': [0.5, 1.5],
    'shear': 0.0,
    'perspective': 0.0,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    # Mosaic & Mixup
    'mosaic_prob': 0.5,
    'mixup_prob': 0.0,
    'mosaic_type': 'yolov5_mosaic',
    'mixup_type': 'yolov5_mixup',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp
}


# ----------------------- YOLOX-Style Transform -----------------------
yolox_huge_trans_config = {
    'aug_type': 'yolov5',
    # Basic Augment
    'degrees': 10.0,
    'translate': 0.1,
    'scale': [0.1, 2.0],
    'shear': 2.0,
    'perspective': 0.0,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    # Mosaic & Mixup
    'mosaic_prob': 1.0,
    'mixup_prob': 1.0,
    'mosaic_type': 'yolov5_mosaic',
    'mixup_type': 'yolox_mixup',
    'mixup_scale': [0.5, 1.5]
}

yolox_large_trans_config = {
    'aug_type': 'yolov5',
    # Basic Augment
    'degrees': 10.0,
    'translate': 0.1,
    'scale': [0.1, 2.0],
    'shear': 2.0,
    'perspective': 0.0,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    # Mosaic & Mixup
    'mosaic_prob': 1.0,
    'mixup_prob': 1.0,
    'mosaic_type': 'yolov5_mosaic',
    'mixup_type': 'yolox_mixup',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp
}

yolox_medium_trans_config = {
    'aug_type': 'yolov5',
    # Basic Augment
    'degrees': 10.0,
    'translate': 0.1,
    'scale': [0.1, 2.0],
    'shear': 2.0,
    'perspective': 0.0,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    # Mosaic & Mixup
    'mosaic_prob': 1.0,
    'mixup_prob': 1.0,
    'mosaic_type': 'yolov5_mosaic',
    'mixup_type': 'yolox_mixup',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp
}

yolox_small_trans_config = {
    'aug_type': 'yolov5',
    # Basic Augment
    'degrees': 10.0,
    'translate': 0.1,
    'scale': [0.1, 2.0],
    'shear': 2.0,
    'perspective': 0.0,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    # Mosaic & Mixup
    'mosaic_prob': 1.0,
    'mixup_prob': 1.0,
    'mosaic_type': 'yolov5_mosaic',
    'mixup_type': 'yolox_mixup',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp
}

yolox_nano_trans_config = {
    'aug_type': 'yolov5',
    # Basic Augment
    'degrees': 10.0,
    'translate': 0.1,
    'scale': [0.5, 1.5],
    'shear': 2.0,
    'perspective': 0.0,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    # Mosaic & Mixup
    'mosaic_prob': 1.0,
    'mixup_prob': 0.5,
    'mosaic_type': 'yolov5_mosaic',
    'mixup_type': 'yolox_mixup',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp
}

yolox_pico_trans_config = {
    'aug_type': 'yolov5',
    # Basic Augment
    'degrees': 10.0,
    'translate': 0.1,
    'scale': [0.5, 1.5],
    'shear': 2.0,
    'perspective': 0.0,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    # Mosaic & Mixup
    'mosaic_prob': 0.5,
    'mixup_prob': 0.0,
    'mosaic_type': 'yolov5_mosaic',
    'mixup_type': 'yolox_mixup',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp
}


# ----------------------- SSD-Style Transform -----------------------
ssd_trans_config = {
    'aug_type': 'ssd',
    # Mosaic & Mixup are not used for SSD-style augmentation
    'mosaic_prob': 0.,
    'mixup_prob': 0.,
    'mosaic_type': 'yolov5_mosaic',
    'mixup_type': 'yolov5_mixup',
    'mixup_scale': [0.5, 1.5]
}
