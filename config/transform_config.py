# transform config


yolov5_trans_config = {
    'aug_type': 'yolov5',
    # Pixel mean & std
    'pixel_mean': [0., 0., 0.],
    'pixel_std': [1., 1., 1.],
    # Basic Augment
    'degrees': 0.0,
    'translate': 0.2,
    'scale': 0.9,
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
    'mixup_scale': [0.5, 1.5]
}


ssd_trans_config = {
    'aug_type': 'ssd',
    'pixel_mean': [0.406, 0.456, 0.485],
    'pixel_std': [0.225, 0.224, 0.229],
    # Mosaic & Mixup
    'mosaic_prob': 0.,
    'mixup_prob': 0.,
    'mosaic_type': 'yolov5_mosaic',
    'mixup_type': 'yolov5_mixup',
    'mixup_scale': [0.5, 1.5]
}
