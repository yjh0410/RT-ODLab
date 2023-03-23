from .ssd_augment import SSDAugmentation, SSDBaseTransform
from .yolov5_augment import YOLOv5Augmentation, YOLOv5BaseTransform


def build_transform(img_size, trans_config, is_train=False):
    if trans_config['aug_type'] == 'ssd':
        if is_train:
            transform = SSDAugmentation(img_size=img_size)
        else:
            transform = SSDBaseTransform(img_size=img_size)

    elif trans_config['aug_type'] == 'yolov5':
        if is_train:
            transform = YOLOv5Augmentation(
                img_size=img_size,
                trans_config=trans_config
                )
        else:
            transform = YOLOv5BaseTransform(
                img_size=img_size,
                )

    return transform
