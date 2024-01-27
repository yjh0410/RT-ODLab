import os

try:
    # dataset class
    from .voc import VOCDataset
    from .coco import COCODataset
    from .crowdhuman import CrowdHumanDataset
    from .widerface import WiderFaceDataset
    from .customed import CustomedDataset
    # transform class
    from .data_augment.ssd_augment import SSDAugmentation, SSDBaseTransform
    from .data_augment.yolov5_augment import YOLOv5Augmentation, YOLOv5BaseTransform
    from .data_augment.rtdetr_augment import RTDetrAugmentation, RTDetrBaseTransform

except:
    # dataset class
    from voc import VOCDataset
    from coco import COCODataset
    from crowdhuman import CrowdHumanDataset
    from widerface import WiderFaceDataset
    from customed import CustomedDataset
    # transform class
    from data_augment.ssd_augment import SSDAugmentation, SSDBaseTransform
    from data_augment.yolov5_augment import YOLOv5Augmentation, YOLOv5BaseTransform
    from data_augment.rtdetr_augment import RTDetrAugmentation, RTDetrBaseTransform


# ------------------------------ Dataset ------------------------------
def build_dataset(args, data_cfg, trans_config, transform, is_train=False):
    # ------------------------- Basic parameters -------------------------
    data_dir = os.path.join(args.root, data_cfg['data_name'])
    num_classes = data_cfg['num_classes']
    class_names = data_cfg['class_names']
    class_indexs = data_cfg['class_indexs']
    dataset_info = {
        'num_classes': num_classes,
        'class_names': class_names,
        'class_indexs': class_indexs
    }

    # ------------------------- Build dataset -------------------------
    ## VOC dataset
    if args.dataset == 'voc':
        image_sets = [('2007', 'trainval'), ('2012', 'trainval')] if is_train else [('2007', 'test')]
        dataset = VOCDataset(img_size     = args.img_size,
                             data_dir     = data_dir,
                             image_sets   = image_sets,
                             transform    = transform,
                             trans_config = trans_config,
                             is_train     = is_train,
                             load_cache   = args.load_cache
                             )
    ## COCO dataset
    elif args.dataset == 'coco':
        image_set = 'train2017' if is_train else 'val2017'
        dataset = COCODataset(img_size     = args.img_size,
                              data_dir     = data_dir,
                              image_set    = image_set,
                              transform    = transform,
                              trans_config = trans_config,
                              is_train     = is_train,
                              load_cache   = args.load_cache
                              )
    ## CrowdHuman dataset
    elif args.dataset == 'crowdhuman':
        image_set = 'train' if is_train else 'val'
        dataset = CrowdHumanDataset(img_size     = args.img_size,
                                    data_dir     = data_dir,
                                    image_set    = image_set,
                                    transform    = transform,
                                    trans_config = trans_config,
                                    is_train     = is_train,
                                    )
    ## WiderFace dataset
    elif args.dataset == 'widerface':
        image_set = 'train' if is_train else 'val'
        dataset = WiderFaceDataset(img_size     = args.img_size,
                                    data_dir     = data_dir,
                                    image_set    = image_set,
                                    transform    = transform,
                                    trans_config = trans_config,
                                    is_train     = is_train,
                                    )
    ## Custom dataset
    elif args.dataset == 'customed':
        image_set = 'train' if is_train else 'val'
        dataset = CustomedDataset(data_dir     = data_dir,
                                  img_size     = args.img_size,
                                  image_set    = image_set,
                                  transform    = transform,
                                  trans_config = trans_config,
                                  is_train      = is_train,
                                  load_cache    = args.load_cache
                                  )

    return dataset, dataset_info


# ------------------------------ Transform ------------------------------
def build_transform(args, trans_config, max_stride=32, is_train=False):
    # ---------------- Modify trans_config ----------------
    if is_train:
        ## mosaic prob.
        if args.mosaic is not None:
            trans_config['mosaic_prob'] = args.mosaic
        ## mixup prob.
        if args.mixup is not None:
            trans_config['mixup_prob'] = args.mixup

    # ---------------- Build transform ----------------
    ## SSD-style transform
    if trans_config['aug_type'] == 'ssd':
        if is_train:
            transform = SSDAugmentation(img_size=args.img_size,)
        else:
            transform = SSDBaseTransform(img_size=args.img_size,)
    ## YOLO-style transform
    elif trans_config['aug_type'] == 'yolov5':
        if is_train:
            transform = YOLOv5Augmentation(
                img_size=args.img_size,
                trans_config=trans_config,
                use_ablu=trans_config['use_ablu']
                )
        else:
            transform = YOLOv5BaseTransform(
                img_size=args.img_size,
                max_stride=max_stride
                )
    ## RT_DETR-style transform
    elif trans_config['aug_type'] == 'rtdetr':
        if is_train:
            use_mosaic = False if trans_config['mosaic_prob'] < 0.2 else True
            transform = RTDetrAugmentation(
                img_size=args.img_size, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375], use_mosaic=use_mosaic)
        else:
            transform = RTDetrBaseTransform(
                img_size=args.img_size, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375])

    return transform, trans_config
