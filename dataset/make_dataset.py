import os
import cv2
import torch
import random
import numpy as np

import sys
sys.path.append("../")
from utils import distributed_utils
from dataset.voc import VOCDataset, VOC_CLASSES
from dataset.coco import COCODataset, coco_class_labels, coco_class_index
from config import build_trans_config, build_dataset_config


def fix_random_seed(args):
    seed = args.seed + distributed_utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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
        dataset = VOCDataset(
            img_size=args.img_size,
            data_dir=data_dir,
            image_sets=[('2007', 'trainval'), ('2012', 'trainval')] if is_train else [('2007', 'test')],
            transform=transform,
            trans_config=trans_config,
            load_cache=args.load_cache
            )
    ## COCO dataset
    elif args.dataset == 'coco':
        dataset = COCODataset(
            img_size=args.img_size,
            data_dir=data_dir,
            image_set='train2017' if is_train else 'val2017',
            transform=transform,
            trans_config=trans_config,
            load_cache=args.load_cache
            )

    return dataset, dataset_info

def visualize(image, target, dataset_name="voc"):
    if dataset_name == "voc":
        class_labels = VOC_CLASSES
        class_indexs = None
        num_classes  = 20
    elif dataset_name == "coco":
        class_labels = coco_class_labels
        class_indexs = coco_class_index
        num_classes  = 80
    else:
        raise NotImplementedError

    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255))
                     for _ in range(num_classes)]

    # to numpy
    # image = image.permute(1, 2, 0).numpy()
    image = image.astype(np.uint8)
    image = image.copy()

    boxes = target["boxes"]
    labels = target["labels"]
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        if x2 - x1 > 1 and y2 - y1 > 1:
            cls_id = int(label)
            color = class_colors[cls_id]
            # class name
            if dataset_name == 'coco':
                assert class_indexs is not None
                class_name = class_labels[class_indexs[cls_id]]
            else:
                class_name = class_labels[cls_id]
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            # put the test on the bbox
            cv2.putText(image, class_name, (int(x1), int(y1 - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)
    cv2.imshow('gt', image)
    cv2.waitKey(0)


if __name__ == "__main__":
    import argparse
    from build import build_transform
    
    parser = argparse.ArgumentParser(description='VOC-Dataset')

    # Seed
    parser.add_argument('--seed', default=42, type=int)
    # Dataset
    parser.add_argument('--root', default='/Users/yjh0410/Desktop/python_work/dataset/',
                        help='data root')
    parser.add_argument('--dataset', type=str, default="voc",
                        help='augmentation type.')
    parser.add_argument('--load_cache', action="store_true", default=False,
                        help='load cached data.')
    parser.add_argument('--vis_tgt', action="store_true", default=False,
                        help='load cached data.')
    parser.add_argument('--is_train', action="store_true", default=False,
                        help='mixup augmentation.')
    # Image size
    parser.add_argument('-size', '--img_size', default=640, type=int,
                        help='input image size.')
    # Augmentation
    parser.add_argument('--aug_type', type=str, default="yolov5_nano",
                        help='augmentation type.')
    parser.add_argument('--mosaic', default=None, type=float,
                        help='mosaic augmentation.')
    parser.add_argument('--mixup', default=None, type=float,
                        help='mixup augmentation.')
    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')
    # Output
    parser.add_argument('--output_dir', type=str, default='cache_data/',
                        help='data root')
    
    args = parser.parse_args()

    
    assert args.aug_type in ["yolov5_pico", "yolov5_nano", "yolov5_small", "yolov5_medium", "yolov5_large", "yolov5_huge",
                             "yolox_pico",  "yolox_nano",  "yolox_small",  "yolox_medium",  "yolox_large",  "yolox_huge"]
    

    # ------------- Build transform config -------------
    dataset_cfg  = build_dataset_config(args)
    trans_config = build_trans_config(args.aug_type)

    # ------------- Build transform -------------
    transform, trans_config = build_transform(args, trans_config, max_stride=32, is_train=args.is_train)

    # ------------- Build dataset -------------
    dataset, dataset_info = build_dataset(args, dataset_cfg, trans_config, transform, is_train=args.is_train)
    print('Data length: ', len(dataset))

    # ---------------------------- Fix random seed ----------------------------
    fix_random_seed(args)

    # ---------------------------- Main process ----------------------------
    # We only cache the taining data
    data_items = []
    for idx in range(len(dataset)):
        if idx % 2000 == 0:
            print("Caching images and targets : {} / {} ...".format(idx, len(dataset)))

        # load a data
        image, target = dataset.load_image_target(idx)
        orig_h, orig_w, _ = image.shape

        # resize image
        r = args.img_size / max(orig_h, orig_w)
        if r != 1: 
            interp = cv2.INTER_LINEAR
            new_size = (int(orig_w * r), int(orig_h * r))
            image = cv2.resize(image, new_size, interpolation=interp)
        img_h, img_w = image.shape[:2]

        # rescale bbox
        boxes = target["boxes"].copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / orig_w * img_w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / orig_h * img_h
        target["boxes"] = boxes

        # visualize data
        if args.vis_tgt:
            print(image.shape)
            visualize(image, target, args.dataset)
            continue

        dict_item = {}
        dict_item["image"] = image
        dict_item["target"] = target

        data_items.append(dict_item)

    output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    print('Cached data size: ', len(data_items))
    if args.is_train:
        save_file = os.path.join(output_dir, "{}_train.pth".format(args.dataset))
    else:
        save_file = os.path.join(output_dir, "{}_valid.pth".format(args.dataset))
    torch.save(data_items, save_file)
