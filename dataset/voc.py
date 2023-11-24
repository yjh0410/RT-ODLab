import os
import cv2
import torch
import random
import numpy as np
import os.path as osp
import xml.etree.ElementTree as ET

import torch
import torch.utils.data as data
try:
    from .data_augment.yolov5_augment import yolov5_mosaic_augment, yolov5_mixup_augment, yolox_mixup_augment
except:
    from data_augment.yolov5_augment import yolov5_mosaic_augment, yolov5_mixup_augment, yolox_mixup_augment


# VOC class names
VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt if i % 2 == 0 else cur_pt
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [x1, y1, x2, y2, label_ind]

        return res  # [[x1, y1, x2, y2, label_ind], ... ]


class VOCDataset(data.Dataset):
    def __init__(self, 
                 img_size     :int = 640,
                 data_dir     :str = None,
                 image_sets   = [('2007', 'trainval'), ('2012', 'trainval')],
                 trans_config = None,
                 transform    = None,
                 is_train     :bool = False,
                 load_cache   :bool = False,
                 ):
        # ----------- Basic parameters -----------
        self.img_size = img_size
        self.image_set = image_sets
        self.is_train = is_train
        self.target_transform = VOCAnnotationTransform()
        # ----------- Path parameters -----------
        self.root = data_dir
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        # ----------- Data parameters -----------
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))
        self.dataset_size = len(self.ids)
        # ----------- Transform parameters -----------
        self.transform = transform
        self.mosaic_prob = trans_config['mosaic_prob'] if trans_config else 0.0
        self.mixup_prob = trans_config['mixup_prob'] if trans_config else 0.0
        self.trans_config = trans_config
        print('==============================')
        print('use Mosaic Augmentation: {}'.format(self.mosaic_prob))
        print('use Mixup Augmentation: {}'.format(self.mixup_prob))
        print('==============================')
        # ----------- Cached data -----------
        self.load_cache = load_cache
        self.cached_datas = None
        if self.load_cache:
            self.cached_datas = self._load_cache()


    # ------------ Basic dataset function ------------
    def __getitem__(self, index):
        image, target, deltas = self.pull_item(index)
        return image, target, deltas

    def __len__(self):
        return self.dataset_size

    def _load_cache(self):
        data_items = []
        for idx in range(self.dataset_size):
            if idx % 2000 == 0:
                print("Caching images and targets : {} / {} ...".format(idx, self.dataset_size))

            # load a data
            image, target = self.load_image_target(idx)
            orig_h, orig_w, _ = image.shape

            # resize image
            r = self.img_size / max(orig_h, orig_w)
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

            dict_item = {}
            dict_item["image"] = image
            dict_item["target"] = target

            data_items.append(dict_item)
        
        return data_items

    # ------------ Mosaic & Mixup ------------
    def load_mosaic(self, index):
        # load 4x mosaic image
        index_list = np.arange(index).tolist() + np.arange(index+1, len(self.ids)).tolist()
        id1 = index
        id2, id3, id4 = random.sample(index_list, 3)
        indexs = [id1, id2, id3, id4]

        # load images and targets
        image_list = []
        target_list = []
        for index in indexs:
            img_i, target_i = self.load_image_target(index)
            image_list.append(img_i)
            target_list.append(target_i)

        # Mosaic
        if self.trans_config['mosaic_type'] == 'yolov5_mosaic':
            image, target = yolov5_mosaic_augment(
                image_list, target_list, self.img_size, self.trans_config, self.is_train)

        return image, target

    def load_mixup(self, origin_image, origin_target):
        # YOLOv5 type Mixup
        if self.trans_config['mixup_type'] == 'yolov5_mixup':
            new_index = np.random.randint(0, len(self.ids))
            new_image, new_target = self.load_mosaic(new_index)
            image, target = yolov5_mixup_augment(
                origin_image, origin_target, new_image, new_target)
        # YOLOX type Mixup
        elif self.trans_config['mixup_type'] == 'yolox_mixup':
            new_index = np.random.randint(0, len(self.ids))
            new_image, new_target = self.load_image_target(new_index)
            image, target = yolox_mixup_augment(
                origin_image, origin_target, new_image, new_target, self.img_size, self.trans_config['mixup_scale'])

        return image, target
    
    # ------------ Load data function ------------
    def load_image_target(self, index):
        # == Load a data from the cached data ==
        if self.cached_datas is not None:
            # load a data
            data_item = self.cached_datas[index]
            image = data_item["image"]
            target = data_item["target"]
        # == Load a data from the local disk ==
        else:        
            # load an image
            image, _ = self.pull_image(index)
            height, width, channels = image.shape

            # laod an annotation
            anno, _ = self.pull_anno(index)

            # guard against no boxes via resizing
            anno = np.array(anno).reshape(-1, 5)
            target = {
                "boxes": anno[:, :4],
                "labels": anno[:, 4],
                "orig_size": [height, width]
            }
        
        return image, target

    def pull_item(self, index):
        if random.random() < self.mosaic_prob:
            # load a mosaic image
            mosaic = True
            image, target = self.load_mosaic(index)
        else:
            mosaic = False
            # load an image and target
            image, target = self.load_image_target(index)

        # MixUp
        if random.random() < self.mixup_prob:
            image, target = self.load_mixup(image, target)

        # augment
        image, target, deltas = self.transform(image, target, mosaic)

        return image, target, deltas

    def pull_image(self, index):
        img_id = self.ids[index]
        image = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

        return image, img_id

    def pull_anno(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self.target_transform(anno)

        return anno, img_id


if __name__ == "__main__":
    import time
    import argparse
    from build import build_transform
    
    parser = argparse.ArgumentParser(description='VOC-Dataset')

    # opt
    parser.add_argument('--root', default='/Users/liuhaoran/Desktop/python_work/object-detection/dataset/VOCdevkit/',
                        help='data root')
    parser.add_argument('-size', '--img_size', default=640, type=int,
                        help='input image size.')
    parser.add_argument('--aug_type', type=str, default='ssd',
                        help='augmentation type')
    parser.add_argument('--mosaic', default=0., type=float,
                        help='mosaic augmentation.')
    parser.add_argument('--mixup', default=0., type=float,
                        help='mixup augmentation.')
    parser.add_argument('--mixup_type', type=str, default='yolov5_mixup',
                        help='mixup augmentation.')
    parser.add_argument('--is_train', action="store_true", default=False,
                        help='mixup augmentation.')
    parser.add_argument('--load_cache', action="store_true", default=False,
                        help='Path to the cached data.')
    
    args = parser.parse_args()

    trans_config = {
        'aug_type': args.aug_type,    # optional: ssd, yolov5
        # Basic Augment
        'degrees': 0.0,
        'translate': 0.2,
        'scale': [0.1, 2.0],
        'shear': 0.0,
        'perspective': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'use_ablu': True,
        # Mosaic & Mixup
        'mosaic_prob': args.mosaic,
        'mixup_prob': args.mixup,
        'mosaic_type': 'yolov5_mosaic',
        'mixup_type': args.mixup_type,   # optional: yolov5_mixup, yolox_mixup
        'mixup_scale': [0.5, 1.5]
    }
    transform, trans_cfg = build_transform(args, trans_config, 32, args.is_train)

    dataset = VOCDataset(
        img_size=args.img_size,
        data_dir=args.root,
        image_sets=[('2007', 'trainval'), ('2012', 'trainval')] if args.is_train else [('2007', 'test')],
        trans_config=trans_config,
        transform=transform,
        is_train=args.is_train,
        load_cache=args.load_cache
        )
    
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(20)]
    print('Data length: ', len(dataset))

    for i in range(1000):
        t0 = time.time()
        image, target, deltas = dataset.pull_item(i)
        print("Load data: {} s".format(time.time() - t0))

        # to numpy
        image = image.permute(1, 2, 0).numpy()
        # to uint8
        image = image.astype(np.uint8)
        image = image.copy()
        img_h, img_w = image.shape[:2]

        boxes = target["boxes"]
        labels = target["labels"]

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            if x2 - x1 > 1 and y2 - y1 > 1:
                cls_id = int(label)
                color = class_colors[cls_id]
                # class name
                label = VOC_CLASSES[cls_id]
                image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
                # put the test on the bbox
                cv2.putText(image, label, (int(x1), int(y1 - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)
        cv2.imshow('gt', image)
        # cv2.imwrite(str(i)+'.jpg', img)
        cv2.waitKey(0)