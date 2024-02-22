import os
import cv2
import time
import random
import numpy as np

from torch.utils.data import Dataset

try:
    from pycocotools.coco import COCO
except:
    print("It seems that the COCOAPI is not installed.")

try:
    from .data_augment.strong_augment import MosaicAugment, MixupAugment
except:
    from  data_augment.strong_augment import MosaicAugment, MixupAugment


class CustomedDataset(Dataset):
    def __init__(self, 
                 img_size     :int  = 640,
                 data_dir     :str  = None, 
                 image_set    :str  = 'train',
                 transform          = None,
                 trans_config       = None,
                 is_train     :bool =False,
                 ):
        # ----------- Basic parameters -----------
        self.img_size = img_size
        self.image_set = image_set
        self.is_train = is_train
        # ----------- Path parameters -----------
        self.data_dir = data_dir
        self.json_file = '{}.json'.format(image_set)
        # ----------- Data parameters -----------
        self.coco = COCO(os.path.join(self.data_dir, image_set, 'annotations', self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        self.dataset_size = len(self.ids)
        # ----------- Transform parameters -----------
        self.trans_config = trans_config
        self.transform = transform
        # ----------- Strong augmentation -----------
        if is_train:
            self.mosaic_prob = trans_config['mosaic_prob'] if trans_config else 0.0
            self.mixup_prob  = trans_config['mixup_prob']  if trans_config else 0.0
            self.mosaic_augment = MosaicAugment(img_size, trans_config, is_train) if self.mosaic_prob > 0. else None
            self.mixup_augment  = MixupAugment(img_size, trans_config)            if self.mixup_prob > 0.  else None
        else:
            self.mosaic_prob = 0.0
            self.mixup_prob  = 0.0
            self.mosaic_augment = None
            self.mixup_augment  = None
        print('==============================')
        print('Image Set: {}'.format(image_set))
        print('Json file: {}'.format(self.json_file))
        print('use Mosaic Augmentation: {}'.format(self.mosaic_prob))
        print('use Mixup Augmentation: {}'.format(self.mixup_prob))

    # ------------ Basic dataset function ------------
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.pull_item(index)

    # ------------ Mosaic & Mixup ------------
    def load_mosaic(self, index):
        # ------------ Prepare 4 indexes of images ------------
        ## Load 4x mosaic image
        index_list = np.arange(index).tolist() + np.arange(index+1, len(self.ids)).tolist()
        id1 = index
        id2, id3, id4 = random.sample(index_list, 3)
        indexs = [id1, id2, id3, id4]

        ## Load images and targets
        image_list = []
        target_list = []
        for index in indexs:
            img_i, target_i = self.load_image_target(index)
            image_list.append(img_i)
            target_list.append(target_i)

        # ------------ Mosaic augmentation ------------
        image, target = self.mosaic_augment(image_list, target_list)

        return image, target

    def load_mixup(self, origin_image, origin_target):
        # ------------ Load a new image & target ------------
        if self.mixup_augment.mixup_type == 'yolov5':
            new_index = np.random.randint(0, len(self.ids))
            new_image, new_target = self.load_mosaic(new_index)
        elif self.mixup_augment.mixup_type == 'yolox':
            new_index = np.random.randint(0, len(self.ids))
            new_image, new_target = self.load_image_target(new_index)
            
        # ------------ Mixup augmentation ------------
        image, target = self.mixup_augment(origin_image, origin_target, new_image, new_target)

        return image, target
    
    # ------------ Load data function ------------
    def load_image_target(self, index):
        # load an image
        image, _ = self.pull_image(index)
        height, width, channels = image.shape

        # load a target
        bboxes, labels = self.pull_anno(index)
        target = {
            "boxes": bboxes,
            "labels": labels,
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
        id_ = self.ids[index]
        im_ann = self.coco.loadImgs(id_)[0] 
        img_file = os.path.join(
                self.data_dir, self.image_set, 'images', im_ann["file_name"])
        image = cv2.imread(img_file)

        return image, id_

    def pull_anno(self, index):
        img_id = self.ids[index]
        im_ann = self.coco.loadImgs(img_id)[0]
        anno_ids = self.coco.getAnnIds(imgIds=[int(img_id)], iscrowd=0)
        annotations = self.coco.loadAnns(anno_ids)
        
        # image infor
        width = im_ann['width']
        height = im_ann['height']
        
        #load a target
        bboxes = []
        labels = []
        for anno in annotations:
            if 'bbox' in anno and anno['area'] > 0:
                # bbox
                x1 = np.max((0, anno['bbox'][0]))
                y1 = np.max((0, anno['bbox'][1]))
                x2 = np.min((width - 1, x1 + np.max((0, anno['bbox'][2] - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, anno['bbox'][3] - 1))))
                if x2 <= x1 or y2 <= y1:
                    continue
                # class label
                cls_id = self.class_ids.index(anno['category_id'])
                
                bboxes.append([x1, y1, x2, y2])
                labels.append(cls_id)

        # guard against no boxes via resizing
        bboxes = np.array(bboxes).reshape(-1, 4)
        labels = np.array(labels).reshape(-1)
        
        return bboxes, labels


if __name__ == "__main__":
    import time
    import argparse
    from build import build_transform

    import sys
    sys.path.append("..")
    from config.data_config.dataset_config import dataset_cfg
    data_config = dataset_cfg["customed"]
    categories = data_config["class_names"]

    
    parser = argparse.ArgumentParser(description='RT-ODLab')

    # opt
    parser.add_argument('--root', default='/Users/liuhaoran/Desktop/python_work/object-detection/dataset/AnimalDataset/',
                        help='data root')
    parser.add_argument('--split', default='train',
                        help='data split')
    parser.add_argument('-size', '--img_size', default=640, type=int, 
                        help='input image size')
    parser.add_argument('--min_box_size', default=8.0, type=float,
                        help='min size of target bounding box.')
    parser.add_argument('--mosaic', default=None, type=float,
                        help='mosaic augmentation.')
    parser.add_argument('--mixup', default=None, type=float,
                        help='mixup augmentation.')
    parser.add_argument('--is_train', action="store_true", default=False,
                        help='mixup augmentation.')
    
    args = parser.parse_args()

    trans_config = {
        'aug_type': args.aug_type,    # optional: ssd, yolov5
        'pixel_mean': [0., 0., 0.],
        'pixel_std':  [255., 255., 255.],
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
        'mosaic_type': 'yolov5',
        'mixup_type':  'yolov5',
        'mixup_scale': [0.5, 1.5]
    }
    transform, trans_cfg = build_transform(args, trans_config, 32, args.is_train)
    pixel_mean = transform.pixel_mean
    pixel_std  = transform.pixel_std
    color_format = transform.color_format

    dataset = CustomedDataset(
        img_size=args.img_size,
        data_dir=args.root,
        image_set=args.split,
        transform=transform,
        trans_config=trans_config,
        is_train=args.is_train,
        load_cache=args.load_cache
        )
    
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(80)]
    print('Data length: ', len(dataset))

    for i in range(1000):
        t0 = time.time()
        image, target, deltas = dataset.pull_item(i)
        print("Load data: {} s".format(time.time() - t0))

        # to numpy
        image = image.permute(1, 2, 0).numpy()
        
        # denormalize
        image = image * pixel_std + pixel_mean
        if color_format == 'rgb':
            # RGB to BGR
            image = image[..., (2, 1, 0)]

        # to uint8
        image = image.astype(np.uint8)
        image = image.copy()
        img_h, img_w = image.shape[:2]

        boxes = target["boxes"]
        labels = target["labels"]

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            cls_id = int(label)
            color = class_colors[cls_id]
            # class name
            label = categories[cls_id]
            if x2 - x1 > 0. and y2 - y1 > 0.:
                # draw bbox
                image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                # put the test on the bbox
                cv2.putText(image, label, (int(x1), int(y1 - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)
        cv2.imshow('gt', image)
        # cv2.imwrite(str(i)+'.jpg', img)
        cv2.waitKey(0)