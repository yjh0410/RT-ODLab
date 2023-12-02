import os
import cv2
import random
import numpy as np

from torch.utils.data import Dataset

try:
    from pycocotools.coco import COCO
except:
    print("It seems that the COCOAPI is not installed.")

try:
    from .data_augment.yolov5_augment import yolov5_mosaic_augment, yolov5_mixup_augment, yolox_mixup_augment
except:
    from data_augment.yolov5_augment import yolov5_mosaic_augment, yolov5_mixup_augment, yolox_mixup_augment


crowd_class_labels = ('person',)



class CrowdHumanDataset(Dataset):
    def __init__(self, 
                 img_size     :int = 640,
                 data_dir     :str = None, 
                 image_set    :str = 'train',
                 trans_config = None,
                 transform    = None,
                 is_train     :bool = False
                 ):
        # ----------- Basic parameters -----------
        self.img_size = img_size
        self.image_set = image_set
        self.is_train = is_train
        # ----------- Path parameters -----------
        self.data_dir = data_dir
        self.json_file = '{}.json'.format(image_set)
        # ----------- Data parameters -----------
        self.coco = COCO(os.path.join(self.data_dir, 'annotations', self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())

        # ----------- Transform parameters -----------
        self.transform = transform
        self.mosaic_prob = trans_config['mosaic_prob'] if trans_config else 0.0
        self.mixup_prob = trans_config['mixup_prob'] if trans_config else 0.0
        self.trans_config = trans_config
        print('==============================')
        print('use Mosaic Augmentation: {}'.format(self.mosaic_prob))
        print('use Mixup Augmentation: {}'.format(self.mixup_prob))
        print('==============================')

    # ------------ Basic dataset function ------------
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.pull_item(index)

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
        img_id = im_ann["file_name"][:-4]
        img_file = os.path.join(
                self.data_dir, 'CrowdHuman_{}'.format(self.image_set), 'Images', im_ann["file_name"])
        image = cv2.imread(img_file)

        return image, img_id

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
    
    parser = argparse.ArgumentParser(description='CrowdHuman-Dataset')

    # opt
    parser.add_argument('--root', default='/Users/liuhaoran/Desktop/python_work/object-detection/dataset/CrowdHuman/',
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

    dataset = CrowdHumanDataset(
        img_size=args.img_size,
        data_dir=args.root,
        image_set='val',
        transform=transform,
        trans_config=trans_config,
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
            label = crowd_class_labels[cls_id]
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            # put the test on the bbox
            cv2.putText(image, label, (int(x1), int(y1 - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)
        cv2.imshow('gt', image)
        # cv2.imwrite(str(i)+'.jpg', img)
        cv2.waitKey(0)
