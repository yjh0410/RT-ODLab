"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import os.path as osp
import random
import torch.utils.data as data
import cv2
import numpy as np
import xml.etree.ElementTree as ET

try:
    from .data_augment.yolov5_augment import yolov5_mosaic_augment, yolov5_mixup_augment, yolox_mixup_augment
except:
    from data_augment.yolov5_augment import yolov5_mosaic_augment, yolov5_mixup_augment, yolox_mixup_augment



VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


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


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, 
                 img_size=640,
                 data_dir=None,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 trans_config=None,
                 transform=None,
                 is_train=False,
                 load_cache=False
                 ):
        self.root = data_dir
        self.img_size = img_size
        self.image_set = image_sets
        self.target_transform = VOCAnnotationTransform()
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        self.is_train = is_train
        self.load_cache = load_cache
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

        # augmentation
        self.transform = transform
        self.mosaic_prob = trans_config['mosaic_prob'] if trans_config else 0.0
        self.mixup_prob = trans_config['mixup_prob'] if trans_config else 0.0
        self.trans_config = trans_config
        print('==============================')
        print('use Mosaic Augmentation: {}'.format(self.mosaic_prob))
        print('use Mixup Augmentation: {}'.format(self.mixup_prob))
        print('==============================')

        # load cache data
        if load_cache:
            self._load_cache()


    def __getitem__(self, index):
        image, target, deltas = self.pull_item(index)
        return image, target, deltas


    def __len__(self):
        return len(self.ids)


    def _load_cache(self):
        # load image cache
        self.cached_images = []
        self.cached_targets = []
        dataset_size = len(self.ids)

        print('loading data into memory ...')
        for i in range(dataset_size):
            if i % 5000 == 0:
                print("[{} / {}]".format(i, dataset_size))
            # load an image
            image, image_id = self.pull_image(i)
            orig_h, orig_w, _ = image.shape

            # resize image
            r = self.img_size / max(orig_h, orig_w)
            if r != 1: 
                interp = cv2.INTER_LINEAR
                new_size = (int(orig_w * r), int(orig_h * r))
                image = cv2.resize(image, new_size, interpolation=interp)
            img_h, img_w = image.shape[:2]
            self.cached_images.append(image)

            # load target cache
            anno = ET.parse(self._annopath % image_id).getroot()
            anno = self.target_transform(anno)
            anno = np.array(anno).reshape(-1, 5)
            boxes = anno[:, :4]
            labels = anno[:, 4]
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / orig_w * img_w
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / orig_h * img_h
            self.cached_targets.append({"boxes": boxes, "labels": labels})
        

    def load_image_target(self, index):
        if self.load_cache:
            image = self.cached_images[index]
            target = self.cached_targets[index]
            height, width, channels = image.shape
            target["orig_size"] = [height, width]
        else:
            # load an image
            img_id = self.ids[index]
            image = cv2.imread(self._imgpath % img_id)
            height, width, channels = image.shape

            # laod an annotation
            anno = ET.parse(self._annopath % img_id).getroot()
            if self.target_transform is not None:
                anno = self.target_transform(anno)

            # guard against no boxes via resizing
            anno = np.array(anno).reshape(-1, 5)
            target = {
                "boxes": anno[:, :4],
                "labels": anno[:, 4],
                "orig_size": [height, width]
            }
        
        return image, target


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
        '''Returns the original image object at index in PIL form
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR), img_id


    def pull_anno(self, index):
        '''Returns the original annotation of image at index
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt


if __name__ == "__main__":
    import argparse
    from build import build_transform
    
    parser = argparse.ArgumentParser(description='VOC-Dataset')

    # opt
    parser.add_argument('--root', default='/Users/liuhaoran/Desktop/python_work/object-detection/dataset/VOCdevkit/',
                        help='data root')
    parser.add_argument('-size', '--img_size', default=640, type=int,
                        help='input image size.')
    parser.add_argument('--mosaic', default=None, type=float,
                        help='mosaic augmentation.')
    parser.add_argument('--mixup', default=None, type=float,
                        help='mixup augmentation.')
    parser.add_argument('--is_train', action="store_true", default=False,
                        help='mixup augmentation.')
    parser.add_argument('--load_cache', action="store_true", default=False,
                        help='load cached data.')
    
    args = parser.parse_args()

    trans_config = {
        'aug_type': 'yolov5',  # optional: ssd, yolov5
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
        'mixup_prob': 1.0,
        'mosaic_type': 'yolov5_mosaic',
        'mixup_type': 'yolox_mixup',
        'mixup_scale': [0.5, 1.5]
    }
    transform, trans_cfg = build_transform(args, trans_config, 32, args.is_train)

    dataset = VOCDetection(
        img_size=args.img_size,
        data_dir=args.root,
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
        image, target, deltas = dataset.pull_item(i)
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