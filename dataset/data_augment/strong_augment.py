import random
import cv2
import numpy as np

from .yolov5_augment import random_perspective


# ------------------------- Strong augmentations -------------------------
## Mosaic Augmentation
class MosaicAugment(object):
    def __init__(self,
                 img_size,
                 transform_config,
                 is_train=False,
                 ) -> None:
        self.img_size = img_size
        self.is_train = is_train
        self.affine_params = transform_config['affine_params']
        self.mosaic_type   = transform_config['mosaic_type']

    def yolov5_mosaic_augment(self, image_list, target_list):
        assert len(image_list) == 4

        mosaic_img = np.ones([self.img_size*2, self.img_size*2, image_list[0].shape[2]], dtype=np.uint8) * 114
        # mosaic center
        yc, xc = [int(random.uniform(-x, 2*self.img_size + x)) for x in [-self.img_size // 2, -self.img_size // 2]]
        # yc = xc = self.img_size

        mosaic_bboxes = []
        mosaic_labels = []
        for i in range(4):
            img_i, target_i = image_list[i], target_list[i]
            bboxes_i = target_i["boxes"]
            labels_i = target_i["labels"]

            orig_h, orig_w, _ = img_i.shape

            # resize
            r = self.img_size / max(orig_h, orig_w)
            if r != 1: 
                interp = cv2.INTER_LINEAR if (self.is_train or r > 1) else cv2.INTER_AREA
                img_i = cv2.resize(img_i, (int(orig_w * r), int(orig_h * r)), interpolation=interp)
            h, w, _ = img_i.shape

            # place img in img4
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.img_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.img_size * 2), min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            mosaic_img[y1a:y2a, x1a:x2a] = img_i[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            # labels
            bboxes_i_ = bboxes_i.copy()
            if len(bboxes_i) > 0:
                # a valid target, and modify it.
                bboxes_i_[:, 0] = (w * bboxes_i[:, 0] / orig_w + padw)
                bboxes_i_[:, 1] = (h * bboxes_i[:, 1] / orig_h + padh)
                bboxes_i_[:, 2] = (w * bboxes_i[:, 2] / orig_w + padw)
                bboxes_i_[:, 3] = (h * bboxes_i[:, 3] / orig_h + padh)    

                mosaic_bboxes.append(bboxes_i_)
                mosaic_labels.append(labels_i)

        if len(mosaic_bboxes) == 0:
            mosaic_bboxes = np.array([]).reshape(-1, 4)
            mosaic_labels = np.array([]).reshape(-1)
        else:
            mosaic_bboxes = np.concatenate(mosaic_bboxes)
            mosaic_labels = np.concatenate(mosaic_labels)

        # clip
        mosaic_bboxes = mosaic_bboxes.clip(0, self.img_size * 2)

        # random perspective
        mosaic_targets = np.concatenate([mosaic_labels[..., None], mosaic_bboxes], axis=-1)
        mosaic_img, mosaic_targets = random_perspective(
            mosaic_img,
            mosaic_targets,
            self.affine_params['degrees'],
            translate=self.affine_params['translate'],
            scale=self.affine_params['scale'],
            shear=self.affine_params['shear'],
            perspective=self.affine_params['perspective'],
            border=[-self.img_size//2, -self.img_size//2]
            )

        # target
        mosaic_target = {
            "boxes": mosaic_targets[..., 1:],
            "labels": mosaic_targets[..., 0],
            "orig_size": [self.img_size, self.img_size]
        }

        return mosaic_img, mosaic_target

    def __call__(self, image_list, target_list):
        if self.mosaic_type == 'yolov5':
            return self.yolov5_mosaic_augment(image_list, target_list)
        else:
            raise NotImplementedError("Unknown mosaic type: {}".format(self.mosaic_type))

## Mixup Augmentation
class MixupAugment(object):
    def __init__(self,
                 img_size,
                 transform_config,
                 ) -> None:
        self.img_size = img_size
        self.mixup_type  = transform_config['mixup_type']
        self.mixup_scale = transform_config['mixup_scale']

    def yolov5_mixup_augment(self, origin_image, origin_target, new_image, new_target):
        if origin_image.shape[:2] != new_image.shape[:2]:
            img_size = max(new_image.shape[:2])
            # origin_image is not a mosaic image
            orig_h, orig_w = origin_image.shape[:2]
            scale_ratio = img_size / max(orig_h, orig_w)
            if scale_ratio != 1: 
                interp = cv2.INTER_LINEAR if scale_ratio > 1 else cv2.INTER_AREA
                resize_size = (int(orig_w * scale_ratio), int(orig_h * scale_ratio))
                origin_image = cv2.resize(origin_image, resize_size, interpolation=interp)

            # pad new image
            pad_origin_image = np.ones([img_size, img_size, origin_image.shape[2]], dtype=np.uint8) * 114
            pad_origin_image[:resize_size[1], :resize_size[0]] = origin_image
            origin_image = pad_origin_image.copy()
            del pad_origin_image

        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        mixup_image = r * origin_image.astype(np.float32) + \
                    (1.0 - r)* new_image.astype(np.float32)
        mixup_image = mixup_image.astype(np.uint8)
        
        cls_labels = new_target["labels"].copy()
        box_labels = new_target["boxes"].copy()

        mixup_bboxes = np.concatenate([origin_target["boxes"], box_labels], axis=0)
        mixup_labels = np.concatenate([origin_target["labels"], cls_labels], axis=0)

        mixup_target = {
            "boxes": mixup_bboxes,
            "labels": mixup_labels,
            'orig_size': mixup_image.shape[:2]
        }
        
        return mixup_image, mixup_target

    def yolox_mixup_augment(self, origin_image, origin_target, new_image, new_target):
        assert self.mixup_scale is not None, "You should set mixup_scale as a List type, such as [0.5, 1.5], not a NoneType."

        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5

        # resize new image
        orig_h, orig_w = new_image.shape[:2]
        cp_scale_ratio = self.img_size / max(orig_h, orig_w)
        if cp_scale_ratio != 1: 
            interp = cv2.INTER_LINEAR if cp_scale_ratio > 1 else cv2.INTER_AREA
            resized_new_img = cv2.resize(
                new_image, (int(orig_w * cp_scale_ratio), int(orig_h * cp_scale_ratio)), interpolation=interp)
        else:
            resized_new_img = new_image

        # pad new image
        cp_img = np.ones([self.img_size, self.img_size, new_image.shape[2]], dtype=np.uint8) * 114
        new_shape = (resized_new_img.shape[1], resized_new_img.shape[0])
        cp_img[:new_shape[1], :new_shape[0]] = resized_new_img

        # resize padded new image
        cp_img_h, cp_img_w = cp_img.shape[:2]
        cp_new_shape = (int(cp_img_w * jit_factor),
                        int(cp_img_h * jit_factor))
        cp_img = cv2.resize(cp_img, (cp_new_shape[0], cp_new_shape[1]))
        cp_scale_ratio *= jit_factor

        # flip new image
        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        # pad image
        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_image.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img

        # crop padded image
        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]

        # process target
        new_boxes = new_target["boxes"]
        new_labels = new_target["labels"]
        new_boxes[:, 0::2] = np.clip(new_boxes[:, 0::2] * cp_scale_ratio, 0, origin_w)
        new_boxes[:, 1::2] = np.clip(new_boxes[:, 1::2] * cp_scale_ratio, 0, origin_h)
        if FLIP:
            new_boxes[:, 0::2] = (
                origin_w - new_boxes[:, 0::2][:, ::-1]
            )
        new_boxes[:, 0::2] = np.clip(
            new_boxes[:, 0::2] - x_offset, 0, target_w
        )
        new_boxes[:, 1::2] = np.clip(
            new_boxes[:, 1::2] - y_offset, 0, target_h
        )

        # mixup target
        mixup_boxes = np.concatenate([new_boxes, origin_target['boxes']], axis=0)
        mixup_labels = np.concatenate([new_labels, origin_target['labels']], axis=0)
        mixup_target = {
            'boxes': mixup_boxes,
            'labels': mixup_labels
        }

        # mixup images
        origin_image = origin_image.astype(np.float32)
        origin_image = 0.5 * origin_image + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_image.astype(np.uint8), mixup_target
            
    def __call__(self, origin_image, origin_target, new_image, new_target):
        if self.mixup_type == "yolov5":
            return self.yolov5_mixup_augment(origin_image, origin_target, new_image, new_target)
        elif self.mixup_type == "yolox":
            return self.yolox_mixup_augment(origin_image, origin_target, new_image, new_target)
        else:
            raise NotImplementedError("Unknown mixup type: {}".format(self.mixup_type))
