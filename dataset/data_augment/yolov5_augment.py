import random
import cv2
import math
import numpy as np
import torch


# ------------------------- Basic augmentations -------------------------
## Spatial transform
def random_perspective(image,
                       targets=(),
                       degrees=10,
                       translate=.1,
                       scale=[0.1, 2.0],
                       shear=10,
                       perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = image.shape[0] + border[0] * 2  # shape(h,w,c)
    width = image.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -image.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -image.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(scale[0], scale[1])
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            image = cv2.warpPerspective(image, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            image = cv2.warpAffine(image, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        new = np.zeros((n, 4))
        # warp boxes
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        targets[:, 1:5] = new

    return image, targets

## Color transform
def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


# ------------------------- Strong augmentations -------------------------
## YOLOv5-Mosaic
def yolov5_mosaic_augment(image_list, target_list, img_size, affine_params, is_train=False):
    assert len(image_list) == 4

    mosaic_img = np.ones([img_size*2, img_size*2, image_list[0].shape[2]], dtype=np.uint8) * 114
    # mosaic center
    yc, xc = [int(random.uniform(-x, 2*img_size + x)) for x in [-img_size // 2, -img_size // 2]]
    # yc = xc = self.img_size

    mosaic_bboxes = []
    mosaic_labels = []
    for i in range(4):
        img_i, target_i = image_list[i], target_list[i]
        bboxes_i = target_i["boxes"]
        labels_i = target_i["labels"]

        orig_h, orig_w, _ = img_i.shape

        # resize
        r = img_size / max(orig_h, orig_w)
        if r != 1: 
            interp = cv2.INTER_LINEAR if (is_train or r > 1) else cv2.INTER_AREA
            img_i = cv2.resize(img_i, (int(orig_w * r), int(orig_h * r)), interpolation=interp)
        h, w, _ = img_i.shape

        # place img in img4
        if i == 0:  # top left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, img_size * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(img_size * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, img_size * 2), min(img_size * 2, yc + h)
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
    mosaic_bboxes = mosaic_bboxes.clip(0, img_size * 2)

    # random perspective
    mosaic_targets = np.concatenate([mosaic_labels[..., None], mosaic_bboxes], axis=-1)
    mosaic_img, mosaic_targets = random_perspective(
        mosaic_img,
        mosaic_targets,
        affine_params['degrees'],
        translate=affine_params['translate'],
        scale=affine_params['scale'],
        shear=affine_params['shear'],
        perspective=affine_params['perspective'],
        border=[-img_size//2, -img_size//2]
        )

    # target
    mosaic_target = {
        "boxes": mosaic_targets[..., 1:],
        "labels": mosaic_targets[..., 0],
        "orig_size": [img_size, img_size]
    }

    return mosaic_img, mosaic_target

## YOLOv5-Mixup
def yolov5_mixup_augment(origin_image, origin_target, new_image, new_target):
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
    
## YOLOX-Mixup
def yolox_mixup_augment(origin_img, origin_target, new_img, new_target, img_size, mixup_scale):
    jit_factor = random.uniform(*mixup_scale)
    FLIP = random.uniform(0, 1) > 0.5

    # resize new image
    orig_h, orig_w = new_img.shape[:2]
    cp_scale_ratio = img_size / max(orig_h, orig_w)
    if cp_scale_ratio != 1: 
        interp = cv2.INTER_LINEAR if cp_scale_ratio > 1 else cv2.INTER_AREA
        resized_new_img = cv2.resize(
            new_img, (int(orig_w * cp_scale_ratio), int(orig_h * cp_scale_ratio)), interpolation=interp)
    else:
        resized_new_img = new_img

    # pad new image
    cp_img = np.ones([img_size, img_size, new_img.shape[2]], dtype=np.uint8) * 114
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
    target_h, target_w = origin_img.shape[:2]
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
    origin_img = origin_img.astype(np.float32)
    origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

    return origin_img.astype(np.uint8), mixup_target
        

# ------------------------- Preprocessers -------------------------
## YOLOv5-style Transform for Train
class YOLOv5Augmentation(object):
    def __init__(self, 
                 img_size=640,
                 trans_config=None):
        self.trans_config = trans_config
        self.img_size = img_size


    def __call__(self, image, target, mosaic=False):
        # resize
        img_h0, img_w0 = image.shape[:2]

        r = self.img_size / max(img_h0, img_w0)
        if r != 1: 
            interp = cv2.INTER_LINEAR
            new_shape = (int(round(img_w0 * r)), int(round(img_h0 * r)))
            img = cv2.resize(image, new_shape, interpolation=interp)
        else:
            img = image

        img_h, img_w = img.shape[:2]

        # hsv augment
        augment_hsv(img, hgain=self.trans_config['hsv_h'], 
                    sgain=self.trans_config['hsv_s'], 
                    vgain=self.trans_config['hsv_v'])
        
        if not mosaic:
            # rescale bbox
            boxes_ = target["boxes"].copy()
            boxes_[:, [0, 2]] = boxes_[:, [0, 2]] / img_w0 * img_w
            boxes_[:, [1, 3]] = boxes_[:, [1, 3]] / img_h0 * img_h
            target["boxes"] = boxes_

            # spatial augment
            target_ = np.concatenate(
                (target['labels'][..., None], target['boxes']), axis=-1)
            img, target_ = random_perspective(
                img, target_,
                degrees=self.trans_config['degrees'],
                translate=self.trans_config['translate'],
                scale=self.trans_config['scale'],
                shear=self.trans_config['shear'],
                perspective=self.trans_config['perspective']
                )
            target['boxes'] = target_[..., 1:]
            target['labels'] = target_[..., 0]
        
        # random flip
        if random.random() < 0.5:
            w = img.shape[1]
            img = np.fliplr(img).copy()
            boxes = target['boxes'].copy()
            boxes[..., [0, 2]] = w - boxes[..., [2, 0]]
            target["boxes"] = boxes

        # to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()

        if target is not None:
            target["boxes"] = torch.as_tensor(target["boxes"]).float()
            target["labels"] = torch.as_tensor(target["labels"]).long()

        # pad img
        img_h0, img_w0 = img_tensor.shape[1:]
        assert max(img_h0, img_w0) <= self.img_size

        pad_image = torch.ones([img_tensor.size(0), self.img_size, self.img_size]).float() * 114.
        pad_image[:, :img_h0, :img_w0] = img_tensor
        dh = self.img_size - img_h0
        dw = self.img_size - img_w0

        return pad_image, target, [dw, dh]

## YOLOv5-style Transform for Eval
class YOLOv5BaseTransform(object):
    def __init__(self, img_size=640, max_stride=32):
        self.img_size = img_size
        self.max_stride = max_stride


    def __call__(self, image, target=None, mosaic=False):
        # resize
        img_h0, img_w0 = image.shape[:2]

        r = self.img_size / max(img_h0, img_w0)
        # r = min(r, 1.0) # only scale down, do not scale up (for better val mAP)
        if r != 1: 
            new_shape = (int(round(img_w0 * r)), int(round(img_h0 * r)))
            img = cv2.resize(image, new_shape, interpolation=cv2.INTER_LINEAR)
        else:
            img = image

        img_h, img_w = img.shape[:2]

        # to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()

        # rescale bboxes
        if target is not None:
            # rescale bbox
            boxes_ = target["boxes"].copy()
            boxes_[:, [0, 2]] = boxes_[:, [0, 2]] / img_w0 * img_w
            boxes_[:, [1, 3]] = boxes_[:, [1, 3]] / img_h0 * img_h
            target["boxes"] = boxes_

            # to tensor
            target["boxes"] = torch.as_tensor(target["boxes"]).float()
            target["labels"] = torch.as_tensor(target["labels"]).long()

        # pad img
        img_h0, img_w0 = img_tensor.shape[1:]
        dh = img_h0 % self.max_stride
        dw = img_w0 % self.max_stride
        dh = dh if dh == 0 else self.max_stride - dh
        dw = dw if dw == 0 else self.max_stride - dw
        
        pad_img_h = img_h0 + dh
        pad_img_w = img_w0 + dw
        pad_image = torch.ones([img_tensor.size(0), pad_img_h, pad_img_w]).float() * 114.
        pad_image[:, :img_h0, :img_w0] = img_tensor

        return pad_image, target, [dw, dh]
