# ------------------------------------------------------------
# Data preprocessor for Real-time DETR
# ------------------------------------------------------------
import cv2
import numpy as np
from numpy import random

import torch
import torch.nn.functional as F


# ------------------------- Augmentations -------------------------
class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

## Convert color format
class ConvertColorFormat(object):
    def __init__(self, color_format='rgb'):
        self.color_format = color_format

    def __call__(self, image, target=None):
        """
        Input:
            image: (np.array) a OpenCV image with BGR color format.
            target: None
        Output:
            image: (np.array) a OpenCV image with given color format.
            target: None
        """
        # Convert color format
        if self.color_format == 'rgb':
            image = image[..., (2, 1, 0)]    # BGR -> RGB
        elif self.color_format == 'bgr':
            image = image
        else:
            raise NotImplementedError("Unknown color format: <{}>".format(self.color_format))

        return image, target

## Random Photometric Distort
class RandomPhotometricDistort(object):
    """
    Distort image w.r.t hue, saturation and exposure.
    """

    def __init__(self, hue=0.1, saturation=1.5, exposure=1.5):
        super().__init__()
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure

    def __call__(self, image: np.ndarray, target=None) -> np.ndarray:
        """
        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].

        Returns:
            ndarray: the distorted image(s).
        """
        if random.random() < 0.5:
            dhue = np.random.uniform(low=-self.hue, high=self.hue)
            dsat = self._rand_scale(self.saturation)
            dexp = self._rand_scale(self.exposure)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image = np.asarray(image, dtype=np.float32) / 255.
            image[:, :, 1] *= dsat
            image[:, :, 2] *= dexp
            H = image[:, :, 0] + dhue * 179 / 255.

            if dhue > 0:
                H[H > 1.0] -= 1.0
            else:
                H[H < 0.0] += 1.0

            image[:, :, 0] = H
            image = (image * 255).clip(0, 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
            image = np.asarray(image, dtype=np.uint8)

        return image, target

    def _rand_scale(self, upper_bound):
        """
        Calculate random scaling factor.

        Args:
            upper_bound (float): range of the random scale.
        Returns:
            random scaling factor (float) whose range is
            from 1 / s to s .
        """
        scale = np.random.uniform(low=1, high=upper_bound)
        if np.random.rand() > 0.5:
            return scale
        return 1 / scale

## Random scaling
class RandomExpand(object):
    def __init__(self, fill_value) -> None:
        self.fill_value = fill_value

    def __call__(self, image, target=None):
        if random.randint(2):
            return image, target

        height, width, channels = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.ones(
            (int(height*ratio), int(width*ratio), channels),
            dtype=image.dtype) * self.fill_value
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = target['boxes'].copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))
        target['boxes'] = boxes

        return image, target

## Random IoU based Sample Crop
class RandomSampleCrop(object):
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def intersect(self, box_a, box_b):
        max_xy = np.minimum(box_a[:, 2:], box_b[2:])
        min_xy = np.maximum(box_a[:, :2], box_b[:2])
        inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)

        return inter[:, 0] * inter[:, 1]

    def compute_iou(self, box_a, box_b):
        inter = self.intersect(box_a, box_b)
        area_a = ((box_a[:, 2]-box_a[:, 0]) *
                (box_a[:, 3]-box_a[:, 1]))  # [A,B]
        area_b = ((box_b[2]-box_b[0]) *
                (box_b[3]-box_b[1]))  # [A,B]
        union = area_a + area_b - inter
        return inter / union  # [A,B]

    def __call__(self, image, target=None):
        height, width, _ = image.shape

        # check target
        if len(target["boxes"]) == 0:
            return image, target

        while True:
            # randomly choose a mode
            sample_id = np.random.randint(len(self.sample_options))
            mode = self.sample_options[sample_id]
            if mode is None:
                return image, target

            boxes = target["boxes"]
            labels = target["labels"]

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = self.compute_iou(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                # update target
                target["boxes"] = current_boxes
                target["labels"] = current_labels

                return current_image, target

## Random JitterCrop
class RandomJitterCrop(object):
    """Jitter and crop the image and box."""
    def __init__(self, fill_value, p=0.5, jitter_ratio=0.3):
        super().__init__()
        self.p = p
        self.jitter_ratio = jitter_ratio
        self.fill_value = fill_value

    def crop(self, image, pleft, pright, ptop, pbot, output_size):
        oh, ow = image.shape[:2]

        swidth, sheight = output_size

        src_rect = [pleft, ptop, swidth + pleft,
                    sheight + ptop]  # x1,y1,x2,y2
        img_rect = [0, 0, ow, oh]
        # rect intersection
        new_src_rect = [max(src_rect[0], img_rect[0]),
                        max(src_rect[1], img_rect[1]),
                        min(src_rect[2], img_rect[2]),
                        min(src_rect[3], img_rect[3])]
        dst_rect = [max(0, -pleft),
                    max(0, -ptop),
                    max(0, -pleft) + new_src_rect[2] - new_src_rect[0],
                    max(0, -ptop) + new_src_rect[3] - new_src_rect[1]]

        # crop the image
        cropped = np.ones([sheight, swidth, 3], dtype=image.dtype) * self.fill_value
        # cropped[:, :, ] = np.mean(image, axis=(0, 1))
        cropped[dst_rect[1]:dst_rect[3], dst_rect[0]:dst_rect[2]] = \
            image[new_src_rect[1]:new_src_rect[3],
            new_src_rect[0]:new_src_rect[2]]

        return cropped

    def __call__(self, image, target=None):
        if random.random() > self.p:
            return image, target
        else:
            oh, ow = image.shape[:2]
            dw = int(ow * self.jitter_ratio)
            dh = int(oh * self.jitter_ratio)
            pleft = np.random.randint(-dw, dw)
            pright = np.random.randint(-dw, dw)
            ptop = np.random.randint(-dh, dh)
            pbot = np.random.randint(-dh, dh)

            swidth = ow - pleft - pright
            sheight = oh - ptop - pbot
            output_size = (swidth, sheight)
            # crop image
            cropped_image = self.crop(image=image,
                                    pleft=pleft, 
                                    pright=pright, 
                                    ptop=ptop, 
                                    pbot=pbot,
                                    output_size=output_size)
            # crop bbox
            if target is not None:
                bboxes = target['boxes'].copy()
                coords_offset = np.array([pleft, ptop], dtype=np.float32)
                bboxes[..., [0, 2]] = bboxes[..., [0, 2]] - coords_offset[0]
                bboxes[..., [1, 3]] = bboxes[..., [1, 3]] - coords_offset[1]
                swidth, sheight = output_size

                bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], 0, swidth - 1)
                bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], 0, sheight - 1)
                target['boxes'] = bboxes

            return cropped_image, target
    
## Random HFlip
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target=None):
        if random.random() < self.p:
            orig_h, orig_w = image.shape[:2]
            image = image[:, ::-1]
            if target is not None:
                if "boxes" in target:
                    boxes = target["boxes"].copy()
                    boxes[..., [0, 2]] = orig_w - boxes[..., [2, 0]]
                    target["boxes"] = boxes

        return image, target

## Resize tensor image
class Resize(object):
    def __init__(self, img_size=640):
        self.img_size = img_size

    def __call__(self, image, target=None):
        orig_h, orig_w = image.shape[:2]

        # resize
        image = cv2.resize(image, (self.img_size, self.img_size)).astype(np.float32)
        img_h, img_w = image.shape[:2]

        # rescale bboxes
        if target is not None:
            boxes = target["boxes"]
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / orig_w * img_w
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / orig_h * img_h
            target["boxes"] = boxes

        return image, target

## Normalize tensor image
class Normalize(object):
    def __init__(self, pixel_mean, pixel_std):
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

    def __call__(self, image, target=None):
        # normalize image
        image = (image - self.pixel_mean) / self.pixel_std

        return image, target

## Convert ndarray to torch.Tensor
class ToTensor(object):
    def __call__(self, image, target=None):        
        # Convert torch.Tensor
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()

        if target is not None:
            target["boxes"] = torch.as_tensor(target["boxes"]).float()
            target["labels"] = torch.as_tensor(target["labels"]).long()

        return image, target


# ------------------------- Preprocessers -------------------------
## Transform for Train
class RTDetrAugmentation(object):
    def __init__(self, img_size=640, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375], use_mosaic=False):
        # ----------------- Basic parameters -----------------
        self.img_size = img_size
        self.use_mosaic = use_mosaic
        self.pixel_mean = pixel_mean  # RGB format
        self.pixel_std = pixel_std    # RGB format
        self.color_format = 'rgb'
        print("================= Pixel Statistics =================")
        print("Pixel mean: {}".format(self.pixel_mean))
        print("Pixel std:  {}".format(self.pixel_std))

        # ----------------- Transforms -----------------
        if use_mosaic:
            # For use-mosaic setting, we do not use RandomSampleCrop processor.
            self.augment = Compose([
                RandomPhotometricDistort(hue=0.5, saturation=1.5, exposure=1.5),
                RandomHorizontalFlip(p=0.5),
                Resize(img_size=self.img_size),
                ConvertColorFormat(self.color_format),
                Normalize(self.pixel_mean, self.pixel_std),
                ToTensor()
            ])
        else:
            # For no-mosaic setting, we use RandomExpand & RandomSampleCrop processor.
            self.augment = Compose([
                RandomPhotometricDistort(hue=0.5, saturation=1.5, exposure=1.5),
                RandomJitterCrop(p=0.8, jitter_ratio=0.3, fill_value=self.pixel_mean[::-1]),
                RandomHorizontalFlip(p=0.5),
                Resize(img_size=self.img_size),
                ConvertColorFormat(self.color_format),
                Normalize(self.pixel_mean, self.pixel_std),
                ToTensor()
            ])

    def set_weak_augment(self):
        self.augment = Compose([
            RandomHorizontalFlip(p=0.5),
            Resize(img_size=self.img_size),
            ConvertColorFormat(self.color_format),
            Normalize(self.pixel_mean, self.pixel_std),
            ToTensor()
        ])

    def __call__(self, image, target, mosaic=False):
        orig_h, orig_w = image.shape[:2]
        ratio = [self.img_size / orig_w, self.img_size / orig_h]

        image, target = self.augment(image, target)

        return image, target, ratio


## Transform for Eval
class RTDetrBaseTransform(object):
    def __init__(self, img_size=640, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375]):
        # ----------------- Basic parameters -----------------
        self.img_size = img_size
        self.pixel_mean = pixel_mean  # RGB format
        self.pixel_std = pixel_std    # RGB format
        self.color_format = 'rgb'
        print("================= Pixel Statistics =================")
        print("Pixel mean: {}".format(self.pixel_mean))
        print("Pixel std:  {}".format(self.pixel_std))

        # ----------------- Transforms -----------------
        self.transform = Compose([
            Resize(img_size=self.img_size),
            ConvertColorFormat(self.color_format),
            Normalize(self.pixel_mean, self.pixel_std),
            ToTensor()
        ])


    def __call__(self, image, target=None, mosaic=False):
        orig_h, orig_w = image.shape[:2]
        ratio = [self.img_size / orig_w, self.img_size / orig_h]

        image, target = self.transform(image, target)

        return image, target, ratio
