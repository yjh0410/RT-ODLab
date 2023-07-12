import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

import cv2
import math
import numpy as np
from copy import deepcopy
from thop import profile


# ---------------------------- For Dataset ----------------------------
## build dataloader
def build_dataloader(args, dataset, batch_size, collate_fn=None):
    # distributed
    if args.distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=True)

    dataloader = DataLoader(dataset, batch_sampler=batch_sampler_train,
                            collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    
    return dataloader
    
## collate_fn for dataloader
class CollateFunc(object):
    def __call__(self, batch):
        targets = []
        images = []

        for sample in batch:
            image = sample[0]
            target = sample[1]

            images.append(image)
            targets.append(target)

        images = torch.stack(images, 0) # [B, C, H, W]

        return images, targets


# ---------------------------- For Loss ----------------------------
## FocalLoss
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

## InverseSigmoid
def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


# ---------------------------- For Model ----------------------------
## fuse Conv & BN layer
def fuse_conv_bn(module):
    """Recursively fuse conv and bn in a module.
    During inference, the functionary of batch norm layers is turned off
    but only the mean and var alone channels are used, which exposes the
    chance to fuse it with the preceding conv layers to save computations and
    simplify network structures.
    Args:
        module (nn.Module): Module to be fused.
    Returns:
        nn.Module: Fused module.
    """
    last_conv = None
    last_conv_name = None
    
    def _fuse_conv_bn(conv, bn):
        """Fuse conv and bn into one module.
        Args:
            conv (nn.Module): Conv to be fused.
            bn (nn.Module): BN to be fused.
        Returns:
            nn.Module: Fused module.
        """
        conv_w = conv.weight
        conv_b = conv.bias if conv.bias is not None else torch.zeros_like(
            bn.running_mean)

        factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        conv.weight = nn.Parameter(conv_w *
                                factor.reshape([conv.out_channels, 1, 1, 1]))
        conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
        return conv
    for name, child in module.named_children():
        if isinstance(child,
                      (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = _fuse_conv_bn(last_conv, child)
            module._modules[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            module._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, nn.Conv2d):
            last_conv = child
            last_conv_name = name
        else:
            fuse_conv_bn(child)
    return module

## replace module
def replace_module(module, replaced_module_type, new_module_type, replace_func=None) -> nn.Module:
    """
    Replace given type in module to a new type. mostly used in deploy.

    Args:
        module (nn.Module): model to apply replace operation.
        replaced_module_type (Type): module type to be replaced.
        new_module_type (Type)
        replace_func (function): python function to describe replace logic. Defalut value None.

    Returns:
        model (nn.Module): module that already been replaced.
    """

    def default_replace_func(replaced_module_type, new_module_type):
        return new_module_type()

    if replace_func is None:
        replace_func = default_replace_func

    model = module
    if isinstance(module, replaced_module_type):
        model = replace_func(replaced_module_type, new_module_type)
    else:  # recurrsively replace
        for name, child in module.named_children():
            new_child = replace_module(child, replaced_module_type, new_module_type)
            if new_child is not child:  # child is already replaced
                model.add_module(name, new_child)

    return model

## compute FLOPs & Parameters
def compute_flops(model, img_size, device):
    x = torch.randn(1, 3, img_size, img_size).to(device)
    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))

## load trained weight
def load_weight(model, path_to_ckpt, fuse_cbn=False):
    # check ckpt file
    if path_to_ckpt is None:
        print('no weight file ...')
    else:
        checkpoint = torch.load(path_to_ckpt, map_location='cpu')
        print('--------------------------------------')
        print('Best model infor:')
        print('Epoch: {}'.format(checkpoint["epoch"]))
        print('mAP: {}'.format(checkpoint["mAP"]))
        print('--------------------------------------')
        checkpoint_state_dict = checkpoint["model"]
        model.load_state_dict(checkpoint_state_dict)

        print('Finished loading model!')

    # fuse conv & bn
    if fuse_cbn:
        print('Fusing Conv & BN ...')
        model = fuse_conv_bn(model)

    return model

## Model EMA
class ModelEMA(object):
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, cfg, model, updates=0):
        # Create EMA
        self.ema = deepcopy(self.de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: cfg['ema_decay'] * (1 - math.exp(-x / cfg['ema_tau']))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)


    def is_parallel(self, model):
        # Returns True if model is of type DP or DDP
        return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


    def de_parallel(self, model):
        # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
        return model.module if self.is_parallel(model) else model


    def copy_attr(self, a, b, include=(), exclude=()):
        # Copy attributes from b to a, options to only include [...] and to exclude [...]
        for k, v in b.__dict__.items():
            if (len(include) and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(a, k, v)


    def update(self, model):
        # Update EMA parameters
        self.updates += 1
        d = self.decay(self.updates)

        msd = self.de_parallel(model).state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()
        # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype} and model {msd[k].dtype} must be FP32'


    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        self.copy_attr(self.ema, model, include, exclude)

## SiLU
class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


# ---------------------------- NMS ----------------------------
## basic NMS
def nms(bboxes, scores, nms_thresh):
    """"Pure Python NMS."""
    x1 = bboxes[:, 0]  #xmin
    y1 = bboxes[:, 1]  #ymin
    x2 = bboxes[:, 2]  #xmax
    y2 = bboxes[:, 3]  #ymax

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # compute iou
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-10, xx2 - xx1)
        h = np.maximum(1e-10, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
        #reserve all the boundingbox whose ovr less than thresh
        inds = np.where(iou <= nms_thresh)[0]
        order = order[inds + 1]

    return keep

## class-agnostic NMS 
def multiclass_nms_class_agnostic(scores, labels, bboxes, nms_thresh):
    # nms
    keep = nms(bboxes, scores, nms_thresh)

    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return scores, labels, bboxes

## class-aware NMS 
def multiclass_nms_class_aware(scores, labels, bboxes, nms_thresh, num_classes):
    # nms
    keep = np.zeros(len(bboxes), dtype=np.int32)
    for i in range(num_classes):
        inds = np.where(labels == i)[0]
        if len(inds) == 0:
            continue
        c_bboxes = bboxes[inds]
        c_scores = scores[inds]
        c_keep = nms(c_bboxes, c_scores, nms_thresh)
        keep[inds[c_keep]] = 1

    keep = np.where(keep > 0)
    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return scores, labels, bboxes

## multi-class NMS 
def multiclass_nms(scores, labels, bboxes, nms_thresh, num_classes, class_agnostic=False):
    if class_agnostic:
        return multiclass_nms_class_agnostic(scores, labels, bboxes, nms_thresh)
    else:
        return multiclass_nms_class_aware(scores, labels, bboxes, nms_thresh, num_classes)


# ---------------------------- Processor for Deployment ----------------------------
## Pre-processer
class PreProcessor(object):
    def __init__(self, img_size):
        self.img_size = img_size
        self.input_size = [img_size, img_size]
        

    def __call__(self, image, swap=(2, 0, 1)):
        """
        Input:
            image: (ndarray) [H, W, 3] or [H, W]
            formar: color format
        """
        if len(image.shape) == 3:
            padded_img = np.ones((self.input_size[0], self.input_size[1], 3), np.float32) * 114.
        else:
            padded_img = np.ones(self.input_size, np.float32) * 114.
        # resize
        orig_h, orig_w = image.shape[:2]
        r = min(self.input_size[0] / orig_h, self.input_size[1] / orig_w)
        resize_size = (int(orig_w * r), int(orig_h * r))
        if r != 1:
            resized_img = cv2.resize(image, resize_size, interpolation=cv2.INTER_LINEAR)
        else:
            resized_img = image

        # padding
        padded_img[:resized_img.shape[0], :resized_img.shape[1]] = resized_img
        
        # [H, W, C] -> [C, H, W]
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32) / 255.


        return padded_img, r

## Post-processer
class PostProcessor(object):
    def __init__(self, num_classes, conf_thresh=0.15, nms_thresh=0.5):
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh


    def __call__(self, predictions):
        """
        Input:
            predictions: (ndarray) [n_anchors_all, 4+1+C]
        """
        bboxes = predictions[..., :4]
        scores = predictions[..., 4:]

        # scores & labels
        labels = np.argmax(scores, axis=1)                      # [M,]
        scores = scores[(np.arange(scores.shape[0]), labels)]   # [M,]

        # thresh
        keep = np.where(scores > self.conf_thresh)
        scores = scores[keep]
        labels = labels[keep]
        bboxes = bboxes[keep]

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, True)

        return bboxes, scores, labels
