import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

import numpy as np
import os
import math
from copy import deepcopy

from evaluator.coco_evaluator import COCOAPIEvaluator
from evaluator.voc_evaluator import VOCAPIEvaluator
from dataset.voc import VOCDetection, VOC_CLASSES
from dataset.coco import COCODataset, coco_class_index, coco_class_labels
from dataset.data_augment import build_transform

from utils import fuse_conv_bn
from models.yolov7.yolov7_basic import RepConv


# ---------------------------- For Dataset ----------------------------
## build dataset
def build_dataset(args, trans_config, device, is_train=False):
    # transform
    print('==============================')
    print('Transform Config: {}'.format(trans_config))
    train_transform = build_transform(args.img_size, trans_config, True)
    val_transform = build_transform(args.img_size, trans_config, False)

    # dataset
    if args.dataset == 'voc':
        data_dir = os.path.join(args.root, 'VOCdevkit')
        num_classes = 20
        class_names = VOC_CLASSES
        class_indexs = None

        # dataset
        dataset = VOCDetection(
            img_size=args.img_size,
            data_dir=data_dir,
            image_sets=[('2007', 'trainval'), ('2012', 'trainval')] if is_train else [('2007', 'test')],
            transform=train_transform,
            trans_config=trans_config,
            is_train=is_train
            )
        
        # evaluator
        evaluator = VOCAPIEvaluator(
            data_dir=data_dir,
            device=device,  
            transform=val_transform
            )

    elif args.dataset == 'coco':
        data_dir = os.path.join(args.root, 'COCO')
        num_classes = 80
        class_names = coco_class_labels
        class_indexs = coco_class_index

        # dataset
        dataset = COCODataset(
            img_size=args.img_size,
            data_dir=data_dir,
            image_set='train2017' if is_train else 'val2017',
            transform=train_transform,
            trans_config=trans_config,
            is_train=is_train
            )
        # evaluator
        evaluator = COCOAPIEvaluator(
            data_dir=data_dir,
            device=device,
            transform=val_transform
            )

    else:
        print('unknow dataset !! Only support voc, coco !!')
        exit(0)

    print('==============================')
    print('Training model on:', args.dataset)
    print('The dataset size:', len(dataset))

    return dataset, (num_classes, class_names, class_indexs), evaluator

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


# ---------------------------- For Model ----------------------------
## load trained weight
def load_weight(model, path_to_ckpt, fuse_cbn=False, fuse_repconv=False):
    # check ckpt file
    if path_to_ckpt is None:
        print('no weight file ...')
    else:
        checkpoint = torch.load(path_to_ckpt, map_location='cpu')
        checkpoint_state_dict = checkpoint.pop("model")
        model.load_state_dict(checkpoint_state_dict)

        print('Finished loading model!')

    # fuse repconv
    if fuse_repconv:
        print('Fusing RepConv block ...')
        for m in model.modules():
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()

    # fuse conv & bn
    if fuse_cbn:
        print('Fusing Conv & BN ...')
        model = fuse_conv_bn.fuse_conv_bn(model)

    return model

## Model EMA
class ModelEMA(object):
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(self.de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
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
