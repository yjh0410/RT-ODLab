from __future__ import division

import os
import argparse
from copy import deepcopy

# ----------------- Torch Components -----------------
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ----------------- Extra Components -----------------
from utils import distributed_utils
from utils.misc import compute_flops
from utils.misc import ModelEMA, CollateFunc, build_dataloader

# ----------------- Evaluator Components -----------------
from evaluator.build import build_evluator

# ----------------- Optimizer & LrScheduler Components -----------------
from utils.solver.optimizer import build_optimizer
from utils.solver.lr_scheduler import build_lr_scheduler

# ----------------- Config Components -----------------
from config import build_dataset_config, build_model_config, build_trans_config

# ----------------- Dataset Components -----------------
from dataset.build import build_dataset, build_transform

# ----------------- Model Components -----------------
from models.detectors import build_model

# ----------------- Train Components -----------------
from engine import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-Tutorial')
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('-size', '--img_size', default=640, type=int, 
                        help='input image size')
    parser.add_argument('--num_workers', default=4, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='path to save weight')
    parser.add_argument('--eval_first', action='store_true', default=False,
                        help='evaluate model before training.')
    parser.add_argument('--fp16', dest="fp16", action="store_true", default=False,
                        help="Adopting mix precision training.")
    parser.add_argument('--vis_tgt', action="store_true", default=False,
                        help="visualize training data.")
    
    # Batchsize
    parser.add_argument('-bs', '--batch_size', default=16, type=int, 
                        help='batch size on all the GPUs.')

    # Epoch
    parser.add_argument('--max_epoch', default=150, type=int, 
                        help='max epoch.')
    parser.add_argument('--wp_epoch', default=1, type=int, 
                        help='warmup epoch.')
    parser.add_argument('--eval_epoch', default=10, type=int, 
                        help='after eval epoch, the model is evaluated on val dataset.')
    parser.add_argument('--step_epoch', nargs='+', default=[90, 120], type=int,
                        help='lr epoch to decay')

    # model
    parser.add_argument('-m', '--model', default='yolov1', type=str,
                        help='build yolo')
    parser.add_argument('-ct', '--conf_thresh', default=0.005, type=float,
                        help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.6, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=1000, type=int,
                        help='topk candidates for evaluation')
    parser.add_argument('-p', '--pretrained', default=None, type=str,
                        help='load pretrained weight')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='keep training')

    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc, widerface, crowdhuman')
    
    # train trick
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='Multi scale')
    parser.add_argument('--ema', action='store_true', default=False,
                        help='Model EMA')
    parser.add_argument('--min_box_size', default=8.0, type=float,
                        help='min size of target bounding box.')
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

    return parser.parse_args()


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # ---------------------------- Build DDP ----------------------------
    world_size = distributed_utils.get_world_size()
    per_gpu_batch = args.batch_size // world_size
    print('World size: {}'.format(world_size))
    if args.distributed:
        distributed_utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(distributed_utils.get_sha()))

    # ---------------------------- Build CUDA ----------------------------
    if args.cuda:
        print('use cuda')
        # cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ---------------------------- Build Dataset & Model & Trans. Config ----------------------------
    data_cfg = build_dataset_config(args)
    model_cfg = build_model_config(args)
    trans_cfg = build_trans_config(model_cfg['trans_type'])

    # ---------------------------- Build Transform ----------------------------
    train_transform, trans_cfg = build_transform(
        args=args, trans_config=trans_cfg, max_stride=model_cfg['max_stride'], is_train=True)
    val_transform, _ = build_transform(
        args=args, trans_config=trans_cfg, max_stride=model_cfg['max_stride'], is_train=False)

    # ---------------------------- Build Dataset & Dataloader ----------------------------
    dataset, dataset_info = build_dataset(args, data_cfg, trans_cfg, train_transform, is_train=True)
    train_loader = build_dataloader(args, dataset, per_gpu_batch, CollateFunc())

    # ---------------------------- Build Evaluator ----------------------------
    evaluator = build_evluator(args, data_cfg, val_transform, device)

    # ---------------------------- Build Model ----------------------------
    model, criterion = build_model(args, model_cfg, device, dataset_info['num_classes'], True)
    model = model.to(device).train()
    if args.sybn and args.distributed:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # ---------------------------- Build DDP Model ----------------------------
    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # ---------------------------- Calcute Params & GFLOPs ----------------------------
    if distributed_utils.is_main_process:
        model_copy = deepcopy(model_without_ddp)
        model_copy.trainable = False
        model_copy.eval()
        compute_flops(model=model_copy,
                      img_size=args.img_size,
                      device=device)
        del model_copy
    if args.distributed:
        # wait for all processes to synchronize
        dist.barrier()
        dist.barrier()

    # ---------------------------- Build Grad. Scaler ----------------------------
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # ---------------------------- Build Optimizer ----------------------------
    accumulate = max(1, round(64 / args.batch_size))
    print('Grad_Accumulate: ', accumulate)
    model_cfg['weight_decay'] *= args.batch_size * accumulate / 64
    optimizer, start_epoch = build_optimizer(model_cfg, model_without_ddp, model_cfg['lr0'], args.resume)

    # ---------------------------- Build LR Scheduler ----------------------------
    args.max_epoch += args.wp_epoch
    lr_scheduler, lf = build_lr_scheduler(model_cfg, optimizer, args.max_epoch)
    lr_scheduler.last_epoch = start_epoch - 1  # do not move
    if args.resume:
        lr_scheduler.step()

    # ---------------------------- Build Model-EMA ----------------------------
    if args.ema and distributed_utils.get_rank() in [-1, 0]:
        print('Build ModelEMA ...')
        model_ema = ModelEMA(model, model_cfg['ema_decay'], model_cfg['ema_tau'], start_epoch * len(train_loader))
    else:
        model_ema = None

    # ---------------------------- Build Trainer ----------------------------
    trainer = Trainer(args, device, model_cfg, model_ema, optimizer, lf, lr_scheduler, criterion, scaler)

    # start training loop
    heavy_eval = False
    optimizer.zero_grad()
    
    # --------------------------------- Main process for training ---------------------------------
    ## Eval before training
    if args.eval_first and distributed_utils.is_main_process():
        # to check whether the evaluator can work
        model_eval = model_without_ddp
        trainer.eval_one_epoch(model_eval, evaluator)

    ## Satrt Training
    for epoch in range(start_epoch, args.max_epoch):
        if args.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)

        # check second stage
        if epoch >= (args.max_epoch - model_cfg['no_aug_epoch'] - 1):
            # close mosaic augmentation
            if train_loader.dataset.mosaic_prob > 0.:
                print('close Mosaic Augmentation ...')
                train_loader.dataset.mosaic_prob = 0.
                heavy_eval = True
            # close mixup augmentation
            if train_loader.dataset.mixup_prob > 0.:
                print('close Mixup Augmentation ...')
                train_loader.dataset.mixup_prob = 0.
                heavy_eval = True

        # train one epoch
        trainer.train_one_epoch(model, train_loader)

        # eval one epoch
        if heavy_eval:
            trainer.eval_one_epoch(model_without_ddp, evaluator)
        else:
            if (epoch % args.eval_epoch) == 0 or (epoch == args.max_epoch - 1):
                trainer.eval_one_epoch(model_without_ddp, evaluator)

    # Empty cache after train loop
    del trainer
    if args.cuda:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    train()
