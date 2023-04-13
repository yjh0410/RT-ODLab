from __future__ import division

import os
import argparse
from copy import deepcopy

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import distributed_utils
from utils.com_flops_params import FLOPs_and_Params
from utils.misc import ModelEMA, CollateFunc, build_dataset, build_dataloader
from utils.solver.optimizer import build_optimizer
from utils.solver.lr_scheduler import build_lr_scheduler

from engine import train_one_epoch, val_one_epoch

from config import build_model_config, build_trans_config
from models import build_model


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

    # dist
    world_size = distributed_utils.get_world_size()
    per_gpu_batch = args.batch_size // world_size
    print('World size: {}'.format(world_size))
    if args.distributed:
        distributed_utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(distributed_utils.get_sha()))

    # path to save model
    path_to_save = os.path.join(args.save_folder, args.dataset, args.model)
    os.makedirs(path_to_save, exist_ok=True)

    # cuda
    if args.cuda:
        print('use cuda')
        # cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    model_cfg = build_model_config(args)
    trans_cfg = build_trans_config(model_cfg['trans_type'])

    # dataset and evaluator
    dataset, dataset_info, evaluator = build_dataset(args, trans_cfg, device, is_train=True)
    num_classes = dataset_info[0]

    # dataloader
    dataloader = build_dataloader(args, dataset, per_gpu_batch, CollateFunc())

    # build model
    model, criterion = build_model(
        args=args, 
        model_cfg=model_cfg,
        device=device,
        num_classes=num_classes,
        trainable=True,
        )
    model = model.to(device).train()

    # DDP
    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # SyncBatchNorm
    if args.sybn and args.distributed:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # compute FLOPs and Params
    if distributed_utils.is_main_process:
        model_copy = deepcopy(model_without_ddp)
        model_copy.trainable = False
        model_copy.eval()
        FLOPs_and_Params(model=model_copy, 
                         img_size=args.img_size, 
                         device=device)
        del model_copy
    if args.distributed:
        # wait for all processes to synchronize
        dist.barrier()

    # amp
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # batch size
    total_bs = args.batch_size
    accumulate = max(1, round(64 / total_bs))
    print('Grad_Accumulate: ', accumulate)

    # optimizer
    model_cfg['weight_decay'] *= total_bs * accumulate / 64
    optimizer, start_epoch = build_optimizer(model_cfg, model_without_ddp, model_cfg['lr0'], args.resume)

    # Scheduler
    scheduler, lf = build_lr_scheduler(model_cfg, optimizer, args.max_epoch)
    scheduler.last_epoch = start_epoch - 1  # do not move
    if args.resume:
        scheduler.step()

    # EMA
    if args.ema and distributed_utils.get_rank() in [-1, 0]:
        print('Build ModelEMA ...')
        ema = ModelEMA(model, decay=model_cfg['ema_decay'], tau=model_cfg['ema_tau'], updates=start_epoch * len(dataloader))
    else:
        ema = None

    # start training loop
    best_map = -1.0
    last_opt_step = -1
    total_epochs = args.max_epoch + args.wp_epoch
    heavy_eval = False
    optimizer.zero_grad()
    
    # eval before training
    if args.eval_first and distributed_utils.is_main_process():
        # to check whether the evaluator can work
        model_eval = ema.ema if ema else model_without_ddp
        val_one_epoch(
            args=args, model=model_eval, evaluator=evaluator, optimizer=optimizer,
            epoch=0, best_map=best_map, path_to_save=path_to_save)

    # start to train
    for epoch in range(start_epoch, total_epochs):
        if args.distributed:
            dataloader.batch_sampler.sampler.set_epoch(epoch)

        # check second stage
        if epoch >= (total_epochs - model_cfg['no_aug_epoch'] - 1):
            # close mosaic augmentation
            if dataloader.dataset.mosaic_prob > 0.:
                print('close Mosaic Augmentation ...')
                dataloader.dataset.mosaic_prob = 0.
                heavy_eval = True
            # close mixup augmentation
            if dataloader.dataset.mixup_prob > 0.:
                print('close Mixup Augmentation ...')
                dataloader.dataset.mixup_prob = 0.
                heavy_eval = True

        # train one epoch
        last_opt_step = train_one_epoch(
            epoch=epoch,
            total_epochs=total_epochs,
            args=args, 
            device=device,
            ema=ema, 
            model=model,
            criterion=criterion,
            cfg=model_cfg, 
            dataloader=dataloader, 
            optimizer=optimizer,
            scheduler=scheduler,
            lf=lf,
            scaler=scaler,
            last_opt_step=last_opt_step)

        # eval
        if heavy_eval:
            best_map = val_one_epoch(
                            args=args, 
                            model=ema.ema if ema else model_without_ddp, 
                            evaluator=evaluator,
                            optimizer=optimizer,
                            epoch=epoch,
                            best_map=best_map,
                            path_to_save=path_to_save)
        else:
            if (epoch % args.eval_epoch) == 0 or (epoch == total_epochs - 1):
                best_map = val_one_epoch(
                                args=args, 
                                model=ema.ema if ema else model_without_ddp, 
                                evaluator=evaluator,
                                optimizer=optimizer,
                                epoch=epoch,
                                best_map=best_map,
                                path_to_save=path_to_save)

    # Empty cache after train loop
    if args.cuda:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    train()
