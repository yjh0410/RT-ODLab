import torch
import torch.distributed as dist

import time
import os
import math
import numpy as np
import random

from utils import distributed_utils
from utils.vis_tools import vis_data


def rescale_image_targets(images, targets, max_stride, min_box_size):
    """
        Deployed for Multi scale trick.
    """
    # During training phase, the shape of input image is square.
    old_img_size = images.shape[-1]
    new_img_size = random.randrange(old_img_size * 0.5, old_img_size * 1.5 + max_stride) // max_stride * max_stride  # size
    if new_img_size / old_img_size != 1:
        # interpolate
        images = torch.nn.functional.interpolate(
                            input=images, 
                            size=new_img_size, 
                            mode='bilinear', 
                            align_corners=False)
        # rescale targets
        for tgt in targets:
            boxes = tgt["boxes"].clone()
            labels = tgt["labels"].clone()
            boxes = torch.clamp(boxes, 0, old_img_size)
            # rescale box
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / old_img_size * new_img_size
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / old_img_size * new_img_size
            # refine tgt
            tgt_boxes_wh = boxes[..., 2:] - boxes[..., :2]
            min_tgt_size = torch.min(tgt_boxes_wh, dim=-1)[0]
            keep = (min_tgt_size >= min_box_size)

            tgt["boxes"] = boxes[keep]
            tgt["labels"] = labels[keep]

    return images, targets, new_img_size


def train_one_epoch(epoch,
                    total_epochs,
                    args, 
                    device, 
                    ema,
                    model,
                    criterion,
                    cfg, 
                    dataloader, 
                    optimizer,
                    scheduler,
                    lf,
                    scaler,
                    last_opt_step):
    epoch_size = len(dataloader)
    img_size = args.img_size
    t0 = time.time()
    nw = epoch_size * args.wp_epoch
    accumulate = accumulate = max(1, round(64 / args.batch_size))

    # train one epoch
    for iter_i, (images, targets) in enumerate(dataloader):
        ni = iter_i + epoch * epoch_size
        # Warmup
        if ni <= nw:
            xi = [0, nw]  # x interp
            accumulate = max(1, np.interp(ni, xi, [1, 64 / args.batch_size]).round())
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(
                    ni, xi, [cfg['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [cfg['warmup_momentum'], cfg['momentum']])
                            
        # visualize train targets
        if args.vis_tgt:
            vis_data(images, targets)

        # to device
        images = images.to(device, non_blocking=True).float() / 255.

        # multi scale
        if args.multi_scale:
            images, targets, img_size = rescale_image_targets(
                images, targets, max(model.stride), args.min_box_size)
            
        # inference
        with torch.cuda.amp.autocast(enabled=args.fp16):
            outputs = model(images)
            # loss
            loss_dict = criterion(outputs=outputs, targets=targets)
            losses = loss_dict['losses']
            losses *= images.shape[0]  # loss * bs

            # reduce            
            loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

            if args.distributed:
                # gradient averaged between devices in DDP mode
                losses *= distributed_utils.get_world_size()

        # check loss
        try:
            if torch.isnan(losses):
                print('loss is NAN !!')
                continue
        except:
            print(loss_dict)

        # backward
        scaler.scale(losses).backward()

        # Optimize
        if ni - last_opt_step >= accumulate:
            if cfg['clip_grad'] > 0:
                # unscale gradients
                scaler.unscale_(optimizer)
                # clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg['clip_grad'])
            # optimizer.step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            # ema
            if ema:
                ema.update(model)
            last_opt_step = ni

        # display
        if distributed_utils.is_main_process() and iter_i % 10 == 0:
            t1 = time.time()
            cur_lr = [param_group['lr']  for param_group in optimizer.param_groups]
            # basic infor
            log =  '[Epoch: {}/{}]'.format(epoch+1, total_epochs)
            log += '[Iter: {}/{}]'.format(iter_i, epoch_size)
            log += '[lr: {:.6f}]'.format(cur_lr[2])
            # loss infor
            for k in loss_dict_reduced.keys():
                if k == 'losses' and args.distributed:
                    world_size = distributed_utils.get_world_size()
                    log += '[{}: {:.2f}]'.format(k, loss_dict[k] / world_size)
                else:
                    log += '[{}: {:.2f}]'.format(k, loss_dict[k])

            # other infor
            log += '[time: {:.2f}]'.format(t1 - t0)
            log += '[size: {}]'.format(img_size)

            # print log infor
            print(log, flush=True)
            
            t0 = time.time()
    
    scheduler.step()

    return last_opt_step


def val_one_epoch(args, 
                  model, 
                  evaluator,
                  optimizer,
                  epoch,
                  best_map,
                  path_to_save):
    # check evaluator
    if distributed_utils.is_main_process():
        if evaluator is None:
            print('No evaluator ... save model and go on training.')
            print('Saving state, epoch: {}'.format(epoch + 1))
            weight_name = '{}_epoch_{}.pth'.format(args.version, epoch + 1)
            checkpoint_path = os.path.join(path_to_save, weight_name)
            torch.save({'model': model.state_dict(),
                        'mAP': -1.,
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'args': args}, 
                        checkpoint_path)                      
            
        else:
            print('eval ...')
            # set eval mode
            model.trainable = False
            model.eval()

            # evaluate
            evaluator.evaluate(model)

            cur_map = evaluator.map
            if cur_map > best_map:
                # update best-map
                best_map = cur_map
                # save model
                print('Saving state, epoch:', epoch + 1)
                weight_name = '{}_epoch_{}_{:.2f}.pth'.format(args.version, epoch + 1, best_map*100)
                checkpoint_path = os.path.join(path_to_save, weight_name)
                torch.save({'model': model.state_dict(),
                            'mAP': round(best_map*100, 1),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch,
                            'args': args}, 
                            checkpoint_path)                      

            # set train mode.
            model.trainable = True
            model.train()

    if args.distributed:
        # wait for all processes to synchronize
        dist.barrier()

    return best_map
