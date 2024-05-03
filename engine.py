import torch
import torch.distributed as dist

import time
import os
import numpy as np
import random

# ----------------- Extra Components -----------------
from utils import distributed_utils
from utils.misc import ModelEMA, CollateFunc, build_dataloader
from utils.misc import MetricLogger, SmoothedValue
from utils.vis_tools import vis_data

# ----------------- Evaluator Components -----------------
from evaluator.build import build_evluator

# ----------------- Optimizer & LrScheduler Components -----------------
from utils.solver.optimizer import build_optimizer
from utils.solver.lr_scheduler import build_lambda_lr_scheduler

# ----------------- Dataset Components -----------------
from dataset.build import build_dataset, build_transform


# ----------------------- Det trainers -----------------------
## Trainer for general YOLO series
class YoloTrainer(object):
    def __init__(self, args, data_cfg, model_cfg, trans_cfg, device, model, criterion, world_size):
        # ------------------- basic parameters -------------------
        self.args = args
        self.epoch = 0
        self.best_map = -1.
        self.device = device
        self.criterion = criterion
        self.world_size = world_size
        self.grad_accumulate = args.grad_accumulate
        self.clip_grad = 35
        self.heavy_eval = False
        # weak augmentatino stage
        self.second_stage = False
        self.second_stage_epoch = args.no_aug_epoch
        # path to save model
        self.path_to_save = os.path.join(args.save_folder, args.dataset, args.model)
        os.makedirs(self.path_to_save, exist_ok=True)

        # ---------------------------- Hyperparameters refer to RTMDet ----------------------------
        self.optimizer_dict = {'optimizer': 'adamw', 'momentum': None, 'weight_decay': 5e-2, 'lr0': 0.001}
        self.ema_dict = {'ema_decay': 0.9998, 'ema_tau': 2000}
        self.lr_schedule_dict = {'scheduler': 'linear', 'lrf': 0.01}
        self.warmup_dict = {'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1}        

        # ---------------------------- Build Dataset & Model & Trans. Config ----------------------------
        self.data_cfg  = data_cfg
        self.model_cfg = model_cfg
        self.trans_cfg = trans_cfg

        # ---------------------------- Build Transform ----------------------------
        self.train_transform, self.trans_cfg = build_transform(
            args=args, trans_config=self.trans_cfg, max_stride=self.model_cfg['max_stride'], is_train=True)
        self.val_transform, _ = build_transform(
            args=args, trans_config=self.trans_cfg, max_stride=self.model_cfg['max_stride'], is_train=False)

        # ---------------------------- Build Dataset & Dataloader ----------------------------
        self.dataset, self.dataset_info = build_dataset(args, self.data_cfg, self.trans_cfg, self.train_transform, is_train=True)
        self.train_loader = build_dataloader(args, self.dataset, self.args.batch_size // self.world_size, CollateFunc())

        # ---------------------------- Build Evaluator ----------------------------
        self.evaluator = build_evluator(args, self.data_cfg, self.val_transform, self.device)

        # ---------------------------- Build Grad. Scaler ----------------------------
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.fp16)

        # ---------------------------- Build Optimizer ----------------------------
        self.optimizer_dict['lr0'] *= args.batch_size * self.grad_accumulate / 64
        self.optimizer, self.start_epoch = build_optimizer(self.optimizer_dict, model, args.resume)

        # ---------------------------- Build LR Scheduler ----------------------------
        self.lr_scheduler, self.lf = build_lambda_lr_scheduler(self.lr_schedule_dict, self.optimizer, args.max_epoch)
        self.lr_scheduler.last_epoch = self.start_epoch - 1  # do not move
        if self.args.resume and self.args.resume != 'None':
            self.lr_scheduler.step()

        # ---------------------------- Build Model-EMA ----------------------------
        if self.args.ema and distributed_utils.get_rank() in [-1, 0]:
            print('Build ModelEMA ...')
            self.model_ema = ModelEMA(self.ema_dict, model, self.start_epoch * len(self.train_loader))
        else:
            self.model_ema = None

    def train(self, model):
        for epoch in range(self.start_epoch, self.args.max_epoch):
            if self.args.distributed:
                self.train_loader.batch_sampler.sampler.set_epoch(epoch)

            # check second stage
            if epoch >= (self.args.max_epoch - self.second_stage_epoch - 1) and not self.second_stage:
                self.check_second_stage()
                # save model of the last mosaic epoch
                weight_name = '{}_last_mosaic_epoch.pth'.format(self.args.model)
                checkpoint_path = os.path.join(self.path_to_save, weight_name)
                print('Saving state of the last Mosaic epoch-{}.'.format(self.epoch))
                torch.save({'model': model.state_dict(),
                            'mAP': round(self.evaluator.map*100, 1),
                            'optimizer': self.optimizer.state_dict(),
                            'epoch': self.epoch,
                            'args': self.args}, 
                            checkpoint_path)

            # train one epoch
            self.epoch = epoch
            self.train_one_epoch(model)

            # eval one epoch
            if self.heavy_eval:
                model_eval = model.module if self.args.distributed else model
                self.eval(model_eval)
            else:
                model_eval = model.module if self.args.distributed else model
                if (epoch % self.args.eval_epoch) == 0 or (epoch == self.args.max_epoch - 1):
                    self.eval(model_eval)

            if self.args.debug:
                print("For debug mode, we only train 1 epoch")
                break

    def eval(self, model):
        # chech model
        model_eval = model if self.model_ema is None else self.model_ema.ema

        if distributed_utils.is_main_process():
            # check evaluator
            if self.evaluator is None:
                print('No evaluator ... save model and go on training.')
                print('Saving state, epoch: {}'.format(self.epoch))
                weight_name = '{}_no_eval.pth'.format(self.args.model)
                checkpoint_path = os.path.join(self.path_to_save, weight_name)
                torch.save({'model': model_eval.state_dict(),
                            'mAP': -1.,
                            'optimizer': self.optimizer.state_dict(),
                            'epoch': self.epoch,
                            'args': self.args}, 
                            checkpoint_path)               
            else:
                print('eval ...')
                # set eval mode
                model_eval.trainable = False
                model_eval.eval()

                # evaluate
                with torch.no_grad():
                    self.evaluator.evaluate(model_eval)

                # save model
                cur_map = self.evaluator.map
                if cur_map > self.best_map:
                    # update best-map
                    self.best_map = cur_map
                    # save model
                    print('Saving state, epoch:', self.epoch)
                    weight_name = '{}_best.pth'.format(self.args.model)
                    checkpoint_path = os.path.join(self.path_to_save, weight_name)
                    torch.save({'model': model_eval.state_dict(),
                                'mAP': round(self.best_map*100, 1),
                                'optimizer': self.optimizer.state_dict(),
                                'epoch': self.epoch,
                                'args': self.args}, 
                                checkpoint_path)                      

                # set train mode.
                model_eval.trainable = True
                model_eval.train()

        if self.args.distributed:
            # wait for all processes to synchronize
            dist.barrier()

    def train_one_epoch(self, model):
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('size', SmoothedValue(window_size=1, fmt='{value:d}'))
        metric_logger.add_meter('grad_norm', SmoothedValue(window_size=1, fmt='{value:.1f}'))
        header = 'Epoch: [{} / {}]'.format(self.epoch, self.args.max_epoch)
        epoch_size = len(self.train_loader)
        print_freq = 10
        grad_norm  = 0.0

        # basic parameters
        epoch_size = len(self.train_loader)
        img_size = self.args.img_size
        nw = epoch_size * self.args.wp_epoch

        # Train one epoch
        for iter_i, (images, targets) in enumerate(metric_logger.log_every(self.train_loader, print_freq, header)):
            ni = iter_i + self.epoch * epoch_size
            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                for j, x in enumerate(self.optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(
                        ni, xi, [self.warmup_dict['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * self.lf(self.epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [self.warmup_dict['warmup_momentum'], self.optimizer_dict['momentum']])
                                
            # To device
            images = images.to(self.device, non_blocking=True).float()

            # Multi scale
            if self.args.multi_scale:
                images, targets, img_size = self.rescale_image_targets(
                    images, targets, self.model_cfg['stride'], self.args.min_box_size, self.model_cfg['multi_scale'])
            else:
                targets = self.refine_targets(targets, self.args.min_box_size)
                
            # Visualize train targets
            if self.args.vis_tgt:
                vis_data(images*255, targets)

            # Inference
            with torch.cuda.amp.autocast(enabled=self.args.fp16):
                outputs = model(images)
                # Compute loss
                loss_dict = self.criterion(outputs=outputs, targets=targets, epoch=self.epoch)
                losses = loss_dict['losses']
                # Grad Accumulate
                if self.grad_accumulate > 1:
                    losses /= self.grad_accumulate

                loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

            # Backward
            self.scaler.scale(losses).backward()

            # Optimize
            if ni % self.grad_accumulate == 0:
                if self.clip_grad > 0:
                    # unscale gradients
                    self.scaler.unscale_(self.optimizer)
                    # clip gradients
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_grad)
                # optimizer.step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                # ema
                if self.model_ema is not None:
                    self.model_ema.update(model)

            # Update log
            metric_logger.update(**loss_dict_reduced)
            metric_logger.update(lr=self.optimizer.param_groups[2]["lr"])
            metric_logger.update(grad_norm=grad_norm)
            metric_logger.update(size=img_size)

            if self.args.debug:
                print("For debug mode, we only train 1 iteration")
                break

        # LR Schedule
        self.lr_scheduler.step()

        # Gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)

    def refine_targets(self, targets, min_box_size):
        # rescale targets
        for tgt in targets:
            boxes = tgt["boxes"].clone()
            labels = tgt["labels"].clone()
            # refine tgt
            tgt_boxes_wh = boxes[..., 2:] - boxes[..., :2]
            min_tgt_size = torch.min(tgt_boxes_wh, dim=-1)[0]
            keep = (min_tgt_size >= min_box_size)

            tgt["boxes"] = boxes[keep]
            tgt["labels"] = labels[keep]
        
        return targets

    def rescale_image_targets(self, images, targets, stride, min_box_size, multi_scale_range=[0.5, 1.5]):
        """
            Deployed for Multi scale trick.
        """
        if isinstance(stride, int):
            max_stride = stride
        elif isinstance(stride, list):
            max_stride = max(stride)

        # During training phase, the shape of input image is square.
        old_img_size = images.shape[-1]
        min_img_size = old_img_size * multi_scale_range[0]
        max_img_size = old_img_size * multi_scale_range[1]

        # Choose a new image size
        new_img_size = random.randrange(min_img_size, max_img_size + max_stride, max_stride)

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

    def check_second_stage(self):
        # set second stage
        print('============== Second stage of Training ==============')
        self.second_stage = True

        # close mosaic augmentation
        if self.train_loader.dataset.mosaic_prob > 0.:
            print(' - Close < Mosaic Augmentation > ...')
            self.train_loader.dataset.mosaic_prob = 0.
            self.heavy_eval = True

        # close mixup augmentation
        if self.train_loader.dataset.mixup_prob > 0.:
            print(' - Close < Mixup Augmentation > ...')
            self.train_loader.dataset.mixup_prob = 0.
            self.heavy_eval = True

        # close rotation augmentation
        if 'degrees' in self.trans_cfg.keys() and self.trans_cfg['degrees'] > 0.0:
            print(' - Close < degress of rotation > ...')
            self.trans_cfg['degrees'] = 0.0
        if 'shear' in self.trans_cfg.keys() and self.trans_cfg['shear'] > 0.0:
            print(' - Close < shear of rotation >...')
            self.trans_cfg['shear'] = 0.0
        if 'perspective' in self.trans_cfg.keys() and self.trans_cfg['perspective'] > 0.0:
            print(' - Close < perspective of rotation > ...')
            self.trans_cfg['perspective'] = 0.0

        # build a new transform for second stage
        print(' - Rebuild transforms ...')
        self.train_transform, self.trans_cfg = build_transform(
            args=self.args, trans_config=self.trans_cfg, max_stride=self.model_cfg['max_stride'], is_train=True)
        self.train_loader.dataset.transform = self.train_transform
        
## Customed Trainer for YOLOX series
class YoloxTrainer(object):
    def __init__(self, args, data_cfg, model_cfg, trans_cfg, device, model, criterion, world_size):
        # ------------------- basic parameters -------------------
        self.args = args
        self.epoch = 0
        self.best_map = -1.
        self.device = device
        self.criterion = criterion
        self.world_size = world_size
        self.grad_accumulate = args.grad_accumulate
        self.no_aug_epoch = args.no_aug_epoch
        self.heavy_eval = False
        # weak augmentatino stage
        self.second_stage = False
        self.second_stage_epoch = args.no_aug_epoch
        # path to save model
        self.path_to_save = os.path.join(args.save_folder, args.dataset, args.model)
        os.makedirs(self.path_to_save, exist_ok=True)

        # ---------------------------- Hyperparameters refer to YOLOX ----------------------------
        self.optimizer_dict = {'optimizer': 'sgd', 'momentum': 0.9, 'weight_decay': 5e-4, 'lr0': 0.01}
        self.ema_dict = {'ema_decay': 0.9999, 'ema_tau': 2000}
        self.lr_schedule_dict = {'scheduler': 'cosine', 'lrf': 0.05}
        self.warmup_dict = {'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1}        

        # ---------------------------- Build Dataset & Model & Trans. Config ----------------------------
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.trans_cfg = trans_cfg

        # ---------------------------- Build Transform ----------------------------
        self.train_transform, self.trans_cfg = build_transform(
            args=self.args, trans_config=self.trans_cfg, max_stride=self.model_cfg['max_stride'], is_train=True)
        self.val_transform, _ = build_transform(
            args=self.args, trans_config=self.trans_cfg, max_stride=self.model_cfg['max_stride'], is_train=False)

        # ---------------------------- Build Dataset & Dataloader ----------------------------
        self.dataset, self.dataset_info = build_dataset(self.args, self.data_cfg, self.trans_cfg, self.train_transform, is_train=True)
        self.train_loader = build_dataloader(self.args, self.dataset, self.args.batch_size // self.world_size, CollateFunc())

        # ---------------------------- Build Evaluator ----------------------------
        self.evaluator = build_evluator(self.args, self.data_cfg, self.val_transform, self.device)

        # ---------------------------- Build Grad. Scaler ----------------------------
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.fp16)

        # ---------------------------- Build Optimizer ----------------------------
        self.optimizer_dict['lr0'] *= self.args.batch_size * self.grad_accumulate / 64
        self.optimizer, self.start_epoch = build_optimizer(self.optimizer_dict, model, self.args.resume)

        # ---------------------------- Build LR Scheduler ----------------------------
        self.lr_scheduler, self.lf = build_lambda_lr_scheduler(self.lr_schedule_dict, self.optimizer, self.args.max_epoch - self.no_aug_epoch)
        self.lr_scheduler.last_epoch = self.start_epoch - 1  # do not move
        if self.args.resume and self.args.resume != 'None':
            self.lr_scheduler.step()

        # ---------------------------- Build Model-EMA ----------------------------
        if self.args.ema and distributed_utils.get_rank() in [-1, 0]:
            print('Build ModelEMA ...')
            self.model_ema = ModelEMA(self.ema_dict, model, self.start_epoch * len(self.train_loader))
        else:
            self.model_ema = None


    def train(self, model):
        for epoch in range(self.start_epoch, self.args.max_epoch):
            if self.args.distributed:
                self.train_loader.batch_sampler.sampler.set_epoch(epoch)

            # check second stage
            if epoch >= (self.args.max_epoch - self.second_stage_epoch - 1) and not self.second_stage:
                self.check_second_stage()
                # save model of the last mosaic epoch
                weight_name = '{}_last_mosaic_epoch.pth'.format(self.args.model)
                checkpoint_path = os.path.join(self.path_to_save, weight_name)
                print('Saving state of the last Mosaic epoch-{}.'.format(self.epoch))
                torch.save({'model': model.state_dict(),
                            'mAP': round(self.evaluator.map*100, 1),
                            'optimizer': self.optimizer.state_dict(),
                            'epoch': self.epoch,
                            'args': self.args}, 
                            checkpoint_path)
                
            # train one epoch
            self.epoch = epoch
            self.train_one_epoch(model)

            # eval one epoch
            if self.heavy_eval:
                model_eval = model.module if self.args.distributed else model
                self.eval(model_eval)
            else:
                model_eval = model.module if self.args.distributed else model
                if (epoch % self.args.eval_epoch) == 0 or (epoch == self.args.max_epoch - 1):
                    self.eval(model_eval)

            if self.args.debug:
                print("For debug mode, we only train 1 epoch")
                break

    def eval(self, model):
        # chech model
        model_eval = model if self.model_ema is None else self.model_ema.ema

        if distributed_utils.is_main_process():
            # check evaluator
            if self.evaluator is None:
                print('No evaluator ... save model and go on training.')
                print('Saving state, epoch: {}'.format(self.epoch))
                weight_name = '{}_no_eval.pth'.format(self.args.model)
                checkpoint_path = os.path.join(self.path_to_save, weight_name)
                torch.save({'model': model_eval.state_dict(),
                            'mAP': -1.,
                            'optimizer': self.optimizer.state_dict(),
                            'epoch': self.epoch,
                            'args': self.args}, 
                            checkpoint_path)               
            else:
                print('eval ...')
                # set eval mode
                model_eval.trainable = False
                model_eval.eval()

                # evaluate
                with torch.no_grad():
                    self.evaluator.evaluate(model_eval)

                # save model
                cur_map = self.evaluator.map
                if cur_map > self.best_map:
                    # update best-map
                    self.best_map = cur_map
                    # save model
                    print('Saving state, epoch:', self.epoch)
                    weight_name = '{}_best.pth'.format(self.args.model)
                    checkpoint_path = os.path.join(self.path_to_save, weight_name)
                    torch.save({'model': model_eval.state_dict(),
                                'mAP': round(self.best_map*100, 1),
                                'optimizer': self.optimizer.state_dict(),
                                'epoch': self.epoch,
                                'args': self.args}, 
                                checkpoint_path)                      

                # set train mode.
                model_eval.trainable = True
                model_eval.train()

        if self.args.distributed:
            # wait for all processes to synchronize
            dist.barrier()

    def train_one_epoch(self, model):
        # basic parameters
        epoch_size = len(self.train_loader)
        img_size = self.args.img_size
        t0 = time.time()
        nw = epoch_size * self.args.wp_epoch

        # Train one epoch
        for iter_i, (images, targets) in enumerate(self.train_loader):
            ni = iter_i + self.epoch * epoch_size
            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                for j, x in enumerate(self.optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(
                        ni, xi, [self.warmup_dict['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * self.lf(self.epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [self.warmup_dict['warmup_momentum'], self.optimizer_dict['momentum']])
                                
            # To device
            images = images.to(self.device, non_blocking=True).float()

            # Multi scale
            if self.args.multi_scale and ni % 10 == 0:
                images, targets, img_size = self.rescale_image_targets(
                    images, targets, self.model_cfg['stride'], self.args.min_box_size, self.model_cfg['multi_scale'])
            else:
                targets = self.refine_targets(targets, self.args.min_box_size)
                
            # Visualize train targets
            if self.args.vis_tgt:
                vis_data(images*255, targets)

            # Inference
            with torch.cuda.amp.autocast(enabled=self.args.fp16):
                outputs = model(images)
                # Compute loss
                loss_dict = self.criterion(outputs=outputs, targets=targets, epoch=self.epoch)
                losses = loss_dict['losses']
                # Grad Accu
                if self.grad_accumulate > 1: 
                    losses /= self.grad_accumulate

                loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

            # Backward
            self.scaler.scale(losses).backward()

            # Optimize
            if ni % self.grad_accumulate == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                # ema
                if self.model_ema is not None:
                    self.model_ema.update(model)

            # Logs
            if distributed_utils.is_main_process() and iter_i % 10 == 0:
                t1 = time.time()
                cur_lr = [param_group['lr']  for param_group in self.optimizer.param_groups]
                # basic infor
                log =  '[Epoch: {}/{}]'.format(self.epoch, self.args.max_epoch)
                log += '[Iter: {}/{}]'.format(iter_i, epoch_size)
                log += '[lr: {:.6f}]'.format(cur_lr[2])
                # loss infor
                for k in loss_dict_reduced.keys():
                    loss_val = loss_dict_reduced[k]
                    if k == 'losses':
                        loss_val *= self.grad_accumulate
                    log += '[{}: {:.2f}]'.format(k, loss_val)

                # other infor
                log += '[time: {:.2f}]'.format(t1 - t0)
                log += '[size: {}]'.format(img_size)

                # print log infor
                print(log, flush=True)
                
                t0 = time.time()

            if self.args.debug:
                print("For debug mode, we only train 1 iteration")
                break

        # LR Schedule
        if not self.second_stage:
            self.lr_scheduler.step()
        
    def check_second_stage(self):
        # set second stage
        print('============== Second stage of Training ==============')
        self.second_stage = True
        self.heavy_eval = True

        # close mosaic augmentation
        if self.train_loader.dataset.mosaic_prob > 0.:
            print(' - Close < Mosaic Augmentation > ...')
            self.train_loader.dataset.mosaic_prob = 0.

        # close mixup augmentation
        if self.train_loader.dataset.mixup_prob > 0.:
            print(' - Close < Mixup Augmentation > ...')
            self.train_loader.dataset.mixup_prob = 0.

        # close rotation augmentation
        if 'degrees' in self.trans_cfg.keys() and self.trans_cfg['degrees'] > 0.0:
            print(' - Close < degress of rotation > ...')
            self.trans_cfg['degrees'] = 0.0
        if 'shear' in self.trans_cfg.keys() and self.trans_cfg['shear'] > 0.0:
            print(' - Close < shear of rotation >...')
            self.trans_cfg['shear'] = 0.0
        if 'perspective' in self.trans_cfg.keys() and self.trans_cfg['perspective'] > 0.0:
            print(' - Close < perspective of rotation > ...')
            self.trans_cfg['perspective'] = 0.0

        # close random affine
        if 'translate' in self.trans_cfg.keys() and self.trans_cfg['translate'] > 0.0:
            print(' - Close < translate of affine > ...')
            self.trans_cfg['translate'] = 0.0
        if 'scale' in self.trans_cfg.keys():
            print(' - Close < scale of affine >...')
            self.trans_cfg['scale'] = [1.0, 1.0]

        # build a new transform for second stage
        print(' - Rebuild transforms ...')
        self.train_transform, self.trans_cfg = build_transform(
            args=self.args, trans_config=self.trans_cfg, max_stride=self.model_cfg['max_stride'], is_train=True)
        self.train_loader.dataset.transform = self.train_transform
        
    def refine_targets(self, targets, min_box_size):
        # rescale targets
        for tgt in targets:
            boxes = tgt["boxes"].clone()
            labels = tgt["labels"].clone()
            # refine tgt
            tgt_boxes_wh = boxes[..., 2:] - boxes[..., :2]
            min_tgt_size = torch.min(tgt_boxes_wh, dim=-1)[0]
            keep = (min_tgt_size >= min_box_size)

            tgt["boxes"] = boxes[keep]
            tgt["labels"] = labels[keep]
        
        return targets

    def rescale_image_targets(self, images, targets, stride, min_box_size, multi_scale_range=[0.5, 1.5]):
        """
            Deployed for Multi scale trick.
        """
        if isinstance(stride, int):
            max_stride = stride
        elif isinstance(stride, list):
            max_stride = max(stride)

        # During training phase, the shape of input image is square.
        old_img_size = images.shape[-1]
        min_img_size = old_img_size * multi_scale_range[0]
        max_img_size = old_img_size * multi_scale_range[1]

        # Choose a new image size
        new_img_size = random.randrange(min_img_size, max_img_size + max_stride, max_stride)
        new_img_size = new_img_size // max_stride * max_stride  # size
        
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


# Build Trainer
def build_trainer(args, data_cfg, model_cfg, trans_cfg, device, model, criterion, world_size):
    # ----------------------- Det trainers -----------------------
    if   model_cfg['trainer_type'] == 'yolo':
        return YoloTrainer(args, data_cfg, model_cfg, trans_cfg, device, model, criterion, world_size)
    elif model_cfg['trainer_type'] == 'yolox':
        return YoloxTrainer(args, data_cfg, model_cfg, trans_cfg, device, model, criterion, world_size)
    else:
        raise NotImplementedError(model_cfg['trainer_type'])
    