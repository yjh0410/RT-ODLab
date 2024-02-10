import math
import torch


# ------------------------- WarmUp LR Scheduler -------------------------
## Warmup LR Scheduler
class LinearWarmUpScheduler(object):
    def __init__(self, base_lr=0.01, wp_iter=500, warmup_factor=0.00066667):
        self.base_lr = base_lr
        self.wp_iter = wp_iter
        self.warmup_factor = warmup_factor


    def set_lr(self, optimizer, lr, base_lr):
        for param_group in optimizer.param_groups:
            init_lr = param_group['initial_lr']
            ratio = init_lr / base_lr
            param_group['lr'] = lr * ratio


    def __call__(self, iter, optimizer):
        # warmup
        alpha = iter / self.wp_iter
        warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        tmp_lr = self.base_lr * warmup_factor
        self.set_lr(optimizer, tmp_lr, self.base_lr)
        
## Build WP LR Scheduler
def build_wp_lr_scheduler(cfg, base_lr=0.01):
    print('==============================')
    print('WarmUpScheduler: {}'.format(cfg['warmup']))
    print('--base_lr: {}'.format(base_lr))
    print('--warmup_iters: {}'.format(cfg['warmup_iters']))
    print('--warmup_factor: {}'.format(cfg['warmup_factor']))

    if cfg['warmup'] == 'linear':
        wp_lr_scheduler = LinearWarmUpScheduler(base_lr, cfg['warmup_iters'], cfg['warmup_factor'])
    
    return wp_lr_scheduler

                           
# ------------------------- LR Scheduler -------------------------
def build_lr_scheduler(cfg, optimizer, resume=None):
    print('==============================')
    print('LR Scheduler: {}'.format(cfg['lr_scheduler']))

    if cfg['lr_scheduler'] == 'step':
        assert 'lr_epoch' in cfg
        print('--lr_epoch: {}'.format(cfg['lr_epoch']))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=cfg['lr_epoch'])
    elif cfg['lr_scheduler'] == 'cosine':
        pass
        
    if resume is not None and resume.lower() != "none":
        print('keep training: ', resume)
        checkpoint = torch.load(resume)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("lr_scheduler")
        lr_scheduler.load_state_dict(checkpoint_state_dict)

    return lr_scheduler


def build_lambda_lr_scheduler(cfg, optimizer, epochs):
    """Build learning rate scheduler from cfg file."""
    print('==============================')
    print('Lr Scheduler: {}'.format(cfg['scheduler']))
    # Cosine LR scheduler
    if cfg['scheduler'] == 'cosine':
        lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (cfg['lrf'] - 1) + 1
    # Linear LR scheduler
    elif cfg['scheduler'] == 'linear':
        lf = lambda x: (1 - x / epochs) * (1.0 - cfg['lrf']) + cfg['lrf']

    else:
        print('unknown lr scheduler.')
        exit(0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    return scheduler, lf
