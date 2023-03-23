import math
import torch


def build_lr_scheduler(args, cfg, optimizer, max_epochs):
    """Build learning rate scheduler from cfg file."""
    print('==============================')
    print('Lr Scheduler: {}'.format(cfg['scheduler']))

    if cfg['scheduler'] == 'cosine':
        lf = lambda x: ((1 - math.cos(x * math.pi / max_epochs)) / 2) * (cfg['lrf'] - 1) + 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        
    elif cfg['scheduler'] == 'linear':
        lf = lambda x: (1 - x / max_epochs) * (1.0 - cfg['lrf']) + cfg['lrf']
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    elif cfg['scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.step_epoch, gamma=0.1)

    else:
        print('unknown lr scheduler.')
        exit(0)


    return scheduler
