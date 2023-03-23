
# Build warmup scheduler


def build_warmup(cfg, base_lr=0.01, wp_iter=500):
    print('==============================')
    print('WarmUpScheduler: {}'.format(cfg['warmup']))
    print('--base_lr: {}'.format(base_lr))
    print('--warmup_factor: {}'.format(cfg['warmup_factor']))
    print('--wp_iter: {}'.format(wp_iter))

    warmup_scheduler = WarmUpScheduler(name=cfg['warmup'], 
                                       base_lr=base_lr, 
                                       wp_iter=wp_iter, 
                                       warmup_factor=cfg['warmup_factor'])
    
    return warmup_scheduler

                           
# Basic Warmup Scheduler
class WarmUpScheduler(object):
    def __init__(self, 
                 name='linear', 
                 base_lr=0.01, 
                 wp_iter=500, 
                 warmup_factor=0.00066667):
        self.name = name
        self.base_lr = base_lr
        self.wp_iter = wp_iter
        self.warmup_factor = warmup_factor


    def set_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def warmup(self, iter, optimizer):
        # warmup
        assert iter < self.wp_iter
        if self.name == 'exp':
            tmp_lr = self.base_lr * pow(iter / self.wp_iter, 4)
            self.set_lr(optimizer, tmp_lr)

        elif self.name == 'linear':
            alpha = iter / self.wp_iter
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            tmp_lr = self.base_lr * warmup_factor
            self.set_lr(optimizer, tmp_lr)


    def __call__(self, iter, optimizer):
        self.warmup(iter, optimizer)
        