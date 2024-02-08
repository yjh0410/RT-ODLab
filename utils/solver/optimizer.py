import torch
import torch.nn as nn


def build_yolo_optimizer(cfg, model, resume=None):
    print('==============================')
    print('Optimizer: {}'.format(cfg['optimizer']))
    print('--base lr: {}'.format(cfg['lr0']))
    print('--momentum: {}'.format(cfg['momentum']))
    print('--weight_decay: {}'.format(cfg['weight_decay']))

    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            g[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g[0].append(v.weight)

    if cfg['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(g[2], lr=cfg['lr0'])  # adjust beta1 to momentum
    elif cfg['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(g[2], lr=cfg['lr0'], weight_decay=0.0)
    elif cfg['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(g[2], lr=cfg['lr0'], momentum=cfg['momentum'], nesterov=True)
    else:
        raise NotImplementedError('Optimizer {} not implemented.'.format(cfg['optimizer']))

    optimizer.add_param_group({'params': g[0], 'weight_decay': cfg['weight_decay']})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})                  # add g1 (BatchNorm2d weights)

    start_epoch = 0
    if resume and resume != "None":
        print('keep training: ', resume)
        checkpoint = torch.load(resume, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("optimizer")
        optimizer.load_state_dict(checkpoint_state_dict)
        start_epoch = checkpoint.pop("epoch") + 1
        del checkpoint, checkpoint_state_dict
                                                        
    return optimizer, start_epoch


def build_rtdetr_optimizer(cfg, model, resume=None):
    print('==============================')
    print('Optimizer: {}'.format(cfg['optimizer']))
    print('--base lr: {}'.format(cfg['lr0']))
    print('--weight_decay: {}'.format(cfg['weight_decay']))

    # ------------- Divide model's parameters -------------
    param_dicts = [], [], [], [], [], []
    norm_names = ["norm"] + ["norm{}".format(i) for i in range(10000)]
    for n, p in model.named_parameters():
        # Non-Backbone's learnable parameters
        if "backbone" not in n and p.requires_grad:
            if "bias" == n.split(".")[-1]:
                param_dicts[0].append(p)      # no weight decay for all layers' bias
            else:
                if n.split(".")[-2] in norm_names:
                    param_dicts[1].append(p)  # no weight decay for all NormLayers' weight
                else:
                    param_dicts[2].append(p)  # weight decay for all Non-NormLayers' weight
        # Backbone's learnable parameters
        elif "backbone" in n and p.requires_grad:
            if "bias" == n.split(".")[-1]:
                param_dicts[3].append(p)      # no weight decay for all layers' bias
            else:
                if n.split(".")[-2] in norm_names:
                    param_dicts[4].append(p)  # no weight decay for all NormLayers' weight
                else:
                    param_dicts[5].append(p)  # weight decay for all Non-NormLayers' weight

    # Non-Backbone's learnable parameters
    optimizer = torch.optim.AdamW(param_dicts[0], lr=cfg['lr0'], weight_decay=0.0)
    optimizer.add_param_group({"params": param_dicts[1], "weight_decay": 0.0})
    optimizer.add_param_group({"params": param_dicts[2], "weight_decay": cfg['weight_decay']})

    # Backbone's learnable parameters
    optimizer.add_param_group({"params": param_dicts[3], "lr": cfg['lr0'] * cfg['backbone_lr_ratio'], "weight_decay": 0.0})
    optimizer.add_param_group({"params": param_dicts[4], "lr": cfg['lr0'] * cfg['backbone_lr_ratio'], "weight_decay": 0.0})
    optimizer.add_param_group({"params": param_dicts[5], "lr": cfg['lr0'] * cfg['backbone_lr_ratio'], "weight_decay": cfg['weight_decay']})

    start_epoch = 0
    if resume and resume != 'None':
        print('keep training: ', resume)
        checkpoint = torch.load(resume)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("optimizer")
        optimizer.load_state_dict(checkpoint_state_dict)
        start_epoch = checkpoint.pop("epoch") + 1
                                                        
    return optimizer, start_epoch
