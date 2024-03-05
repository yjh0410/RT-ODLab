import torch


def build_optimizer(cfg, model, resume=None):
    print('==============================')
    print('Optimizer: {}'.format(cfg['optimizer']))
    print('--base lr: {}'.format(cfg['lr0']))
    print('--momentum: {}'.format(cfg['momentum']))
    print('--weight_decay: {}'.format(cfg['weight_decay']))

    # ------------- Divide model's parameters -------------
    param_dicts = [], [], []
    norm_names = ["norm"] + ["norm{}".format(i) for i in range(10000)]
    for n, p in model.named_parameters():
        if p.requires_grad:
            if "bias" == n.split(".")[-1]:
                param_dicts[0].append(p)      # no weight decay for all layers' bias
            else:
                if n.split(".")[-2] in norm_names:
                    param_dicts[1].append(p)  # no weight decay for all NormLayers' weight
                else:
                    param_dicts[2].append(p)  # weight decay for all Non-NormLayers' weight

    # Build optimizer
    if cfg['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(param_dicts[0], lr=cfg['lr0'], momentum=cfg['momentum'], weight_decay=0.0)
    elif cfg['optimizer'] =='adamw':
        optimizer = torch.optim.AdamW(param_dicts[0], lr=cfg['lr0'], weight_decay=0.0)
    else:
        raise NotImplementedError("Unknown optimizer: {}".format(cfg['optimizer']))
    
    # Add param groups
    optimizer.add_param_group({"params": param_dicts[1], "weight_decay": 0.0})
    optimizer.add_param_group({"params": param_dicts[2], "weight_decay": cfg['weight_decay']})

    start_epoch = 0
    if resume and resume != 'None':
        checkpoint = torch.load(resume)
        # checkpoint state dict
        try:
            checkpoint_state_dict = checkpoint.pop("optimizer")
            print('Load optimizer from the checkpoint: ', resume)
            optimizer.load_state_dict(checkpoint_state_dict)
            start_epoch = checkpoint.pop("epoch") + 1
            del checkpoint, checkpoint_state_dict
        except:
            print("No optimzier in the given checkpoint.")
                                                        
    return optimizer, start_epoch
