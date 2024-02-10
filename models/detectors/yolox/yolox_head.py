import torch
import torch.nn as nn

try:
    from .yolox_basic import Conv
except:
    from  yolox_basic import Conv


class DecoupledHead(nn.Module):
    def __init__(self, cfg, in_dim, out_dim, num_classes=80):
        super().__init__()
        print('==============================')
        print('Head: Decoupled Head')
        # --------- Basic Parameters ----------
        self.in_dim = in_dim
        self.num_cls_head=cfg['num_cls_head']
        self.num_reg_head=cfg['num_reg_head']

        # --------- Network Parameters ----------
        ## cls head
        cls_feats = []
        self.cls_out_dim = max(out_dim, num_classes)
        for i in range(cfg['num_cls_head']):
            if i == 0:
                cls_feats.append(
                    Conv(in_dim, self.cls_out_dim, k=3, p=1, s=1, 
                        act_type=cfg['head_act'],
                        norm_type=cfg['head_norm'],
                        depthwise=cfg['head_depthwise'])
                        )
            else:
                cls_feats.append(
                    Conv(self.cls_out_dim, self.cls_out_dim, k=3, p=1, s=1, 
                        act_type=cfg['head_act'],
                        norm_type=cfg['head_norm'],
                        depthwise=cfg['head_depthwise'])
                        )      
        ## reg head
        reg_feats = []
        self.reg_out_dim = max(out_dim, 64)
        for i in range(cfg['num_reg_head']):
            if i == 0:
                reg_feats.append(
                    Conv(in_dim, self.reg_out_dim, k=3, p=1, s=1, 
                        act_type=cfg['head_act'],
                        norm_type=cfg['head_norm'],
                        depthwise=cfg['head_depthwise'])
                        )
            else:
                reg_feats.append(
                    Conv(self.reg_out_dim, self.reg_out_dim, k=3, p=1, s=1, 
                        act_type=cfg['head_act'],
                        norm_type=cfg['head_norm'],
                        depthwise=cfg['head_depthwise'])
                        )

        self.cls_feats = nn.Sequential(*cls_feats)
        self.reg_feats = nn.Sequential(*reg_feats)


    def forward(self, x):
        """
            in_feats: (Tensor) [B, C, H, W]
        """
        cls_feats = self.cls_feats(x)
        reg_feats = self.reg_feats(x)

        return cls_feats, reg_feats
    

# build detection head
def build_head(cfg, in_dim, out_dim, num_classes=80):
    head = DecoupledHead(cfg, in_dim, out_dim, num_classes) 

    return head


if __name__ == '__main__':
    import time
    from thop import profile
    cfg = {
        'head': 'decoupled_head',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_act': 'silu',
        'head_norm': 'BN',
        'head_depthwise': False,
    }
    fpn_dims = [256, 256, 256]
    cls_out_dim = 256
    reg_out_dim = 256
    # Head-1
    model = build_head(cfg, 256, 256, num_classes=80)
    print(model)
    fpn_feats = [torch.randn(1, fpn_dims[0], 80, 80), torch.randn(1, fpn_dims[1], 40, 40), torch.randn(1, fpn_dims[2], 20, 20)]
    t0 = time.time()
    outputs = model(fpn_feats)
    t1 = time.time()
    print('Time: ', t1 - t0)
    # for out in outputs:
    #     print(out.shape)

    print('==============================')
    flops, params = profile(model, inputs=(fpn_feats, ), verbose=False)
    print('==============================')
    print('Head-1: GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Head-1: Params : {:.2f} M'.format(params / 1e6))
