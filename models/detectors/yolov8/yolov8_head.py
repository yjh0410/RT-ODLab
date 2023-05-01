import torch
import torch.nn as nn
try:
    from .yolov8_basic import Conv
except:
    from yolov8_basic import Conv


class DecoupledHead(nn.Module):
    def __init__(self, cfg, in_dim, fpn_dims, num_classes=80):
        super().__init__()
        print('==============================')
        print('Head: Decoupled Head')
        self.in_dim = in_dim
        self.num_cls_head=cfg['num_cls_head']
        self.num_reg_head=cfg['num_reg_head']
        self.act_type=cfg['head_act']
        self.norm_type=cfg['head_norm']

        # cls head
        cls_feats = []
        self.cls_out_dim = max(fpn_dims[0], num_classes)
        for i in range(cfg['num_cls_head']):
            if i == 0:
                cls_feats.append(
                    Conv(in_dim, self.cls_out_dim, k=3, p=1, s=1, 
                        act_type=self.act_type,
                        norm_type=self.norm_type,
                        depthwise=cfg['head_depthwise'])
                        )
            else:
                cls_feats.append(
                    Conv(self.cls_out_dim, self.cls_out_dim, k=3, p=1, s=1, 
                        act_type=self.act_type,
                        norm_type=self.norm_type,
                        depthwise=cfg['head_depthwise'])
                        )
                
        # reg head
        reg_feats = []
        self.reg_out_dim = max(16, fpn_dims[0]//4, 4*cfg['reg_max'])
        for i in range(cfg['num_reg_head']):
            if i == 0:
                reg_feats.append(
                    Conv(in_dim, self.reg_out_dim, k=3, p=1, s=1, 
                        act_type=self.act_type,
                        norm_type=self.norm_type,
                        depthwise=cfg['head_depthwise'])
                        )
            else:
                reg_feats.append(
                    Conv(self.reg_out_dim, self.reg_out_dim, k=3, p=1, s=1, 
                        act_type=self.act_type,
                        norm_type=self.norm_type,
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
def build_head(cfg, in_dim, max_dim, num_classes=80):
    head = DecoupledHead(cfg, in_dim, max_dim, num_classes) 

    return head


if __name__ == '__main__':
    import time
    from thop import profile
    cfg = {
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_act': 'silu',
        'head_norm': 'BN',
        'head_depthwise': False,
        'reg_max': 16,
    }
    fpn_dims = [256, 512, 512]
    # Head-1
    model = build_head(cfg, 256, fpn_dims, num_classes=80)
    x = torch.randn(1, 256, 80, 80)
    t0 = time.time()
    outputs = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    # for out in outputs:
    #     print(out.shape)

    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('==============================')
    print('Head-1: GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Head-1: Params : {:.2f} M'.format(params / 1e6))

    # Head-2
    model = build_head(cfg, 512, fpn_dims, num_classes=80)
    x = torch.randn(1, 512, 40, 40)
    t0 = time.time()
    outputs = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    # for out in outputs:
    #     print(out.shape)

    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('==============================')
    print('Head-2: GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Head-2: Params : {:.2f} M'.format(params / 1e6))

    # Head-3
    model = build_head(cfg, 512, fpn_dims, num_classes=80)
    x = torch.randn(1, 512, 20, 20)
    t0 = time.time()
    outputs = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    # for out in outputs:
    #     print(out.shape)

    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('==============================')
    print('Head-3: GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Head-3: Params : {:.2f} M'.format(params / 1e6))