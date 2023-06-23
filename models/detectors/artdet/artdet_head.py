import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .artdet_basic import Conv
except:
    from artdet_basic import Conv


class DecoupledHead(nn.Module):
    def __init__(self, cfg, in_dim, out_dim, num_classes=80):
        super().__init__()
        print('==============================')
        print('Head: Decoupled Head')
        # --------- Basic Parameters ----------
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.reg_max = cfg['reg_max']
        self.num_cls_head=cfg['num_cls_head']
        self.num_reg_head=cfg['num_reg_head']

        # --------- Network Parameters ----------
        ## cls head
        cls_feats = []
        self.cls_out_dim = out_dim
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
        self.reg_out_dim = out_dim
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

        ## Pred
        self.cls_pred = nn.Conv2d(self.cls_out_dim, num_classes, kernel_size=1) 
        self.reg_pred = nn.Conv2d(self.reg_out_dim, 4*cfg['reg_max'], kernel_size=1) 

        ## ----------- proj_conv ------------
        self.proj = nn.Parameter(torch.linspace(0, cfg['reg_max'], cfg['reg_max']), requires_grad=False)
        self.proj_conv = nn.Conv2d(self.reg_max, 1, kernel_size=1, bias=False)
        self.proj_conv.weight = nn.Parameter(self.proj.view([1, cfg['reg_max'], 1, 1]).clone().detach(), requires_grad=False)


    def forward(self, x, anchors, stride):
        """
            in_feats: (Tensor) [B, C, H, W]
        """
        cls_feats = self.cls_feats(x)
        reg_feats = self.reg_feats(x)

        cls_pred = self.cls_pred(cls_feats)
        reg_pred = self.reg_pred(reg_feats)

        # process preds
        B = x.shape[0]
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4*self.reg_max)

        # ----------------------- Decode bbox -----------------------
        M = reg_pred.shape[1]
        # [B, M, 4*(reg_max)] -> [B, M, 4, reg_max] -> [B, 4, M, reg_max]
        reg_pred_ = reg_pred.reshape([B, M, 4, self.reg_max])
        # [B, M, 4, reg_max] -> [B, reg_max, 4, M]
        reg_pred_ = reg_pred_.permute(0, 3, 2, 1).contiguous()
        # [B, reg_max, 4, M] -> [B, 1, 4, M]
        reg_pred_ = self.proj_conv(F.softmax(reg_pred_, dim=1))
        # [B, 1, 4, M] -> [B, 4, M] -> [B, M, 4]
        reg_pred_ = reg_pred_.view(B, 4, M).permute(0, 2, 1).contiguous()
        ## tlbr -> xyxy
        x1y1_pred = anchors[None] - reg_pred_[..., :2] * stride
        x2y2_pred = anchors[None] + reg_pred_[..., 2:] * stride
        box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

        return cls_pred, reg_pred, box_pred
    

# build detection head
def build_head(cfg, in_dim, out_dim, num_classes=80):
    if cfg['head'] == 'decoupled_head':
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