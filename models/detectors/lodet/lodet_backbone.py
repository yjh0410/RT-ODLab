import torch
import torch.nn as nn
try:
    from .lodet_basic import Conv, SMBlock, DSBlock
except:
    from lodet_basic import Conv, SMBlock, DSBlock



model_urls = {
    'smnet': None,
}


# ---------------------------- Backbones ----------------------------
class ScaleModulationNet(nn.Module):
    def __init__(self, act_type='silu', norm_type='BN', depthwise=False):
        super(ScaleModulationNet, self).__init__()
        self.feat_dims = [64, 128, 256]
        
        # P1/2
        self.layer_1 = nn.Sequential(
            Conv(3, 16, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            Conv(16, 16, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
        )

        # P2/4
        self.layer_2 = nn.Sequential(   
            DSBlock(16, 16, act_type, norm_type, depthwise),             
            SMBlock(16, 32, act_type, norm_type, depthwise)
        )
        # P3/8
        self.layer_3 = nn.Sequential(
            DSBlock(32, 32, act_type, norm_type, depthwise),             
            SMBlock(32, 64, act_type, norm_type, depthwise)
        )
        # P4/16
        self.layer_4 = nn.Sequential(
            DSBlock(64, 64, act_type, norm_type, depthwise),             
            SMBlock(64, 128, act_type, norm_type, depthwise)
        )
        # P5/32
        self.layer_5 = nn.Sequential(
            DSBlock(128, 128, act_type, norm_type, depthwise),             
            SMBlock(128, 256, act_type, norm_type, depthwise)
        )


    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        outputs = [c3, c4, c5]

        return outputs


# ---------------------------- Functions ----------------------------
## load pretrained weight
def load_weight(model, model_name):
    # load weight
    print('Loading pretrained weight ...')
    url = model_urls[model_name]
    if url is not None:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
            else:
                checkpoint_state_dict.pop(k)
                print(k)

        model.load_state_dict(checkpoint_state_dict)
    else:
        print('No pretrained for {}'.format(model_name))

    return model


## build SMnet
def build_backbone(cfg, pretrained=False): 
    # model
    backbone = ScaleModulationNet(
        act_type=cfg['bk_act'],
        norm_type=cfg['bk_norm'],
        depthwise=cfg['bk_dpw']
        )
    # check whether to load imagenet pretrained weight
    if pretrained:
        backbone = load_weight(backbone, model_name='smnet')
    feat_dims = backbone.feat_dims

    return backbone, feat_dims


if __name__ == '__main__':
    import time
    from thop import profile
    cfg = {
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': True,
    }
    model, feats = build_backbone(cfg)
    x = torch.randn(1, 3, 640, 640)
    t0 = time.time()
    outputs = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    for out in outputs:
        print(out.shape)

    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))