import torch
import torch.nn as nn
try:
    from .rtmdet_v2_basic import Conv, MCBlock, DSBlock
except:
    from rtmdet_v2_basic import Conv, MCBlock, DSBlock



model_urls = {
    'mcnet_p': None,
    'mcnet_n': None,
    'mcnet_t': None,
    'mcnet_s': None,
    'mcnet_m': None,
    'mcnet_l': None,
    'mcnet_x': None,
}


# ---------------------------- Backbones ----------------------------
class MixedConvNet(nn.Module):
    def __init__(self, width=1.0, depth=1.0, num_heads=4, act_type='silu', norm_type='BN', depthwise=False):
        super(MixedConvNet, self).__init__()
        # ------------------ Basic parameters ------------------
        self.feat_dims_base = [64, 128, 256, 512, 1024]
        self.nblocks_base = [3, 6, 9, 3]
        self.feat_dims = [round(dim * width) for dim in self.feat_dims_base]
        self.nblocks = [round(nblock * depth) for nblock in self.nblocks_base]
        self.num_heads = num_heads
        self.act_type = act_type
        self.norm_type = norm_type
        self.depthwise = depthwise
        
        # ------------------ Network parameters ------------------
        ## P1/2
        self.layer_1 = nn.Sequential(
            Conv(3, self.feat_dims[0], k=3, p=1, s=2, act_type=self.act_type, norm_type=self.norm_type),
            Conv(self.feat_dims[0], self.feat_dims[0], k=3, p=1, act_type=self.act_type, norm_type=self.norm_type, depthwise=self.depthwise),
        )
        ## P2/4
        self.layer_2 = nn.Sequential(   
            Conv(self.feat_dims[0], self.feat_dims[1], k=3, p=1, s=2, act_type=self.act_type, norm_type=self.norm_type),
            MCBlock(self.feat_dims[1], self.feat_dims[1], self.nblocks[0], self.num_heads, True, self.act_type, self.norm_type, self.depthwise)
        )
        ## P3/8
        self.layer_3 = nn.Sequential(
            DSBlock(self.feat_dims[1], self.feat_dims[2], self.num_heads, self.act_type, self.norm_type, self.depthwise),             
            MCBlock(self.feat_dims[2], self.feat_dims[2], self.nblocks[1], self.num_heads, True, self.act_type, self.norm_type, self.depthwise)
        )
        ## P4/16
        self.layer_4 = nn.Sequential(
            DSBlock(self.feat_dims[2], self.feat_dims[3], self.num_heads, self.act_type, self.norm_type, self.depthwise),             
            MCBlock(self.feat_dims[3], self.feat_dims[3], self.nblocks[2], self.num_heads, True, self.act_type, self.norm_type, self.depthwise)
        )
        ## P5/32
        self.layer_5 = nn.Sequential(
            DSBlock(self.feat_dims[3], self.feat_dims[4], self.num_heads, self.act_type, self.norm_type, self.depthwise),             
            MCBlock(self.feat_dims[4], self.feat_dims[4], self.nblocks[3], self.num_heads, True, self.act_type, self.norm_type, self.depthwise)
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


## build MCNet
def build_backbone(cfg, pretrained=False):
    # model
    backbone = MixedConvNet(cfg['width'], cfg['depth'], cfg['bk_num_heads'], cfg['bk_act'], cfg['bk_norm'], cfg['bk_depthwise'])

    # check whether to load imagenet pretrained weight
    if pretrained:
        if cfg['width'] == 0.25 and cfg['depth'] == 0.34 and cfg['bk_depthwise']:
            backbone = load_weight(backbone, model_name='mcnet_p')
        elif cfg['width'] == 0.25 and cfg['depth'] == 0.34:
            backbone = load_weight(backbone, model_name='mcnet_n')
        elif cfg['width'] == 0.375 and cfg['depth'] == 0.34:
            backbone = load_weight(backbone, model_name='mcnet_t')
        elif cfg['width'] == 0.5 and cfg['depth'] == 0.34:
            backbone = load_weight(backbone, model_name='mcnet_s')
        elif cfg['width'] == 0.75 and cfg['depth'] == 0.67:
            backbone = load_weight(backbone, model_name='mcnet_m')
        elif cfg['width'] == 1.0 and cfg['depth'] == 1.0:
            backbone = load_weight(backbone, model_name='mcnet_l')
        elif cfg['width'] == 1.25 and cfg['depth'] == 1.34:
            backbone = load_weight(backbone, model_name='mcnet_x')
    feat_dims = backbone.feat_dims[-3:]

    return backbone, feat_dims


if __name__ == '__main__':
    import time
    from thop import profile
    cfg = {
        ## Backbone
        'backbone': 'mcnet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'bk_num_heads': 4,
        'width': 0.25,
        'depth': 0.34,
        'stride': [8, 16, 32],  # P3, P4, P5
        'max_stride': 32,
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