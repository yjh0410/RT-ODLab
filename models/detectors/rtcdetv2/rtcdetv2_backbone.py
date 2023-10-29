import torch
import torch.nn as nn

try:
    from .rtcdetv2_basic import Conv, ResXStage
except:
    from rtcdetv2_basic import Conv, ResXStage
    
model_urls = {
    'resxnet_pico':   None,
    'resxnet_nano':   None,
    'resxnet_tiny':   None,
    'resxnet_small':  None,
    'resxnet_medium': None,
    'resxnet_large':  None,
    'resxnet_huge':   None,
}

# --------------------- ResXNet -----------------------
class ResXNet(nn.Module):
    def __init__(self,
                 embed_dim    = 96,
                 expand_ratio = 0.25,
                 ffn_ratio    = 4.0,
                 num_branches = 4,
                 num_stages   = [3, 3, 9, 3],
                 act_type     = 'silu',
                 norm_type    = 'BN',
                 depthwise    = False):
        super(ResXNet, self).__init__()
        # ------------------ Basic parameters ------------------
        self.embed_dim = embed_dim
        self.expand_ratio = expand_ratio
        self.ffn_ratio = ffn_ratio
        self.num_branches = num_branches
        self.num_stages = num_stages
        self.feat_dims = [embed_dim * 2, embed_dim * 4, embed_dim * 8]
        
        # ------------------ Network parameters ------------------
        ## P2/4
        self.layer_1 = nn.Sequential(
            Conv(3, embed_dim, k=7, p=3, s=2, act_type=act_type, norm_type=norm_type),
            nn.MaxPool2d((3, 3), stride=2, padding=1)
        )
        self.layer_2 = ResXStage(embed_dim, embed_dim, self.expand_ratio, self.ffn_ratio, self.num_branches, self.num_stages[0], True, act_type, norm_type, depthwise)
        ## P3/8
        self.layer_3 = nn.Sequential(
            Conv(embed_dim, embed_dim*2, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),             
            ResXStage(embed_dim*2, embed_dim*2, self.expand_ratio, self.ffn_ratio, self.num_branches, self.num_stages[1], True, act_type, norm_type, depthwise)
        )
        ## P4/16
        self.layer_4 = nn.Sequential(
            Conv(embed_dim*2, embed_dim*4, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),             
            ResXStage(embed_dim*4, embed_dim*4, self.expand_ratio, self.ffn_ratio, self.num_branches, self.num_stages[2], True, act_type, norm_type, depthwise)
        )
        ## P5/32
        self.layer_5 = nn.Sequential(
            Conv(embed_dim*4, embed_dim*8, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),             
            ResXStage(embed_dim*8, embed_dim*8, self.expand_ratio, self.ffn_ratio, self.num_branches, self.num_stages[3], True, act_type, norm_type, depthwise)
        )

    def forward(self, x):
        c2 = self.layer_1(x)
        c2 = self.layer_2(c2)
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

## build ELAN-Net
def build_backbone(cfg, pretrained=False): 
    # model
    backbone = ResXNet(
        embed_dim=cfg['embed_dim'],
        expand_ratio=cfg['expand_ratio'],
        ffn_ratio=cfg['ffn_ratio'],
        num_branches=cfg['num_branches'],
        num_stages=cfg['num_stages'],
        act_type=cfg['bk_act'],
        norm_type=cfg['bk_norm'],
        depthwise=cfg['bk_depthwise']
        )
    # check whether to load imagenet pretrained weight
    if pretrained:
        if cfg['width'] == 0.25 and cfg['depth'] == 0.34 and cfg['bk_depthwise']:
            backbone = load_weight(backbone, model_name='resxnet_pico')
        elif cfg['width'] == 0.25 and cfg['depth'] == 0.34:
            backbone = load_weight(backbone, model_name='resxnet_nano')
        elif cfg['width'] == 0.375 and cfg['depth'] == 0.34:
            backbone = load_weight(backbone, model_name='resxnet_tiny')
        elif cfg['width'] == 0.5 and cfg['depth'] == 0.34:
            backbone = load_weight(backbone, model_name='resxnet_small')
        elif cfg['width'] == 0.75 and cfg['depth'] == 0.67:
            backbone = load_weight(backbone, model_name='resxnet_medium')
        elif cfg['width'] == 1.0 and cfg['depth'] == 1.0:
            backbone = load_weight(backbone, model_name='resxnet_large')
        elif cfg['width'] == 1.25 and cfg['depth'] == 1.34:
            backbone = load_weight(backbone, model_name='resxnet_huge')

    return backbone, backbone.feat_dims


if __name__ == '__main__':
    import time
    from thop import profile
    cfg = {
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'embed_dim': 96,
        'expand_ratio': 0.25,
        'ffn_ratio': 4.0,
        'num_branches': 4,
        'num_stages'  : [3, 3, 9, 3],
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