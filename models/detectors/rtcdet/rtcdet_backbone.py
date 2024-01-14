import torch
import torch.nn as nn

try:
    from .rtcdet_basic import Conv, RTCBlock
except:
    from rtcdet_basic import Conv, RTCBlock


# Pretrained weights
model_urls = {
    # ImageNet-1K pretrained weight
    "rtcnet_n": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elan_cspnet_nano.pth",
    "rtcnet_s": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elan_cspnet_small.pth",
    "rtcnet_m": None,
    "rtcnet_l": None,
    "rtcnet_x": None,
    # MIM-pretrained weights
    "mae_rtcnet_n": None,
    "mae_rtcnet_s": None,
    "mae_rtcnet_m": None,
    "mae_rtcnet_l": None,
    "mae_rtcnet_x": None,
}


# ---------------------------- Basic functions ----------------------------
## Real-time Convolutional Backbone
class RTCBackbone(nn.Module):
    def __init__(self, width=1.0, depth=1.0, ratio=1.0, act_type='silu', norm_type='BN', depthwise=False):
        super(RTCBackbone, self).__init__()
        # ---------------- Basic parameters ----------------
        self.width_factor = width
        self.depth_factor = depth
        self.last_stage_factor = ratio
        self.feat_dims = [round(64 * width), round(128 * width), round(256 * width), round(512 * width), round(512 * width * ratio)]
        # ---------------- Network parameters ----------------
        ## P1/2
        self.layer_1 = Conv(3, self.feat_dims[0], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type)
        ## P2/4
        self.layer_2 = nn.Sequential(
            Conv(self.feat_dims[0], self.feat_dims[1], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            RTCBlock(in_dim     = self.feat_dims[1],
                     out_dim    = self.feat_dims[1],
                     num_blocks = round(3*depth),
                     shortcut   = True,
                     act_type   = act_type,
                     norm_type  = norm_type,
                     depthwise  = depthwise)
        )
        ## P3/8
        self.layer_3 = nn.Sequential(
            Conv(self.feat_dims[1], self.feat_dims[2], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            RTCBlock(in_dim     = self.feat_dims[2],
                     out_dim    = self.feat_dims[2],
                     num_blocks = round(6*depth),
                     shortcut   = True,
                     act_type   = act_type,
                     norm_type  = norm_type,
                     depthwise  = depthwise)
        )
        ## P4/16
        self.layer_4 = nn.Sequential(
            Conv(self.feat_dims[2], self.feat_dims[3], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            RTCBlock(in_dim     = self.feat_dims[3],
                     out_dim    = self.feat_dims[3],
                     num_blocks = round(6*depth),
                     shortcut   = True,
                     act_type   = act_type,
                     norm_type  = norm_type,
                     depthwise  = depthwise)
        )
        ## P5/32
        self.layer_5 = nn.Sequential(
            Conv(self.feat_dims[3], self.feat_dims[4], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            RTCBlock(in_dim     = self.feat_dims[4],
                     out_dim    = self.feat_dims[4],
                     num_blocks = round(3*depth),
                     shortcut   = True,
                     act_type   = act_type,
                     norm_type  = norm_type,
                     depthwise  = depthwise)
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
## Build Backbone network
def build_backbone(cfg, pretrained=False): 
    # build backbone model
    backbone = RTCBackbone(width=cfg['width'],
                           depth=cfg['depth'],
                           ratio=cfg['ratio'],
                           act_type=cfg['bk_act'],
                           norm_type=cfg['bk_norm'],
                           depthwise=cfg['bk_depthwise']
                           )
    feat_dims = backbone.feat_dims[-3:]

    # Model name
    width, depth, ratio = cfg['width'], cfg['depth'], cfg['ratio']
    model_name = "{}" if not cfg['bk_pretrained_mae'] else "mae_{}"
    if  width == 0.25   and depth == 0.34 and ratio == 2.0:
        model_name = model_name.format("rtcnet_n")
    elif width == 0.375 and depth == 0.34 and ratio == 2.0:
        model_name = model_name.format("rtcnet_t")
    elif width == 0.50  and depth == 0.34 and ratio == 2.0:
        model_name = model_name.format("rtcnet_s")
    elif width == 0.75  and depth == 0.67 and ratio == 1.5:
        model_name = model_name.format("rtcnet_m")
    elif width == 1.0   and depth == 1.0  and ratio == 1.0:
        model_name = model_name.format("rtcnet_l")
    elif width == 1.25  and depth == 1.34  and ratio == 1.0:
        model_name = model_name.format("rtcnet_x")
    else:
        raise NotImplementedError("No such model size : width={}, depth={}, ratio={}. ".format(width, depth, ratio))

    # Load pretrained weight
    if pretrained:
        backbone = load_pretrained_weight(backbone, model_name)
        
    return backbone, feat_dims

## Load pretrained weight
def load_pretrained_weight(model, model_name):
    # Load pretrained weight
    url = model_urls[model_name]
    if url is not None:
        print('Loading pretrained weight ...')
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
        # load the weight
        model.load_state_dict(checkpoint_state_dict)
    else:
        print('No backbone pretrained for {}.'.format(model_name))

    return model


if __name__ == '__main__':
    import time
    from thop import profile
    cfg = {
        'bk_pretrained': True,
        'bk_pretrained_mae': False,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'width': 0.25,
        'depth': 0.34,
        'ratio': 2.0,
    }
    model, feats = build_backbone(cfg, pretrained=cfg['bk_pretrained'])
    x = torch.randn(1, 3, 640, 640)
    t0 = time.time()
    outputs = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    for out in outputs:
        print(out.shape)

    x = torch.randn(1, 3, 640, 640)
    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))