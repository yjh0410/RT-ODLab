import torch
import torch.nn as nn

try:
    from .ctrnet_basic import Conv, RTCBlock
except:
    from ctrnet_basic import Conv, RTCBlock


# MIM-pretrained weights
model_urls = {
    "rtcnet_n": None,
    "rtcnet_t": None,
    "rtcnet_s": None,
    "rtcnet_m": None,
    "rtcnet_l": None,
    "rtcnet_x": None,
}


# ---------------------------- Basic functions ----------------------------
## Real-time Convolutional Backbone
class CTREncoder(nn.Module):
    def __init__(self, width=1.0, depth=1.0, ratio=1.0, act_type='silu', norm_type='BN', depthwise=False):
        super(CTREncoder, self).__init__()
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
## build Backbone
def build_encoder(cfg, pretrained=False): 
    # build backbone model
    backbone = CTREncoder(width=cfg['width'],
                          depth=cfg['depth'],
                          ratio=cfg['ratio'],
                          act_type=cfg['bk_act'],
                          norm_type=cfg['bk_norm'],
                          depthwise=cfg['bk_depthwise']
                          )
    feat_dims = backbone.feat_dims[-3:]

    # load pretrained weight
    if pretrained:
        backbone = load_pretrained_weight(backbone)
        
    return backbone, feat_dims

## load pretrained weight
def load_pretrained_weight(model):
    # Model name
    width, depth, ratio = model.width_factor, model.depth_factor, model.last_stage_factor
    if width == 0.25 and depth == 0.34 and ratio == 2.0:
        model_name = "rtcnet_n"
    elif width == 0.375 and depth == 0.34 and ratio == 2.0:
        model_name = "rtcnet_t"
    elif width == 0.50 and depth == 0.34 and ratio == 2.0:
        model_name = "rtcnet_s"
    elif width == 0.75 and depth == 0.67 and ratio == 1.5:
        model_name = "rtcnet_m"
    elif width == 1.0 and depth == 1.0 and ratio == 1.0:
        model_name = "rtcnet_l"
    elif width == 1.25 and depth == 1.34 and ratio == 1.0:
        model_name = "rtcnet_x"
    
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
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'width': 1.0,
        'depth': 1.0,
        'ratio': 1.0,
    }
    model, feats = build_encoder(cfg)
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