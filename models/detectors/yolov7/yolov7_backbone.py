import torch
import torch.nn as nn

try:
    from .yolov7_basic import Conv, ELANBlock, DownSample
except:
    from yolov7_basic import Conv, ELANBlock, DownSample
    

model_urls = {
    "elannet_nano": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/yolov7_elannet_nano.pth",
    "elannet_tiny": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/yolov7_elannet_tiny.pth",
    "elannet_large": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/yolov7_elannet_large.pth",
    "elannet_huge": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/yolov7_elannet_huge.pth",
}


# --------------------- ELANNet -----------------------
# ELANNet-Nano
class ELANNet_Nano(nn.Module):
    def __init__(self, act_type='lrelu', norm_type='BN', depthwise=True):
        super(ELANNet_Nano, self).__init__()
        self.feat_dims = [64, 128, 256]
        
        # P1/2
        self.layer_1 = Conv(3, 16, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        # P2/4
        self.layer_2 = nn.Sequential(   
            Conv(16, 32, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),             
            ELANBlock(in_dim=32, out_dim=32, expand_ratio=0.5, depth=1,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P3/8
        self.layer_3 = nn.Sequential(
            nn.MaxPool2d((2, 2), 2),             
            ELANBlock(in_dim=32, out_dim=64, expand_ratio=0.5, depth=1,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P4/16
        self.layer_4 = nn.Sequential(
            nn.MaxPool2d((2, 2), 2),             
            ELANBlock(in_dim=64, out_dim=128, expand_ratio=0.5, depth=1,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P5/32
        self.layer_5 = nn.Sequential(
            nn.MaxPool2d((2, 2), 2),             
            ELANBlock(in_dim=128, out_dim=256, expand_ratio=0.5, depth=1,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )


    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        outputs = [c3, c4, c5]

        return outputs


# ELANNet-Tiny
class ELANNet_Tiny(nn.Module):
    """
    ELAN-Net of YOLOv7-Tiny.
    """
    def __init__(self, act_type='silu', norm_type='BN', depthwise=False):
        super(ELANNet_Tiny, self).__init__()
        self.feat_dims = [128, 256, 512]
        
        # P1/2
        self.layer_1 = Conv(3, 32, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        # P2/4
        self.layer_2 = nn.Sequential(   
            Conv(32, 64, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),             
            ELANBlock(in_dim=64, out_dim=64, expand_ratio=0.5, depth=1,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P3/8
        self.layer_3 = nn.Sequential(
            nn.MaxPool2d((2, 2), 2),             
            ELANBlock(in_dim=64, out_dim=128, expand_ratio=0.5, depth=1,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P4/16
        self.layer_4 = nn.Sequential(
            nn.MaxPool2d((2, 2), 2),             
            ELANBlock(in_dim=128, out_dim=256, expand_ratio=0.5, depth=1,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P5/32
        self.layer_5 = nn.Sequential(
            nn.MaxPool2d((2, 2), 2),             
            ELANBlock(in_dim=256, out_dim=512, expand_ratio=0.5, depth=1,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )


    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        outputs = [c3, c4, c5]

        return outputs


## ELANNet-Large
class ELANNet_Lagre(nn.Module):
    def __init__(self, act_type='silu', norm_type='BN', depthwise=False):
        super(ELANNet_Lagre, self).__init__()
        self.feat_dims = [512, 1024, 1024]
        
        # P1/2
        self.layer_1 = nn.Sequential(
            Conv(3, 32, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise),      
            Conv(32, 64, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            Conv(64, 64, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P2/4
        self.layer_2 = nn.Sequential(   
            Conv(64, 128, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),             
            ELANBlock(in_dim=128, out_dim=256, expand_ratio=0.5, depth=2,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P3/8
        self.layer_3 = nn.Sequential(
            DownSample(in_dim=256, out_dim=256, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            ELANBlock(in_dim=256, out_dim=512, expand_ratio=0.5, depth=2,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P4/16
        self.layer_4 = nn.Sequential(
            DownSample(in_dim=512, out_dim=512, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            ELANBlock(in_dim=512, out_dim=1024, expand_ratio=0.5, depth=2,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P5/32
        self.layer_5 = nn.Sequential(
            DownSample(in_dim=1024, out_dim=1024, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            ELANBlock(in_dim=1024, out_dim=1024, expand_ratio=0.25, depth=2,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )


    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        outputs = [c3, c4, c5]

        return outputs


## ELANNet-Huge
class ELANNet_Huge(nn.Module):
    def __init__(self, act_type='silu', norm_type='BN', depthwise=False):
        super(ELANNet_Huge, self).__init__()
        self.feat_dims = [640, 1280, 1280]
        
        # P1/2
        self.layer_1 = nn.Sequential(
            Conv(3, 40, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            Conv(40, 80, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            Conv(80, 80, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P2/4
        self.layer_2 = nn.Sequential(   
            Conv(80, 160, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            ELANBlock(in_dim=160, out_dim=320, expand_ratio=0.5, depth=3,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P3/8
        self.layer_3 = nn.Sequential(
            DownSample(in_dim=320, out_dim=320, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            ELANBlock(in_dim=320, out_dim=640, expand_ratio=0.5, depth=3,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P4/16
        self.layer_4 = nn.Sequential(
            DownSample(in_dim=640, out_dim=640, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            ELANBlock(in_dim=640, out_dim=1280, expand_ratio=0.5, depth=3,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P5/32
        self.layer_5 = nn.Sequential(
            DownSample(in_dim=1280, out_dim=1280, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            ELANBlock(in_dim=1280, out_dim=1280, expand_ratio=0.25, depth=3,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )


    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        outputs = [c3, c4, c5]

        return outputs


# --------------------- Functions -----------------------
def build_backbone(cfg, pretrained=False): 
    """Constructs a ELANNet model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # build backbone
    if cfg['backbone'] == 'elannet_huge':
        backbone = ELANNet_Huge(cfg['bk_act'], cfg['bk_norm'], cfg['bk_dpw'])
    elif cfg['backbone'] == 'elannet_large':
        backbone = ELANNet_Lagre(cfg['bk_act'], cfg['bk_norm'], cfg['bk_dpw'])
    elif cfg['backbone'] == 'elannet_tiny':
        backbone = ELANNet_Tiny(cfg['bk_act'], cfg['bk_norm'], cfg['bk_dpw'])
    elif cfg['backbone'] == 'elannet_nano':
        backbone = ELANNet_Nano(cfg['bk_act'], cfg['bk_norm'], cfg['bk_dpw'])
    # pyramid feat dims
    feat_dims = backbone.feat_dims

    # load imagenet pretrained weight
    if pretrained:
        url = model_urls[cfg['backbone']]
        if url is not None:
            print('Loading pretrained weight for {}.'.format(cfg['backbone'].upper()))
            checkpoint = torch.hub.load_state_dict_from_url(
                url=url, map_location="cpu", check_hash=True)
            # checkpoint state dict
            checkpoint_state_dict = checkpoint.pop("model")
            # model state dict
            model_state_dict = backbone.state_dict()
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

            backbone.load_state_dict(checkpoint_state_dict)
        else:
            print('No backbone pretrained: ELANNet')        

    return backbone, feat_dims


if __name__ == '__main__':
    import time
    from thop import profile
    cfg = {
        'pretrained': False,
        'backbone': 'elannet_huge',
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'p6_feat': False,
        'p7_feat': False,
    }
    model, feats = build_backbone(cfg)
    x = torch.randn(1, 3, 224, 224)
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