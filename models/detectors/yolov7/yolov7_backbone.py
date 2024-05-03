import torch
import torch.nn as nn

try:
    from .yolov7_basic import Conv, ELANBlock, DownSample
except:
    from yolov7_basic import Conv, ELANBlock, DownSample
    

model_urls = {
    "elannet_tiny": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/yolov7_elannet_tiny.pth",
    "elannet_large": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/yolov7_elannet_large.pth",
    "elannet_huge": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/yolov7_elannet_huge.pth",
}


# --------------------- ELANNet -----------------------
## ELANNet-Tiny
class ELANNet_Tiny(nn.Module):
    """
    ELAN-Net of YOLOv7-Tiny.
    """
    def __init__(self, act_type='silu', norm_type='BN', depthwise=False):
        super(ELANNet_Tiny, self).__init__()
        # -------------- Basic parameters --------------
        self.feat_dims = [32, 64, 128, 256, 512]
        self.squeeze_ratios = [0.5, 0.5, 0.5, 0.5]   # Stage-1 -> Stage-4
        self.branch_depths = [1, 1, 1, 1]            # Stage-1 -> Stage-4
        
        # -------------- Network parameters --------------
        ## P1/2
        self.layer_1 = Conv(3, self.feat_dims[0], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        ## P2/4: Stage-1
        self.layer_2 = nn.Sequential(   
            Conv(self.feat_dims[0], self.feat_dims[1], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),             
            ELANBlock(self.feat_dims[1], self.feat_dims[1], self.squeeze_ratios[0], self.branch_depths[0], act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        ## P3/8: Stage-2
        self.layer_3 = nn.Sequential(
            nn.MaxPool2d((2, 2), 2),             
            ELANBlock(self.feat_dims[1], self.feat_dims[2], self.squeeze_ratios[1], self.branch_depths[1], act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        ## P4/16: Stage-3
        self.layer_4 = nn.Sequential(
            nn.MaxPool2d((2, 2), 2),             
            ELANBlock(self.feat_dims[2], self.feat_dims[3], self.squeeze_ratios[2], self.branch_depths[2], act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        ## P5/32: Stage-4
        self.layer_5 = nn.Sequential(
            nn.MaxPool2d((2, 2), 2),             
            ELANBlock(self.feat_dims[3], self.feat_dims[4], self.squeeze_ratios[3], self.branch_depths[3], act_type=act_type, norm_type=norm_type, depthwise=depthwise)
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
        # -------------------- Basic parameters --------------------
        self.feat_dims = [32, 64, 128, 256, 512, 1024, 1024]
        self.squeeze_ratios = [0.5, 0.5, 0.5, 0.25]  # Stage-1 -> Stage-4
        self.branch_depths = [2, 2, 2, 2]            # Stage-1 -> Stage-4

        # -------------------- Network parameters --------------------
        ## P1/2
        self.layer_1 = nn.Sequential(
            Conv(3, self.feat_dims[0], k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise),      
            Conv(self.feat_dims[0], self.feat_dims[1], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            Conv(self.feat_dims[1], self.feat_dims[1], k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        ## P2/4: Stage-1
        self.layer_2 = nn.Sequential(   
            Conv(self.feat_dims[1], self.feat_dims[2], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),             
            ELANBlock(self.feat_dims[2], self.feat_dims[3], self.squeeze_ratios[0], self.branch_depths[0], act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        ## P3/8: Stage-2
        self.layer_3 = nn.Sequential(
            DownSample(self.feat_dims[3], self.feat_dims[3], act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            ELANBlock(self.feat_dims[3], self.feat_dims[4], self.squeeze_ratios[1], self.branch_depths[1], act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        ## P4/16: Stage-3
        self.layer_4 = nn.Sequential(
            DownSample(self.feat_dims[4], self.feat_dims[4], act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            ELANBlock(self.feat_dims[4], self.feat_dims[5], self.squeeze_ratios[2], self.branch_depths[2], act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        ## P5/32: Stage-4
        self.layer_5 = nn.Sequential(
            DownSample(self.feat_dims[5], self.feat_dims[5], act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            ELANBlock(self.feat_dims[5], self.feat_dims[6], self.squeeze_ratios[3], self.branch_depths[3], act_type=act_type, norm_type=norm_type, depthwise=depthwise)
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
        # -------------------- Basic parameters --------------------
        self.feat_dims = [40, 80, 160, 320, 640, 1280, 1280]
        self.squeeze_ratios = [0.5, 0.5, 0.5, 0.25]  # Stage-1 -> Stage-4
        self.branch_depths = [3, 3, 3, 3]            # Stage-1 -> Stage-4

        # -------------------- Network parameters --------------------
        ## P1/2
        self.layer_1 = nn.Sequential(
            Conv(3, self.feat_dims[0], k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise),      
            Conv(self.feat_dims[0], self.feat_dims[1], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            Conv(self.feat_dims[1], self.feat_dims[1], k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        ## P2/4: Stage-1
        self.layer_2 = nn.Sequential(   
            Conv(self.feat_dims[1], self.feat_dims[2], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),             
            ELANBlock(self.feat_dims[2], self.feat_dims[3], self.squeeze_ratios[0], self.branch_depths[0], act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        ## P3/8: Stage-2
        self.layer_3 = nn.Sequential(
            DownSample(self.feat_dims[3], self.feat_dims[3], act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            ELANBlock(self.feat_dims[3], self.feat_dims[4], self.squeeze_ratios[1], self.branch_depths[1], act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        ## P4/16: Stage-3
        self.layer_4 = nn.Sequential(
            DownSample(self.feat_dims[4], self.feat_dims[4], act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            ELANBlock(self.feat_dims[4], self.feat_dims[5], self.squeeze_ratios[2], self.branch_depths[2], act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        ## P5/32: Stage-4
        self.layer_5 = nn.Sequential(
            DownSample(self.feat_dims[5], self.feat_dims[5], act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            ELANBlock(self.feat_dims[5], self.feat_dims[6], self.squeeze_ratios[3], self.branch_depths[3], act_type=act_type, norm_type=norm_type, depthwise=depthwise)
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
## build backbone
def build_backbone(cfg, pretrained=False): 
    # build backbone
    if cfg['backbone'] == 'elannet_huge':
        backbone = ELANNet_Huge(cfg['bk_act'], cfg['bk_norm'], cfg['bk_dpw'])
    elif cfg['backbone'] == 'elannet_large':
        backbone = ELANNet_Lagre(cfg['bk_act'], cfg['bk_norm'], cfg['bk_dpw'])
    elif cfg['backbone'] == 'elannet_tiny':
        backbone = ELANNet_Tiny(cfg['bk_act'], cfg['bk_norm'], cfg['bk_dpw'])
    # pyramid feat dims
    feat_dims = backbone.feat_dims[-3:]

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
                    print('Unused key: ', k)

            backbone.load_state_dict(checkpoint_state_dict)
        else:
            print('No backbone pretrained: ELANNet')        

    return backbone, feat_dims


if __name__ == '__main__':
    import time
    from thop import profile
    cfg = {
        'pretrained': False,
        'backbone': 'elannet_tiny',
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
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