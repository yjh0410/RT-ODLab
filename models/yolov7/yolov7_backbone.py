import torch
import torch.nn as nn

try:
    from .yolov7_basic import Conv, ELANBlock, DownSample
except:
    from yolov7_basic import Conv, ELANBlock, DownSample
    

model_urls = {
    "elannet": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/yolov7_elannet_large.pth",
}

# --------------------- CSPDarkNet-53 -----------------------
# ELANNet
class ELANNet(nn.Module):
    """
    ELAN-Net of YOLOv7-L.
    """
    def __init__(self, act_type='silu', norm_type='BN', depthwise=False):
        super(ELANNet, self).__init__()
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
            ELANBlock(in_dim=128, out_dim=256, expand_ratio=0.5,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P3/8
        self.layer_3 = nn.Sequential(
            DownSample(in_dim=256, act_type=act_type),             
            ELANBlock(in_dim=256, out_dim=512, expand_ratio=0.5,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P4/16
        self.layer_4 = nn.Sequential(
            DownSample(in_dim=512, act_type=act_type),             
            ELANBlock(in_dim=512, out_dim=1024, expand_ratio=0.5,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P5/32
        self.layer_5 = nn.Sequential(
            DownSample(in_dim=1024, act_type=act_type),             
            ELANBlock(in_dim=1024, out_dim=1024, expand_ratio=0.25,
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
    backbone = ELANNet(cfg['bk_act'], cfg['bk_norm'], cfg['bk_dpw'])
    feat_dims = backbone.feat_dims

    if pretrained:
        url = model_urls['elannet']
        if url is not None:
            print('Loading pretrained weight ...')
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

    x = torch.randn(1, 3, 224, 224)
    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))