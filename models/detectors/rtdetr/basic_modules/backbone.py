import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import (ResNet18_Weights,
                                       ResNet34_Weights,
                                       ResNet50_Weights,
                                       ResNet101_Weights)
try:
    from .basic import FrozenBatchNorm2d
except:
    from basic  import FrozenBatchNorm2d
   

# IN1K pretrained weights
pretrained_urls = {
    # ResNet series
    'resnet18':  ResNet18_Weights,
    'resnet34':  ResNet34_Weights,
    'resnet50':  ResNet50_Weights,
    'resnet101': ResNet101_Weights,
    # ShuffleNet series
}


# ----------------- Model functions -----------------
## Build backbone network
def build_backbone(cfg, pretrained):
    print('==============================')
    print('Backbone: {}'.format(cfg['backbone']))
    # ResNet
    if 'resnet' in cfg['backbone']:
        pretrained_weight = cfg['pretrained_weight'] if pretrained else None
        model, feats = build_resnet(cfg, pretrained_weight)
    elif 'svnetv2' in cfg['backbone']:
        pretrained_weight = cfg['pretrained_weight'] if pretrained else None
        model, feats = build_scnetv2(cfg, pretrained_weight)
    else:
        raise NotImplementedError("Unknown backbone: <>.".format(cfg['backbone']))
    
    return model, feats


# ----------------- ResNet Backbone -----------------
class ResNet(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str, res5_dilation: bool, norm_type: str, pretrained_weights: str = "imagenet1k_v1"):
        super().__init__()
        # Pretrained
        assert pretrained_weights in [None, "imagenet1k_v1", "imagenet1k_v2"]
        if pretrained_weights is not None:
            if name in ('resnet18', 'resnet34'):
                pretrained_weights = pretrained_urls[name].IMAGENET1K_V1
            else:
                if pretrained_weights == "imagenet1k_v1":
                    pretrained_weights = pretrained_urls[name].IMAGENET1K_V1
                else:
                    pretrained_weights = pretrained_urls[name].IMAGENET1K_V2
        else:
            pretrained_weights = None
        print('ImageNet pretrained weight: ', pretrained_weights)
        # Norm layer
        if norm_type == 'BN':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'FrozeBN':
            norm_layer = FrozenBatchNorm2d
        # Backbone
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, res5_dilation],
            norm_layer=norm_layer, weights=pretrained_weights)
        return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.feat_dims = [128, 256, 512] if name in ('resnet18', 'resnet34') else [512, 1024, 2048]
        # Freeze
        for name, parameter in backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

    def forward(self, x):
        xs = self.body(x)
        fmp_list = []
        for name, fmp in xs.items():
            fmp_list.append(fmp)

        return fmp_list

def build_resnet(cfg, pretrained_weight=None):
    # ResNet series
    backbone = ResNet(cfg['backbone'], cfg['res5_dilation'], cfg['backbone_norm'], pretrained_weight)

    return backbone, backbone.feat_dims


# ----------------- ShuffleNet Backbone -----------------
## TODO: Add shufflenet-v2
class ShuffleNetv2:
    pass

def build_scnetv2(cfg, pretrained_weight=None):
    return


if __name__ == '__main__':
    cfg = {
        'backbone':      'resnet18',
        'backbone_norm': 'FrozeBN',
        'res5_dilation': False,
        'pretrained': True,
        'pretrained_weight': 'imagenet1k_v1',
    }
    model, feat_dim = build_backbone(cfg, cfg['pretrained'])
    print(feat_dim)

    x = torch.randn(2, 3, 320, 320)
    output = model(x)
    for y in output:
        print(y.size())
