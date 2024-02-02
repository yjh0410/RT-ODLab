import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

try:
    from .basic import FrozenBatchNorm2d
except:
    from basic  import FrozenBatchNorm2d
   

# IN1K MIM pretrained weights (from SparK: https://github.com/keyu-tian/SparK)
pretrained_urls = {
    # ResNet series
    'resnet18':  None,
    'resnet34':  None,
    'resnet50':  "https://github.com/yjh0410/RT-ODLab/releases/download/backbone_weight/resnet50_in1k_spark_pretrained_timm_style.pth",
    'resnet101': None,
    # ShuffleNet series
}


# ----------------- Model functions -----------------
## Build backbone network
def build_backbone(cfg, pretrained=False):
    print('==============================')
    print('Backbone: {}'.format(cfg['backbone']))
    # ResNet
    if 'resnet' in cfg['backbone']:
        model, feats = build_resnet(cfg, pretrained)
    elif 'svnetv2' in cfg['backbone']:
        pretrained_weight = cfg['pretrained_weight'] if pretrained else None
        model, feats = build_scnetv2(cfg, pretrained_weight)
    else:
        raise NotImplementedError("Unknown backbone: <>.".format(cfg['backbone']))
    
    return model, feats


# ----------------- ResNet Backbone -----------------
class ResNet(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self,
                 name: str,
                 norm_type: str,
                 pretrained: bool = False,
                 freeze_at: int = -1,
                 freeze_stem_only: bool = False):
        super().__init__()
        # Pretrained
        # Norm layer
        if norm_type == 'BN':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'FrozeBN':
            norm_layer = FrozenBatchNorm2d
        # Backbone
        backbone = getattr(torchvision.models, name)(norm_layer=norm_layer,)
        return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.feat_dims = [128, 256, 512] if name in ('resnet18', 'resnet34') else [512, 1024, 2048]
        
        # Load pretrained
        if pretrained:
            self.load_pretrained(name)

        # Freeze
        if freeze_at >= 0:
            for name, parameter in backbone.named_parameters():
                if freeze_stem_only:
                    if 'layer1' not in name and 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                        parameter.requires_grad_(False)
                else:
                    if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                        parameter.requires_grad_(False)

    def load_pretrained(self, name):
        url = pretrained_urls[name]
        if url is not None:
            print('Loading pretrained weight from : {}'.format(url))
            # checkpoint state dict
            checkpoint_state_dict = torch.hub.load_state_dict_from_url(
                url=url, map_location="cpu", check_hash=True)
            # model state dict
            model_state_dict = self.body.state_dict()
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
            # load the weight
            self.body.load_state_dict(checkpoint_state_dict)
        else:
            print('No backbone pretrained for {}.'.format(name))

    def forward(self, x):
        xs = self.body(x)
        fmp_list = []
        for name, fmp in xs.items():
            fmp_list.append(fmp)

        return fmp_list

def build_resnet(cfg, pretrained=False):
    # ResNet series
    backbone = ResNet(cfg['backbone'],
                      cfg['backbone_norm'],
                      pretrained,
                      cfg['freeze_at'],
                      cfg['freeze_stem_only'])

    return backbone, backbone.feat_dims


# ----------------- ShuffleNet Backbone -----------------
## TODO: Add shufflenet-v2
class ShuffleNetv2:
    pass

def build_scnetv2(cfg, pretrained_weight=None):
    return


if __name__ == '__main__':
    cfg = {
        'backbone': 'resnet50',
        'backbone_norm': 'FrozeBN',
        'pretrained': True,
        'freeze_at': 0,
        'freeze_stem_only': False,
    }
    model, feat_dim = build_backbone(cfg, cfg['pretrained'])
    model.eval()
    print(feat_dim)

    x = torch.ones(2, 3, 320, 320)
    output = model(x)
    for y in output:
        print(y.size())
    print(output[-1])

