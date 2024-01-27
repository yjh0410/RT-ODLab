import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, List, Optional, Type, Union

try:
    from .basic import conv1x1, BasicBlock, Bottleneck
except:
    from basic import conv1x1, BasicBlock, Bottleneck
   

# IN1K pretrained weights
pretrained_urls = {
    # ResNet series
    'resnet18': None,
    'resnet34': None,
    'resnet50': None,
    'resnet101': None,
    'resnet152': None,
    # ShuffleNet series
}


# ----------------- Model functions -----------------
## Build backbone network
def build_backbone(cfg, pretrained):
    if 'resnet' in cfg['backbone']:
        # Build ResNet
        model, feats = build_resnet(cfg, pretrained)
    else:
        raise NotImplementedError("Unknown backbone: <>.".format(cfg['backbone']))
    
    return model, feats

## Load pretrained weight
def load_pretrained(model_name):
    return


# ----------------- ResNet Backbone -----------------
class ResNet(nn.Module):
    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 num_classes: int = 1000,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 ) -> None:
        super().__init__()
        # --------------- Basic parameters ----------------
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 64
        self.dilation = 1
        self.zero_init_residual = zero_init_residual
        self.replace_stride_with_dilation = [False, False, False] if replace_stride_with_dilation is None else replace_stride_with_dilation
        if len(self.replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {self.replace_stride_with_dilation}"
            )

        # --------------- Network parameters ----------------
        self._norm_layer = nn.BatchNorm2d if norm_layer is None else norm_layer
        ## Stem layer
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ## Res Layer
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=self.replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=self.replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=self.replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_layer()

    def _init_layer(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def _resnet(block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], **kwargs) -> ResNet:
    return ResNet(block, layers, **kwargs)

def build_resnet(cfg, pretrained=False, **kwargs):
    # ---------- Build ResNet ----------
    if   cfg['backbone'] == 'resnet18':
        model = _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)
        feats = [128, 256, 512]
    elif cfg['backbone'] == 'resnet34':
        model = _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)
        feats = [128, 256, 512]
    elif cfg['backbone'] == 'resnet50':
        model = _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)
        feats = [512, 1024, 2048]
    elif cfg['backbone'] == 'resnet101':
        model = _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)
        feats = [512, 1024, 2048]
    elif cfg['backbone'] == 'resnet152':
        model = _resnet(Bottleneck, [3, 8, 36, 3], **kwargs)
        feats = [512, 1024, 2048]

    # ---------- Load pretrained ----------
    if pretrained:
        # TODO: load IN1K pretrained
        pass

    return model, feats


# ----------------- ShuffleNet Backbone -----------------
## TODO: Add shufflenet-v2