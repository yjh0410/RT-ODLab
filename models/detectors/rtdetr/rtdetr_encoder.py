import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_modules.backbone import build_backbone
from .basic_modules.pafpn    import build_pafpn


# ----------------- Image Encoder -----------------
class ImageEncoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.backbone = None
        self.neck = None
        self.fpn = None

    def forward(self, x):
        return
    