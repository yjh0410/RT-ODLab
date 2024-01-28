import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------- Dencoder for Detection task -----------------
## RTDETR's Transformer
class DetDecoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.backbone = None
        self.neck = None
        self.fpn = None

    def forward(self, x):
        return


# ----------------- Dencoder for Segmentation task -----------------
class SegDecoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        # TODO: design seg-decoder

    def forward(self, x):
        return


# ----------------- Dencoder for Pose estimation task -----------------
class PosDecoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        # TODO: design seg-decoder

    def forward(self, x):
        return
