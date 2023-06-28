import torch
import torch.nn as nn

from .rtdetr_basic import MLP


class DetectHead(nn.Module):
    def __init__(self, cfg, d_model, num_classes, with_box_refine=False):
        super().__init__()
        # --------- Basic Parameters ----------
        self.cfg = cfg
        self.num_classes = num_classes

        # --------- Network Parameters ----------
        self.class_embed = nn.ModuleList([nn.Linear(d_model, self.num_classes)])
        self.bbox_embed = nn.ModuleList([MLP(d_model, d_model, 4, 3)])
        if with_box_refine:
            self.class_embed = nn.ModuleList([
                self.class_embed[0] for _ in range(cfg['num_decoder_layers'])])
            self.bbox_embed = nn.ModuleList([
                self.bbox_embed[0] for _ in range(cfg['num_decoder_layers'])])

        self.init_weight()


    def init_weight(self):
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))

        # cls pred
        for class_embed in self.class_embed:
            class_embed.bias.data = torch.ones(self.num_classes) * bias_value

        # box pred
        for bbox_embed in self.bbox_embed:
            nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        

    def inverse_sigmoid(self, x):
        x = x.clamp(min=0, max=1)
        return torch.log(x.clamp(min=1e-5)/(1 - x).clamp(min=1e-5))


    def decode_bbox(self, outputs_coords):
        ## cxcywh -> xyxy
        x1y1_pred = outputs_coords[..., :2] - outputs_coords[..., 2:] * 0.5
        x2y2_pred = outputs_coords[..., :2] + outputs_coords[..., 2:] * 0.5
        box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)
        
        return box_pred


    def forward(self, hs, reference, multi_layer=False):
        if multi_layer:
            # class embed
            outputs_class = torch.stack([
                layer_cls_embed(layer_hs) for layer_cls_embed, layer_hs in zip(self.class_embed, hs)])
            # Bbox embed
            outputs_coords = []
            for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
                layer_delta_unsig = layer_bbox_embed(layer_hs)
                layer_ref_sig = self.inverse_sigmoid(layer_ref_sig)
                layer_outputs_unsig = layer_delta_unsig + layer_ref_sig
                layer_outputs_unsig = layer_outputs_unsig.sigmoid()
                outputs_coords.append(layer_outputs_unsig)
        else:
            # class embed
            outputs_class = self.class_embed[-1](hs[-1]) 
            # bbox embed
            delta_unsig = self.bbox_embed[-1](hs[-1])
            ref_sig = reference[-2]
            ref_sig = self.inverse_sigmoid(ref_sig)
            outputs_unsig = delta_unsig + ref_sig
            outputs_coords = outputs_unsig.sigmoid()
            # decode bbox
            outputs_coords = self.decode_bbox(outputs_coords)


        return outputs_class, outputs_coords


def build_dethead(cfg, d_model, num_classes, with_box_refine):
    return DetectHead(cfg, d_model, num_classes, with_box_refine)
