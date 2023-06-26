import torch
import torch.nn as nn

from .rtdetr_encoder import build_encoder
from .rtdetr_decoder import build_decoder
from .rtdetr_dethead import build_dethead


# Real-time DETR
class RTDETR(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes = 20, 
                 trainable = False, 
                 aux_loss = False,
                 with_box_refine = False,
                 deploy = False):
        super(RTDETR, self).__init__()
        # --------- Basic Parameters ----------
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes
        self.trainable = trainable
        self.max_stride = max(cfg['stride'])
        self.d_model = round(cfg['d_model'] * self.cfg['width'])
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.deploy = deploy
        
        # --------- Network Parameters ----------
        ## Encoder
        self.img_encoder = build_encoder(cfg, trainable, 'img_encoder')

        ## Decoder
        self.decoder = build_decoder(cfg, self.d_model, return_intermediate=aux_loss)

        ## DetHead
        self.dethead = build_dethead(cfg, self.d_model, num_classes, with_box_refine)
            
        # set for TR-Decoder
        self.decoder.class_embed = self.dethead.class_embed
        self.decoder.bbox_embed = self.dethead.bbox_embed


    # ---------------------- Basic Functions ----------------------
    @torch.jit.unused
    def set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


    # ---------------------- Main Process for Inference ----------------------
    @torch.no_grad()
    def inference_single_image(self, x):
        # -------------------- Encoder --------------------
        memory, memory_pos = self.img_encoder(x)

        # -------------------- Decoder --------------------
        hs, reference = self.decoder(memory, memory_pos)

        # -------------------- DetHead --------------------
        out_logits, out_bbox = self.dethead(hs, reference, False)

        # -------------------- Decode bbox --------------------
        cls_pred = out_logits[0]
        box_pred = out_bbox[0]
        x1y1_pred = box_pred[..., :2] - box_pred[..., 2:] * 0.5
        x2y2_pred = box_pred[..., :2] + box_pred[..., 2:] * 0.5
        box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

        # -------------------- Top-k --------------------
        cls_pred = cls_pred.flatten().sigmoid_()
        num_topk = 100
        predicted_prob, topk_idxs = cls_pred.sort(descending=True)
        topk_idxs = topk_idxs[:num_topk]
        topk_box_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
        topk_scores = predicted_prob[:num_topk]
        topk_labels = topk_idxs % self.num_classes
        topk_bboxes = box_pred[topk_box_idxs]

        return topk_bboxes, topk_scores, topk_labels
        

    # ---------------------- Main Process for Training ----------------------
    def forward(self, x):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            # -------------------- Encoder --------------------
            memory, memory_pos = self.img_encoder(x)

            # -------------------- Decoder --------------------
            hs, reference = self.decoder(memory, memory_pos)

            # -------------------- DetHead --------------------
            outputs_class, outputs_coords = self.dethead(hs, reference, True)

            outputs = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coords[-1]}
            if self.aux_loss:
                outputs['aux_outputs'] = self.set_aux_loss(outputs_class, outputs_coords)
            
            return outputs
    