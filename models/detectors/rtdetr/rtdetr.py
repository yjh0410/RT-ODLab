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
        self.encoder = build_encoder(cfg, trainable, 'img_encoder')

        ## Decoder
        self.decoder = build_decoder(cfg, self.d_model, return_intermediate=aux_loss)

        ## DetHead
        self.dethead = build_dethead(cfg, self.d_model, num_classes, with_box_refine)
            
        # set for TR-Decoder
        self.decoder.class_embed = self.dethead.class_embed
        self.decoder.bbox_embed = self.dethead.bbox_embed


    # ---------------------- Basic Functions ----------------------
    def position_embedding(self, x, temperature=10000):
        hs, ws = x.shape[-2:]
        device = x.device
        num_pos_feats = x.shape[1] // 2       
        scale = 2 * 3.141592653589793

        # generate xy coord mat
        y_embed, x_embed = torch.meshgrid(
            [torch.arange(1, hs+1, dtype=torch.float32),
             torch.arange(1, ws+1, dtype=torch.float32)])
        y_embed = y_embed / (hs + 1e-6) * scale
        x_embed = x_embed / (ws + 1e-6) * scale
    
        # [H, W] -> [1, H, W]
        y_embed = y_embed[None, :, :].to(device)
        x_embed = x_embed[None, :, :].to(device)

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=device)
        dim_t_ = torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats
        dim_t = temperature ** (2 * dim_t_)

        pos_x = torch.div(x_embed[:, :, :, None], dim_t)
        pos_y = torch.div(y_embed[:, :, :, None], dim_t)
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        # [B, C, H, W]
        pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        return pos_embed
        

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
        pyramid_feats = self.encoder(x)

        # -------------------- Pos Embed --------------------
        memory = torch.cat([feat.flatten(2) for feat in pyramid_feats], dim=-1)
        memory_pos = torch.cat([self.position_embedding(feat).flatten(2) for feat in pyramid_feats], dim=-1)
        memory = memory.permute(0, 2, 1).contiguous()
        memory_pos = memory_pos.permute(0, 2, 1).contiguous()

        # -------------------- Decoder --------------------
        hs, reference = self.decoder(memory, memory_pos)

        # -------------------- DetHead --------------------
        out_logits, out_bbox = self.dethead(hs, reference, False)
        cls_pred, box_pred = out_logits[0], out_bbox[0]

        # -------------------- Top-k --------------------
        cls_pred = cls_pred.flatten().sigmoid_()
        num_topk = 100
        predicted_prob, topk_idxs = cls_pred.sort(descending=True)
        topk_idxs = topk_idxs[:num_topk]
        topk_box_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
        topk_scores = predicted_prob[:num_topk]
        topk_labels = topk_idxs % self.num_classes
        topk_bboxes = box_pred[topk_box_idxs]

        # denormalize bbox
        img_h, img_w = x.shape[-2:]
        topk_bboxes[..., 0::2] *= img_w
        topk_bboxes[..., 1::2] *= img_h

        if self.deploy:
            return topk_bboxes, topk_scores, topk_labels
        else:
            return topk_bboxes.cpu().numpy(), topk_scores.cpu().numpy(), topk_labels.cpu().numpy()
        

    # ---------------------- Main Process for Training ----------------------
    def forward(self, x):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            # -------------------- Encoder --------------------
            pyramid_feats = self.encoder(x)

            # -------------------- Pos Embed --------------------
            memory = torch.cat([feat.flatten(2) for feat in pyramid_feats], dim=-1)
            memory_pos = torch.cat([self.position_embedding(feat).flatten(2) for feat in pyramid_feats], dim=-1)
            memory = memory.permute(0, 2, 1).contiguous()
            memory_pos = memory_pos.permute(0, 2, 1).contiguous()
            
            # -------------------- Decoder --------------------
            hs, reference = self.decoder(memory, memory_pos)

            # -------------------- DetHead --------------------
            outputs_class, outputs_coords = self.dethead(hs, reference, True)

            outputs = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coords[-1]}
            if self.aux_loss:
                outputs['aux_outputs'] = self.set_aux_loss(outputs_class, outputs_coords)
            
            return outputs
    