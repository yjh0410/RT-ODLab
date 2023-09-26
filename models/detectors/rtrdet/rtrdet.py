import torch
import torch.nn as nn
import torch.nn.functional as F

from .rtrdet_backbone import build_backbone
from .rtrdet_encoder import build_encoder
from .rtrdet_decoder import build_decoder


# Real-time Detection with Transformer
class RTRDet(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes :int = 20, 
                 trainable   :bool = False, 
                 aux_loss    :bool = False,
                 deploy      :bool = False):
        super(RTRDet, self).__init__()
        # ------------------ Basic parameters ------------------
        self.cfg = cfg
        self.device = device
        self.max_stride = cfg['max_stride']
        self.num_topk = cfg['num_topk']
        self.d_model = round(cfg['d_model'] * cfg['width'])
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        self.trainable = trainable
        self.deploy = deploy
        
        # ------------------ Network parameters ------------------
        ## Backbone
        self.backbone, self.feat_dims = build_backbone(cfg, trainable&cfg['pretrained'])
        self.input_proj1 = nn.Conv2d(self.feat_dims[-1], self.d_model, kernel_size=1)
        self.input_proj2 = nn.Conv2d(self.feat_dims[-2], self.d_model, kernel_size=1)

        ## Transformer Encoder
        self.encoder = build_encoder(cfg)

        ## Transformer Decoder
        self.decoder = build_decoder(cfg, num_classes, return_intermediate=aux_loss)


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
        # -------------------- Inference --------------------
        ## Backbone
        pyramid_feats = self.backbone(x)
        high_level_feat = self.input_proj1(pyramid_feats[-1])
        bs, c, h, w = high_level_feat.size()

        ## Transformer Encoder
        pos_embed1 = self.position_embedding(high_level_feat)
        high_level_feat = self.encoder(high_level_feat, pos_embed1, self.decoder.adapt_pos2d)
        high_level_feat = high_level_feat.permute(0, 2, 1).reshape(bs, c, h, w)
        p4_level_feat = self.input_proj2(pyramid_feats[-2]) + F.interpolate(high_level_feat, scale_factor=2.0)

        ## Transformer Decoder
        pos_embed2 = self.position_embedding(p4_level_feat)
        output_classes, output_coords = self.decoder(p4_level_feat, pos_embed2)

        # -------------------- Post-process --------------------
        ## Top-k
        cls_pred, box_pred = output_classes[-1].flatten().sigmoid_(), output_coords[-1]
        cls_pred = cls_pred[0].flatten().sigmoid_()
        box_pred = box_pred[0]
        predicted_prob, topk_idxs = cls_pred.sort(descending=True)
        topk_idxs = topk_idxs[:self.num_topk]
        topk_box_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
        topk_scores = predicted_prob[:self.num_topk]
        topk_labels = topk_idxs % self.num_classes
        topk_bboxes = box_pred[topk_box_idxs]
        ## Denormalize bbox
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
            # -------------------- Inference --------------------
            ## Backbone
            pyramid_feats = self.backbone(x)
            high_level_feat = self.input_proj1(pyramid_feats[-1])
            bs, c, h, w = high_level_feat.size()

            ## Transformer Encoder
            pos_embed1 = self.position_embedding(high_level_feat)
            high_level_feat = self.encoder(high_level_feat, pos_embed1, self.decoder.adapt_pos2d)
            high_level_feat = high_level_feat.permute(0, 2, 1).reshape(bs, c, h, w)
            p4_level_feat = self.input_proj2(pyramid_feats[-2]) + F.interpolate(high_level_feat, scale_factor=2.0)

            ## Transformer Decoder
            pos_embed2 = self.position_embedding(p4_level_feat)
            output_classes, output_coords = self.decoder(p4_level_feat, pos_embed2)

            outputs = {'pred_logits': output_classes[-1], 'pred_boxes': output_coords[-1]}
            if self.aux_loss:
                outputs['aux_outputs'] = self.set_aux_loss(output_classes, output_coords)
            
            return outputs
    