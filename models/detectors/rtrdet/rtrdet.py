import torch
import torch.nn as nn

from .rtrdet_backbone import build_backbone
from .rtrdet_transformer import build_transformer


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
        assert cfg['out_stride'] == 16 or cfg['out_stride'] == 32
        # ------------------ Basic parameters ------------------
        self.cfg = cfg
        self.device = device
        self.out_stride = cfg['out_stride']
        self.max_stride = cfg['max_stride']
        self.num_levels = 2 if cfg['out_stride'] == 16 else 1
        self.num_topk = cfg['num_topk']
        self.num_classes = num_classes
        self.d_model = round(cfg['d_model'] * cfg['width'])
        self.aux_loss = aux_loss
        self.trainable = trainable
        self.deploy = deploy
        
        # ------------------ Network parameters ------------------
        ## Backbone
        self.backbone, self.feat_dims = build_backbone(cfg, trainable&cfg['pretrained'])
        self.input_projs = nn.ModuleList(nn.Conv2d(self.feat_dims[-i], self.d_model, kernel_size=1) for i in range(1, self.num_levels+1))
        
        ## Transformer
        self.transformer = build_transformer(cfg, num_classes, return_intermediate=aux_loss)


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
        
        ## Input proj
        for idx in range(1, self.num_levels + 1):
            pyramid_feats[-idx] = self.input_projs[idx-1](pyramid_feats[-idx])

        ## Transformer
        if self.num_levels == 2:
            src1, src2 = pyramid_feats[-2], pyramid_feats[-1]
        else:
            src1, src2 = None, pyramid_feats[-1]
        output_classes, output_coords = self.transformer(src1, src2)

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
            
            ## Input proj
            for idx in range(1, self.num_levels + 1):
                pyramid_feats[-idx] = self.input_projs[idx-1](pyramid_feats[-idx])

            ## Transformer
            if self.num_levels == 2:
                src1, src2 = pyramid_feats[-2], pyramid_feats[-1]
            else:
                src1, src2 = None, pyramid_feats[-1]
            output_classes, output_coords = self.transformer(src1, src2)

            outputs = {'pred_logits': output_classes[-1], 'pred_boxes': output_coords[-1]}
            if self.aux_loss:
                outputs['aux_outputs'] = self.set_aux_loss(output_classes, output_coords)
            
            return outputs
    