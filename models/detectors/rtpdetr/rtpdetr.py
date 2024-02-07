import math
import torch
import torch.nn as nn

try:
    from .basic_modules.basic import MLP, multiclass_nms
    from .basic_modules.transformer import get_clones
    from .rtpdetr_encoder import build_image_encoder
    from .rtpdetr_decoder import build_transformer
except:
    from  basic_modules.basic import MLP, multiclass_nms
    from  basic_modules.transformer import get_clones
    from  rtpdetr_encoder import build_image_encoder
    from  rtpdetr_decoder import build_transformer


# Real-time PlainDETR
class RT_PDETR(nn.Module):
    def __init__(self,
                 cfg,
                 num_classes = 80,
                 conf_thresh = 0.1,
                 nms_thresh  = 0.5,
                 topk        = 300,
                 deploy      = False,
                 no_multi_labels = False,
                 use_nms     = False,
                 nms_class_agnostic = False,
                 aux_loss    = False,
                 ):
        super().__init__()
        # ----------- Basic setting -----------
        self.num_queries_one2one = cfg['num_queries_one2one']
        self.num_queries_one2many = cfg['num_queries_one2many']
        self.num_queries = self.num_queries_one2one + self.num_queries_one2many
        self.num_classes = num_classes
        self.num_topk = topk
        self.aux_loss = aux_loss
        self.deploy = deploy
        # scale hidden channels by width_factor
        cfg['hidden_dim'] = round(cfg['hidden_dim'] * cfg['width'])
        ## Post-process parameters
        self.use_nms = use_nms
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.no_multi_labels = no_multi_labels
        self.nms_class_agnostic = nms_class_agnostic

        # ----------- Network setting -----------
        ## Image encoder
        self.image_encoder = build_image_encoder(cfg)

        ## Transformer Decoder
        self.transformer = build_transformer(cfg, return_intermediate=self.training)
        self.query_embed = nn.Embedding(self.num_queries, cfg['hidden_dim'])

        ## Detect Head
        class_embed = nn.Linear(cfg['hidden_dim'], num_classes)
        bbox_embed = MLP(cfg['hidden_dim'], cfg['hidden_dim'], 4, 3)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)

        self.class_embed = get_clones(class_embed, cfg['de_num_layers'] + 1)
        self.bbox_embed  = get_clones(bbox_embed, cfg['de_num_layers'] + 1)
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)

        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

    def pos2posembed(self, d_model, pos, temperature=10000):
        scale = 2 * torch.pi
        num_pos_feats = d_model // 2

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t_ = torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats
        dim_t = temperature ** (2 * dim_t_)

        # Position embedding for XY
        x_embed = pos[..., 0] * scale
        y_embed = pos[..., 1] * scale
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        posemb = torch.cat((pos_y, pos_x), dim=-1)
        
        # Position embedding for WH
        if pos.size(-1) == 4:
            w_embed = pos[..., 2] * scale
            h_embed = pos[..., 3] * scale
            pos_w = w_embed[..., None] / dim_t
            pos_h = h_embed[..., None] / dim_t
            pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()), dim=-1).flatten(-2)
            pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()), dim=-1).flatten(-2)
            posemb = torch.cat((posemb, pos_w, pos_h), dim=-1)
        
        return posemb

    def get_posembed(self, d_model, mask, temperature=10000, normalize=False):
        not_mask = ~mask
        # [B, H, W]
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if normalize:
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + 1e-6)
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + 1e-6)
        else:
            y_embed = y_embed - 0.5
            x_embed = x_embed - 0.5
    
        # [H, W] -> [B, H, W, 2]
        pos = torch.stack([x_embed, y_embed], dim=-1)

        # [B, H, W, C]
        pos_embed = self.pos2posembed(d_model, pos, temperature)
        pos_embed = pos_embed.permute(0, 3, 1, 2)
        
        return pos_embed

    def post_process(self, box_pred, cls_pred):
        # xywh -> xyxy
        box_preds_x1y1 = box_pred[..., :2] - 0.5 * box_pred[..., 2:]
        box_preds_x2y2 = box_pred[..., :2] + 0.5 * box_pred[..., 2:]
        box_pred = torch.cat([box_preds_x1y1, box_preds_x2y2], dim=-1)

        cls_pred = cls_pred[0]
        box_pred = box_pred[0]
        if self.no_multi_labels:
            # [M,]
            scores, labels = torch.max(cls_pred.sigmoid(), dim=1)

            # Keep top k top scoring indices only.
            num_topk = min(self.num_topk, box_pred.size(0))

            # Topk candidates
            predicted_prob, topk_idxs = scores.sort(descending=True)
            topk_scores = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # Filter out the proposals with low confidence score
            keep_idxs = topk_scores > self.conf_thresh
            topk_idxs = topk_idxs[keep_idxs]

            # Top-k results
            topk_scores = topk_scores[keep_idxs]
            topk_labels = labels[topk_idxs]
            topk_bboxes = box_pred[topk_idxs]

        else:
            # Top-k select
            cls_pred = cls_pred.flatten().sigmoid_()
            box_pred = box_pred

            # Keep top k top scoring indices only.
            num_topk = min(self.num_topk, box_pred.size(0))

            # Topk candidates
            predicted_prob, topk_idxs = cls_pred.sort(descending=True)
            topk_scores = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:self.num_topk]

            # Filter out the proposals with low confidence score
            keep_idxs = topk_scores > self.conf_thresh
            topk_scores = topk_scores[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]
            topk_box_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')

            ## Top-k results
            topk_labels = topk_idxs % self.num_classes
            topk_bboxes = box_pred[topk_box_idxs]

        topk_scores = topk_scores.cpu().numpy()
        topk_labels = topk_labels.cpu().numpy()
        topk_bboxes = topk_bboxes.cpu().numpy()

        # nms
        if self.use_nms:
            topk_scores, topk_labels, topk_bboxes = multiclass_nms(
                topk_scores, topk_labels, topk_bboxes, self.nms_thresh, self.num_classes, self.nms_class_agnostic)

        return topk_bboxes, topk_scores, topk_labels
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_coord_old, outputs_deltas):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b, "pred_boxes_old": c, "pred_deltas": d, }
            for a, b, c, d in zip(outputs_class[:-1], outputs_coord[:-1], outputs_coord_old[:-1], outputs_deltas[:-1])
        ]

    def inference_single_image(self, x):
        # ----------- Image Encoder -----------
        src = self.image_encoder(x)

        # ----------- Prepare inputs for Transformer -----------
        mask = torch.zeros([src.shape[0], src.shape[2], src.shape[3]]).bool().to(src.device)
        pos_embed = self.get_posembed(src.shape[1], mask, normalize=False)
        self_attn_mask = None
        query_embeds = self.query_embed.weight[:self.num_queries_one2one]

        # -----------Transformer -----------
        (
            hs,
            init_reference,
            inter_references,
            _,
            _,
            _,
            _,
            max_shape
        ) = self.transformer(src, mask, pos_embed, query_embeds, self_attn_mask)

        # ----------- Process outputs -----------
        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_deltas_one2one = []

        for lid in range(hs.shape[0]):
            if lid == 0:
                reference = init_reference
            else:
                reference = inter_references[lid - 1]
            outputs_class = self.class_embed[lid](hs[lid])
            tmp = self.bbox_embed[lid](hs[lid])
            outputs_coord = self.transformer.decoder.delta2bbox(reference, tmp, max_shape)  # xyxy

            outputs_classes_one2one.append(outputs_class[:, :self.num_queries_one2one])
            outputs_coords_one2one.append(outputs_coord[:, :self.num_queries_one2one])
            outputs_deltas_one2one.append(tmp[:, :self.num_queries_one2one])

        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)

        # ------------ Post process ------------
        cls_pred = outputs_classes_one2one[-1]
        box_pred = outputs_coords_one2one[-1]
        
        # post-process
        bboxes, scores, labels = self.post_process(box_pred, cls_pred)

        outputs = {
            "scores": scores,
            "labels": labels,
            "bboxes": bboxes,
        }

        return outputs
        
    def forward(self, x):
        if not self.training:
            return self.inference_single_image(x)

        # ----------- Image Encoder -----------
        src = self.image_encoder(x)

        # ----------- Prepare inputs for Transformer -----------
        mask = torch.zeros([src.shape[0], src.shape[2], src.shape[3]]).bool().to(src.device)
        pos_embed = self.get_posembed(src.shape[1], mask, normalize=False)
        self_attn_mask = torch.zeros(
            [self.num_queries, self.num_queries, ]).bool().to(src.device)
        self_attn_mask[self.num_queries_one2one:, 0: self.num_queries_one2one, ] = True
        self_attn_mask[0: self.num_queries_one2one, self.num_queries_one2one:, ] = True
        query_embeds = self.query_embed.weight

        # -----------Transformer -----------
        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
            enc_outputs_delta,
            output_proposals,
            max_shape
        ) = self.transformer(src, mask, pos_embed, query_embeds, self_attn_mask)

        # ----------- Process outputs -----------
        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_classes_one2many = []
        outputs_coords_one2many = []

        outputs_coords_old_one2one = []
        outputs_deltas_one2one = []
        outputs_coords_old_one2many = []
        outputs_deltas_one2many = []

        for lid in range(hs.shape[0]):
            if lid == 0:
                reference = init_reference
            else:
                reference = inter_references[lid - 1]
            outputs_class = self.class_embed[lid](hs[lid])
            tmp = self.bbox_embed[lid](hs[lid])
            outputs_coord = self.transformer.decoder.box_xyxy_to_cxcywh(
                self.transformer.decoder.delta2bbox(reference, tmp, max_shape))

            outputs_classes_one2one.append(outputs_class[:, 0: self.num_queries_one2one])
            outputs_classes_one2many.append(outputs_class[:, self.num_queries_one2one:])

            outputs_coords_one2one.append(outputs_coord[:, 0: self.num_queries_one2one])
            outputs_coords_one2many.append(outputs_coord[:, self.num_queries_one2one:])

            outputs_coords_old_one2one.append(reference[:, :self.num_queries_one2one])
            outputs_coords_old_one2many.append(reference[:, self.num_queries_one2one:])
            outputs_deltas_one2one.append(tmp[:, :self.num_queries_one2one])
            outputs_deltas_one2many.append(tmp[:, self.num_queries_one2one:])

        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)

        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)

        out = {
            "pred_logits": outputs_classes_one2one[-1],
            "pred_boxes": outputs_coords_one2one[-1],
            "pred_logits_one2many": outputs_classes_one2many[-1],
            "pred_boxes_one2many": outputs_coords_one2many[-1],

            "pred_boxes_old": outputs_coords_old_one2one[-1],
            "pred_deltas": outputs_deltas_one2one[-1],
            "pred_boxes_old_one2many": outputs_coords_old_one2many[-1],
            "pred_deltas_one2many": outputs_deltas_one2many[-1],
        }

        out["aux_outputs"] = self._set_aux_loss(
            outputs_classes_one2one, outputs_coords_one2one, outputs_coords_old_one2one, outputs_deltas_one2one
        )
        out["aux_outputs_one2many"] = self._set_aux_loss(
            outputs_classes_one2many, outputs_coords_one2many, outputs_coords_old_one2many, outputs_deltas_one2many
        )

        out["enc_outputs"] = {
            "pred_logits": enc_outputs_class,
            "pred_boxes": enc_outputs_coord_unact,
            "pred_boxes_old": output_proposals,
            "pred_deltas": enc_outputs_delta,
        }

        return out
                

if __name__ == '__main__':
    import time
    from thop import profile
    from loss import build_criterion

    # Model config
    cfg = {
        'width': 1.0,
        'depth': 1.0,
        'max_stride': 32,
        'out_stride': 16,
        # Image Encoder - Backbone
        'backbone': 'resnet50',
        'backbone_norm': 'FrozeBN',
        'pretrained': True,
        'freeze_at': 0,
        'freeze_stem_only': False,
        'hidden_dim': 256,
        'en_num_heads': 8,
        'en_num_layers': 6,
        'en_mlp_ratio': 4.0,
        'en_dropout': 0.0,
        'en_act': 'gelu',
        # Transformer Decoder
        'transformer': 'plain_detr_transformer',
        'hidden_dim': 256,
        'de_num_heads': 8,
        'de_num_layers': 6,
        'de_mlp_ratio': 4.0,
        'de_dropout': 0.0,
        'de_act': 'gelu',
        'de_pre_norm': True,
        'rpe_hidden_dim': 512,
        'use_checkpoint': False,
        'proposal_feature_levels': 3,
        'proposal_tgt_strides': [8, 16, 32],
        'num_queries_one2one': 300,
        'num_queries_one2many': 300,
        # Matcher
        'matcher_hpy': {'cost_class': 2.0,
                        'cost_bbox': 1.0,
                        'cost_giou': 2.0,},
        # Loss
        'use_vfl': True,
        'k_one2many': 6,
        'lambda_one2many': 1.0,
        'loss_coeff': {'class': 2,
                       'bbox': 1,
                       'giou': 2,
                       'no_object': 0.1,},
        }
    bs = 1
    # Create a batch of images & targets
    image = torch.randn(bs, 3, 640, 640)
    targets = [{
        'labels': torch.tensor([2, 4, 5, 8]).long(),
        'boxes':  torch.tensor([[0, 0, 10, 10], [12, 23, 56, 70], [0, 10, 20, 30], [50, 60, 55, 150]]).float() / 640.
    }] * bs

    # Create model
    model = RT_PDETR(cfg, num_classes=80)
    model.train()

    # Model inference
    t0 = time.time()
    outputs = model(image)
    t1 = time.time()
    print('Infer time: ', t1 - t0)

    # Create criterion
    criterion = build_criterion(cfg, num_classes=80, aux_loss=True)

    # Compute loss
    loss = criterion(outputs, targets)
    for k in loss.keys():
        print("{} : {}".format(k, loss[k].item()))

    print('==============================')
    model.eval()
    flops, params = profile(model, inputs=(image, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
