import torch
import torch.nn as nn

try:
    from .rtpdetr_encoder import build_image_encoder
    from .rtpdetr_decoder import build_transformer
except:
    from  rtpdetr_encoder import build_image_encoder
    from  rtpdetr_decoder import build_transformer


# Real-time Plain Transformer-based Object Detector
class RT_PDETR(nn.Module):
    def __init__(self,
                 cfg,
                 num_classes = 80,
                 conf_thresh = 0.1,
                 topk        = 300,
                 deploy      = False,
                 no_multi_labels = False,
                 ):
        super().__init__()
        # ----------- Basic setting -----------
        self.num_classes = num_classes
        self.num_topk = topk
        self.conf_thresh = conf_thresh
        self.no_multi_labels = no_multi_labels
        self.deploy = deploy

        # ----------- Network setting -----------
        ## Image encoder
        self.image_encoder = build_image_encoder(cfg)
        self.feat_dim = self.image_encoder.fpn_dims[-1]

        ## Detect decoder
        self.detect_decoder = build_transformer(cfg, self.feat_dim, num_classes, return_intermediate=self.training)

    def post_process(self, box_pred, cls_pred):
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

            return topk_bboxes, topk_scores, topk_labels
        else:
            # Top-k select
            cls_pred = cls_pred[0].flatten().sigmoid_()
            box_pred = box_pred[0]

            # Keep top k top scoring indices only.
            num_topk = min(self.num_topk, box_pred.size(0))

            # Topk candidates
            predicted_prob, topk_idxs = cls_pred.sort(descending=True)
            topk_scores = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:self.num_topk]

            # Filter out the proposals with low confidence score
            keep_idxs = topk_scores > self.conf_thresh
            scores = topk_scores[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]
            topk_box_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')

            ## Top-k results
            topk_scores = predicted_prob[:self.num_topk]
            topk_labels = topk_idxs % self.num_classes
            topk_bboxes = box_pred[topk_box_idxs]

        return topk_bboxes, topk_scores, topk_labels
    
    def forward(self, x, targets=None):
        # ----------- Image Encoder -----------
        pyramid_feats = self.image_encoder(x)

        # ----------- Transformer -----------
        transformer_outputs = self.detect_decoder(pyramid_feats, targets)

        if self.training:
            return transformer_outputs
        else:
            pred_boxes, pred_logits = transformer_outputs[0], transformer_outputs[1]
            box_preds = pred_boxes[-1]
            cls_preds = pred_logits[-1]
            
            # post-process
            bboxes, scores, labels = self.post_process(box_preds, cls_preds)

            outputs = {
                "scores": scores.cpu().numpy(),
                "labels": labels.cpu().numpy(),
                "bboxes": bboxes.cpu().numpy(),
            }

            return outputs
        

if __name__ == '__main__':
    import time
    from thop import profile
    from loss import build_criterion

    # Model config
    cfg = {
        'width': 1.0,
        'depth': 1.0,
        'out_stride': [8, 16, 32],
        # Image Encoder - Backbone
        'backbone': 'resnet18',
        'backbone_norm': 'BN',
        'res5_dilation': False,
        'pretrained': True,
        'pretrained_weight': 'imagenet1k_v1',
        # Image Encoder - FPN
        'fpn': 'hybrid_encoder',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        'hidden_dim': 256,
        'en_num_heads': 8,
        'en_num_layers': 1,
        'en_mlp_ratio': 4.0,
        'en_dropout': 0.1,
        'pe_temperature': 10000.,
        'en_act': 'gelu',
        # Transformer Decoder
        'transformer': 'rtdetr_transformer',
        'hidden_dim': 256,
        'de_num_heads': 8,
        'de_num_layers': 6,
        'de_mlp_ratio': 4.0,
        'de_dropout': 0.0,
        'de_act': 'gelu',
        'de_num_points': 4,
        'num_queries': 300,
        'learnt_init_query': False,
        'pe_temperature': 10000.,
        'dn_num_denoising': 100,
        'dn_label_noise_ratio': 0.5,
        'dn_box_noise_scale': 1,
        # Head
        'det_head': 'dino_head',
        # Matcher
        'matcher_hpy': {'cost_class': 2.0,
                        'cost_bbox': 5.0,
                        'cost_giou': 2.0,},
        # Loss
        'use_vfl': True,
        'loss_coeff': {'class': 1,
                       'bbox': 5,
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
    model = RT_DETR(cfg, num_classes=80)
    model.train()

    # Create criterion
    criterion = build_criterion(cfg, num_classes=80)

    # Model inference
    t0 = time.time()
    outputs = model(image, targets)
    t1 = time.time()
    print('Infer time: ', t1 - t0)

    # Compute loss
    loss = criterion(*outputs, targets)
    for k in loss.keys():
        print("{} : {}".format(k, loss[k].item()))

    print('==============================')
    model.eval()
    flops, params = profile(model, inputs=(image, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
