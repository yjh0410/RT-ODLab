# --------------- Torch components ---------------
import torch
import torch.nn as nn

# --------------- Model components ---------------
try:
    from .vitdet_encoder import build_image_encoder
    from .vitdet_decoder import build_decoder
    from .vitdet_head    import build_predictor
    from .basic_modules.basic import multiclass_nms
except:
    from  vitdet_encoder import build_image_encoder
    from  vitdet_decoder import build_decoder
    from  vitdet_head    import build_predictor
    from  basic_modules.basic import multiclass_nms



# Real-time ViT-based Object Detector
class ViTDet(nn.Module):
    def __init__(self,
                 cfg,
                 device,
                 num_classes = 20,
                 conf_thresh = 0.01,
                 nms_thresh  = 0.5,
                 topk        = 1000,
                 trainable   = False,
                 deploy      = False,
                 no_multi_labels    = False,
                 nms_class_agnostic = False,
                 ):
        super(ViTDet, self).__init__()
        # ---------------------- Basic Parameters ----------------------
        self.cfg = cfg
        self.device = device
        self.strides = cfg['stride']
        self.num_classes = num_classes
        ## Scale hidden channels by width_factor
        cfg['hidden_dim'] = round(cfg['hidden_dim'] * cfg['width'])
        cfg['pretrained'] = cfg['pretrained'] & trainable
        ## Post-process parameters
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk
        self.deploy = deploy
        self.no_multi_labels = no_multi_labels
        self.nms_class_agnostic = nms_class_agnostic
        
        # ---------------------- Network Parameters ----------------------
        ## ----------- Encoder -----------
        self.encoder = build_image_encoder(cfg)

        ## ----------- Decoder -----------
        self.decoder = build_decoder(cfg, self.encoder.fpn_dims, num_levels=3)
        
        ## ----------- Preds -----------
        self.predictor = build_predictor(cfg, self.strides, num_classes, 4, 3)

    def post_process(self, cls_preds, box_preds):
        """
        Input:
            cls_preds: List[np.array] -> [[M, C], ...]
            box_preds: List[np.array] -> [[M, 4], ...]
        Output:
            bboxes: np.array -> [N, 4]
            scores: np.array -> [N,]
            labels: np.array -> [N,]
        """
        all_scores = []
        all_labels = []
        all_bboxes = []
        
        for cls_pred_i, box_pred_i in zip(cls_preds, box_preds):
            cls_pred_i = cls_pred_i[0]
            box_pred_i = box_pred_i[0]
            if self.no_multi_labels:
                # [M,]
                scores, labels = torch.max(cls_pred_i.sigmoid(), dim=1)

                # Keep top k top scoring indices only.
                num_topk = min(self.topk_candidates, box_pred_i.size(0))

                # topk candidates
                predicted_prob, topk_idxs = scores.sort(descending=True)
                topk_scores = predicted_prob[:num_topk]
                topk_idxs = topk_idxs[:num_topk]

                # filter out the proposals with low confidence score
                keep_idxs = topk_scores > self.conf_thresh
                scores = topk_scores[keep_idxs]
                topk_idxs = topk_idxs[keep_idxs]

                labels = labels[topk_idxs]
                bboxes = box_pred_i[topk_idxs]
            else:
                # [M, C] -> [MC,]
                scores_i = cls_pred_i.sigmoid().flatten()

                # Keep top k top scoring indices only.
                num_topk = min(self.topk_candidates, box_pred_i.size(0))

                # torch.sort is actually faster than .topk (at least on GPUs)
                predicted_prob, topk_idxs = scores_i.sort(descending=True)
                topk_scores = predicted_prob[:num_topk]
                topk_idxs = topk_idxs[:num_topk]

                # filter out the proposals with low confidence score
                keep_idxs = topk_scores > self.conf_thresh
                scores = topk_scores[keep_idxs]
                topk_idxs = topk_idxs[keep_idxs]

                anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
                labels = topk_idxs % self.num_classes

                bboxes = box_pred_i[anchor_idxs]

            all_scores.append(scores)
            all_labels.append(labels)
            all_bboxes.append(bboxes)

        scores = torch.cat(all_scores, dim=0)
        labels = torch.cat(all_labels, dim=0)
        bboxes = torch.cat(all_bboxes, dim=0)

        if not self.deploy:
            # to cpu & numpy
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            bboxes = bboxes.cpu().numpy()

            # nms
            scores, labels, bboxes = multiclass_nms(
                scores, labels, bboxes, self.nms_thresh, self.num_classes, self.nms_class_agnostic)

        return bboxes, scores, labels
    
    def forward(self, x):
        # ---------------- Backbone ----------------
        pyramid_feats = self.encoder(x)

        # ---------------- Heads ----------------
        outputs = self.decoder(pyramid_feats)

        # ---------------- Preds ----------------
        outputs = self.predictor(outputs['cls_feats'], outputs['reg_feats'])

        if not self.training:
            cls_pred = outputs["pred_cls"]
            box_pred = outputs["pred_box"]
            # post process
            bboxes, scores, labels = self.post_process(cls_pred, box_pred)

            outputs = {
                "scores": scores,
                "labels": labels,
                "bboxes": bboxes
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
        'freeze_at': 0,
        'freeze_stem_only': False,
        'out_stride': [8, 16, 32],
        'max_stride': 32,
        # Convolutional Decoder
        'hidden_dim': 256,
        'decoder': 'det_decoder',
        'de_num_cls_layers': 2,
        'de_num_reg_layers': 2,
        'de_act': 'silu',
        'de_norm': 'BN',
        # Matcher
        'matcher_hpy': {'soft_center_radius': 2.5,
                        'topk_candidates': 13,},
        # Loss
        'use_vfl': True,
        'loss_coeff': {'class': 1,
                       'bbox': 1,
                       'giou': 2,},
        }
    bs = 1
    # Create a batch of images & targets
    image = torch.randn(bs, 3, 640, 640).cuda()
    targets = [{
        'labels': torch.tensor([2, 4, 5, 8]).long().cuda(),
        'boxes':  torch.tensor([[0, 0, 10, 10], [12, 23, 56, 70], [0, 10, 20, 30], [50, 60, 55, 150]]).float().cuda() / 640.
    }] * bs

    # Create model
    model = ViTDet(cfg, num_classes=20)
    model.train().cuda()

    # Create criterion
    criterion = build_criterion(cfg, num_classes=20)

    # Model inference
    t0 = time.time()
    outputs = model(image, targets)
    t1 = time.time()
    print('Infer time: ', t1 - t0)

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
