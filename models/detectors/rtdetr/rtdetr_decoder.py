import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_, uniform_, normal_
from typing import List

try:
    from .basic_modules.basic import BasicConv, MLP
    from .basic_modules.transformer import DeformableTransformerDecoder
    from .basic_modules.dn_compoments import get_contrastive_denoising_training_group
except:
    from  basic_modules.basic import BasicConv, MLP
    from  basic_modules.transformer import DeformableTransformerDecoder
    from  basic_modules.dn_compoments import get_contrastive_denoising_training_group


def build_transformer(cfg, in_dims, num_classes, return_intermediate=False):
    if cfg['transformer'] == 'rtdetr_transformer':
        return RTDETRTransformer(in_dims             = in_dims,
                                 hidden_dim          = cfg['hidden_dim'],
                                 strides             = cfg['out_stride'],
                                 num_classes         = num_classes,
                                 num_queries         = cfg['num_queries'],
                                 pos_embed_type      = 'sine',
                                 num_heads           = cfg['de_num_heads'],
                                 num_layers          = cfg['de_num_layers'],
                                 num_levels          = len(cfg['out_stride']),
                                 num_points          = cfg['de_num_points'],
                                 ffn_dim           = cfg['de_ffn_dim'],
                                 dropout             = cfg['de_dropout'],
                                 act_type            = cfg['de_act'],
                                 return_intermediate = return_intermediate,
                                 num_denoising       = cfg['dn_num_denoising'],
                                 label_noise_ratio   = cfg['dn_label_noise_ratio'],
                                 box_noise_scale     = cfg['dn_box_noise_scale'],
                                 learnt_init_query   = cfg['learnt_init_query'],
                                 )


# ----------------- Dencoder for Detection task -----------------
## RTDETR's Transformer for Detection task
class RTDETRTransformer(nn.Module):
    def __init__(self,
                 # basic parameters
                 in_dims        :List = [256, 512, 1024],
                 hidden_dim     :int  = 256,
                 strides        :List = [8, 16, 32],
                 num_classes    :int  = 80,
                 num_queries    :int  = 300,
                 pos_embed_type :str  = 'sine',
                 # transformer parameters
                 num_heads      :int   = 8,
                 num_layers     :int   = 1,
                 num_levels     :int   = 3,
                 num_points     :int   = 4,
                 ffn_dim        :int   = 1024,
                 dropout        :float = 0.1,
                 act_type       :str   = "relu",
                 return_intermediate :bool = False,
                 # Denoising parameters
                 num_denoising       :int  = 100,
                 label_noise_ratio   :float = 0.5,
                 box_noise_scale     :float = 1.0,
                 learnt_init_query   :bool  = False,
                 aux_loss            :bool  = True
                 ):
        super().__init__()
        # --------------- Basic setting ---------------
        ## Basic parameters
        self.in_dims = in_dims
        self.strides = strides
        self.num_queries = num_queries
        self.pos_embed_type = pos_embed_type
        self.num_classes = num_classes
        self.eps = 1e-2
        self.aux_loss = aux_loss
        ## Transformer parameters
        self.num_heads  = num_heads
        self.num_layers = num_layers
        self.num_levels = num_levels
        self.num_points = num_points
        self.ffn_dim  = ffn_dim
        self.dropout    = dropout
        self.act_type   = act_type
        self.return_intermediate = return_intermediate
        ## Denoising parameters
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.learnt_init_query = learnt_init_query

        # --------------- Network setting ---------------
        ## Input proj layers
        self.input_proj_layers = nn.ModuleList(
            BasicConv(in_dims[i], hidden_dim, kernel_size=1, act_type=None, norm_type="BN")
            for i in range(num_levels)
        )

        ## Deformable transformer decoder
        self.decoder = DeformableTransformerDecoder(
                                    d_model    = hidden_dim,
                                    num_heads  = num_heads,
                                    num_layers = num_layers,
                                    num_levels = num_levels,
                                    num_points = num_points,
                                    ffn_dim  = ffn_dim,
                                    dropout    = dropout,
                                    act_type   = act_type,
                                    return_intermediate = return_intermediate
                                    )
        
        ## Detection head for Encoder
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
            )
        self.enc_class_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        ## Detection head for Decoder
        self.dec_class_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_layers)
        ])

        ## Object query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        ## Denoising part
        if num_denoising > 0: 
            self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)

        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        prior_prob = 0.01
        cls_bias_init = float(-math.log((1 - prior_prob) / prior_prob))

        nn.init.constant_(self.enc_class_head.bias, cls_bias_init)
        nn.init.constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        nn.init.constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        for cls_, reg_ in zip(self.dec_class_head, self.dec_bbox_head):
            nn.init.constant_(cls_.bias, cls_bias_init)
            nn.init.constant_(reg_.layers[-1].weight, 0.)
            nn.init.constant_(reg_.layers[-1].bias, 0.)

        nn.init.xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            nn.init.xavier_uniform_(self.tgt_embed.weight)
        nn.init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        nn.init.xavier_uniform_(self.query_pos_head.layers[1].weight)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class, outputs_coord)]

    def generate_anchors(self, spatial_shapes, grid_size=0.05):
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))
            # [H, W, 2]
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()

            valid_WH = torch.as_tensor([w, h]).float()
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            # [H, W, 4] -> [1, N, 4], N=HxW
            anchors.append(torch.cat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4))
        # List[L, 1, N_i, 4] -> [1, N, 4], N=N_0 + N_1 + N_2 + ...
        anchors = torch.cat(anchors, dim=1)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        # Equal to operation: anchors = torch.masked_fill(anchors, ~valid_mask, torch.as_tensor(float("inf")))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        
        return anchors, valid_mask
    
    def get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj_layers[i](feat) for i, feat in enumerate(feats)]

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])
            # [B, C, H, W] -> [B, N, C], N=HxW
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))

        # [B, N, C], N = N_0 + N_1 + ...
        feat_flatten = torch.cat(feat_flatten, dim=1)
        level_start_index.pop()

        return (feat_flatten, spatial_shapes, level_start_index)

    def get_decoder_input(self,
                          memory,
                          spatial_shapes,
                          denoising_class=None,
                          denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        # Prepare input for decoder
        anchors, valid_mask = self.generate_anchors(spatial_shapes)
        anchors = anchors.to(memory.device)
        valid_mask = valid_mask.to(memory.device)
        
        # Process encoder's output
        memory = torch.where(valid_mask, memory, torch.as_tensor(0., device=memory.device))
        output_memory = self.enc_output(memory)

        # Head for encoder's output : [bs, num_quries, c]
        enc_outputs_class = self.enc_class_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        # Topk proposals from encoder's output
        topk = self.num_queries
        topk_ind = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]  # [bs, num_queries]
        enc_topk_logits = torch.gather(
            enc_outputs_class, 1, topk_ind.unsqueeze(-1).repeat(1, 1, self.num_classes))  # [bs, num_queries, nc]
        reference_points_unact = torch.gather(
            enc_outputs_coord_unact, 1, topk_ind.unsqueeze(-1).repeat(1, 1, 4))    # [bs, num_queries, 4]
        enc_topk_bboxes = F.sigmoid(reference_points_unact)

        if denoising_bbox_unact is not None:
            reference_points_unact = torch.cat(
                [denoising_bbox_unact, reference_points_unact], dim=1)

        # Extract region features
        if self.learnt_init_query:
            # [num_queries, c] -> [b, num_queries, c]
            target = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        else:
            # [num_queries, c] -> [b, num_queries, c]
            target = torch.gather(output_memory, 1, topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.detach()
        
        if denoising_class is not None:
            target = torch.cat([denoising_class, target], dim=1)

        return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits
    
    def forward(self, feats, targets=None):
        # input projection and embedding
        memory, spatial_shapes, _ = self.get_encoder_input(feats)

        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                                                         self.num_classes, 
                                                         self.num_queries, 
                                                         self.denoising_class_embed, 
                                                         num_denoising=self.num_denoising, 
                                                         label_noise_ratio=self.label_noise_ratio, 
                                                         box_noise_scale=self.box_noise_scale, )
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self.get_decoder_input(
            memory, spatial_shapes, denoising_class, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits = self.decoder(target,
                                              init_ref_points_unact,
                                              memory,
                                              spatial_shapes,
                                              self.dec_bbox_head,
                                              self.dec_class_head,
                                              self.query_pos_head,
                                              attn_mask)

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))
            
            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta

        return out


# ----------------- Dencoder for Segmentation task -----------------
## RTDETR's Transformer for Segmentation task
class SegTransformerDecoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        # TODO: design seg-decoder

    def forward(self, x):
        return


# ----------------- Dencoder for Pose estimation task -----------------
## RTDETR's Transformer for Pose estimation task
class PosTransformerDecoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        # TODO: design seg-decoder

    def forward(self, x):
        return


if __name__ == '__main__':
    import time
    from thop import profile
    cfg = {
        'out_stride': [8, 16, 32],
        # Transformer Decoder
        'transformer': 'rtdetr_transformer',
        'hidden_dim': 256,
        'de_num_heads': 8,
        'de_num_layers': 6,
        'de_ffn_dim': 1024,
        'de_dropout': 0.1,
        'de_act': 'gelu',
        'de_num_points': 4,
        'num_queries': 300,
        'learnt_init_query': False,
        'pe_temperature': 10000.,
        'dn_num_denoising': 100,
        'dn_label_noise_ratio': 0.5,
        'dn_box_noise_scale': 1,
    }
    bs = 1
    hidden_dim = cfg['hidden_dim']
    in_dims = [hidden_dim] * 3
    targets = [{
        'labels': torch.tensor([2, 4, 5, 8]).long(),
        'boxes':  torch.tensor([[0, 0, 10, 10], [12, 23, 56, 70], [0, 10, 20, 30], [50, 60, 55, 150]]).float()
    }] * bs
    pyramid_feats = [torch.randn(bs, hidden_dim, 80, 80),
                     torch.randn(bs, hidden_dim, 40, 40),
                     torch.randn(bs, hidden_dim, 20, 20)]
    model = build_transformer(cfg, in_dims, 80, True)
    model.train()

    t0 = time.time()
    outputs = model(pyramid_feats, targets)
    t1 = time.time()
    print('Time: ', t1 - t0)

    print(outputs["pred_logits"].shape)
    print(outputs["pred_boxes"].shape)

    print('==============================')
    model.eval()
    flops, params = profile(model, inputs=(pyramid_feats, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
