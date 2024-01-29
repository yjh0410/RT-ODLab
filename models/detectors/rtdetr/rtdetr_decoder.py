import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_, uniform_
from typing import List

try:
    from .basic_modules.basic import BasicConv, MLP, DeformableTransformerDecoder
    from .basic_modules.dn_compoments import get_contrastive_denoising_training_group
except:
    from  basic_modules.basic import BasicConv, MLP, DeformableTransformerDecoder
    from  basic_modules.dn_compoments import get_contrastive_denoising_training_group


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
                 trainable      :bool = False,
                 # transformer parameters
                 num_heads      :int   = 8,
                 num_layers     :int   = 1,
                 num_levels     :int   = 3,
                 num_points     :int   = 4,
                 mlp_ratio      :float = 4.0,
                 pe_temperature :float = 10000.,
                 dropout        :float = 0.1,
                 act_type       :str   = "relu",
                 return_intermediate :bool = False,
                 # Denoising parameters
                 num_denoising       :int  = 100,
                 label_noise_ratio   :float = 0.5,
                 box_noise_scale     :float = 1.0,
                 learnt_init_query   :bool  = True,
                 ):
        super().__init__()
        # --------------- Basic setting ---------------
        ## Basic parameters
        self.in_dims = in_dims
        self.strides = strides
        self.trainable = trainable
        self.num_queries = num_queries
        self.pos_embed_type = pos_embed_type
        self.num_classes = num_classes
        self.eps = 1e-2
        ## Transformer parameters
        self.num_heads  = num_heads
        self.num_layers = num_layers
        self.num_levels = num_levels
        self.num_points = num_points
        self.mlp_ratio  = mlp_ratio
        self.dropout    = dropout
        self.act_type   = act_type
        self.pe_temperature = pe_temperature
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
        self.transformer_decoder = DeformableTransformerDecoder(
            d_model    = hidden_dim,
            num_heads  = num_heads,
            num_layers = num_layers,
            num_levels = num_levels,
            num_points = num_points,
            mlp_ratio  = mlp_ratio,
            pe_temperature = pe_temperature,
            dropout        = dropout,
            act_type       = act_type,
            return_intermediate = return_intermediate
            )
        
        ## Detection head for Encoder
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
            )
        self.enc_class_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        ##  Detection head for Decoder
        self.dec_class_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_layers)
        ])

        ## Denoising part
        self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim)

        ## Object query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        self._reset_parameters()

    def _reset_parameters(self):
        def _linear_init(module):
            bound = 1 / math.sqrt(module.weight.shape[0])
            uniform_(module.weight, -bound, bound)
            if hasattr(module, "bias") and module.bias is not None:
                uniform_(module.bias, -bound, bound)

        # class and bbox head init
        prior_prob = 0.01
        cls_bias_init = float(-math.log((1 - prior_prob) / prior_prob))
        _linear_init(self.enc_class_head)
        constant_(self.enc_class_head.bias, cls_bias_init)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        for cls_, reg_ in zip(self.dec_class_head, self.dec_bbox_head):
            _linear_init(cls_)
            constant_(cls_.bias, cls_bias_init)
            constant_(reg_.layers[-1].weight, 0.)
            constant_(reg_.layers[-1].bias, 0.)

        _linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj:
            xavier_uniform_(l[0].weight)

    def generate_anchors(self, spatial_shapes, grid_size=0.05):
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()

            valid_WH = torch.as_tensor([w, h]).float()
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(torch.cat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = torch.cat(anchors, 1)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.as_tensor(float("inf")))
        
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
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.cat(feat_flatten, 1)
        level_start_index.pop()

        return (feat_flatten, spatial_shapes, level_start_index)

    def get_decoder_input(self,
                          memory,
                          spatial_shapes,
                          denoising_class=None,
                          denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        # prepare input for decoder
        anchors, valid_mask = self.generate_anchors(spatial_shapes)
        memory = torch.where(valid_mask, memory, torch.as_tensor(0.))
        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_class_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        topk = self.num_queries
        topk_ind = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
        reference_points_unact = torch.gather(enc_outputs_coord_unact, 1, topk_ind.unsqueeze(-1).repeat(1, 1, 4))
        enc_topk_bboxes = F.sigmoid(reference_points_unact)

        if denoising_bbox_unact is not None:
            reference_points_unact = torch.cat(
                [denoising_bbox_unact, reference_points_unact], 1)
        if self.trainable:
            reference_points_unact = reference_points_unact.detach()
        enc_topk_logits = torch.gather(enc_outputs_class, 1, topk_ind.unsqueeze(-1).repeat(1, 1, self.num_classes))

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        else:
            target = torch.gather(output_memory, 1, topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[1]))
            if self.training:
                target = target.detach()
        if denoising_class is not None:
            target = torch.cat([denoising_class, target], dim=1)

        return target, reference_points_unact, enc_topk_bboxes, enc_topk_logits
    
    def forward(self, feats, gt_meta=None):
        # input projection and embedding
        memory, spatial_shapes, _ = self.get_encoder_input(feats)

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(gt_meta,
                                                         self.num_classes,
                                                         self.num_queries,
                                                         self.denoising_class_embed.weight,
                                                         self.num_denoising,
                                                         self.label_noise_ratio,
                                                         self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self.get_decoder_input(
            memory, spatial_shapes, denoising_class, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits = self.transformer_decoder(target,
                                                          init_ref_points_unact,
                                                          memory,
                                                          spatial_shapes,
                                                          self.dec_bbox_head,
                                                          self.dec_class_head,
                                                          self.query_pos_head,
                                                          attn_mask)
        
        return (out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,
                dn_meta)


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
