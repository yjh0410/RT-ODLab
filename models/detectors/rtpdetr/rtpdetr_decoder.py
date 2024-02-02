import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .basic_modules.basic import LayerNorm2D
    from .basic_modules.transformer import GlobalDecoder
except:
    from  basic_modules.basic import LayerNorm2D
    from  basic_modules.transformer import GlobalDecoder

def build_transformer(cfg, return_intermediate=False):
    if cfg['transformer'] == 'plain_detr_transformer':
        return PlainDETRTransformer(d_model             = cfg['hidden_dim'],
                                    num_heads           = cfg['de_num_heads'],
                                    mlp_ratio           = cfg['de_mlp_ratio'],
                                    dropout             = cfg['de_dropout'],
                                    act_type            = cfg['de_act'],
                                    pre_norm            = cfg['de_pre_norm'],
                                    rpe_hidden_dim      = cfg['rpe_hidden_dim'],
                                    feature_stride      = cfg['out_stride'],
                                    num_layers          = cfg['de_num_layers'],
                                    return_intermediate = return_intermediate,
                                    use_checkpoint      = cfg['use_checkpoint'],
                                    num_queries_one2one = cfg['num_queries_one2one'],
                                    num_queries_one2many = cfg['num_queries_one2many'],
                                    proposal_feature_levels = cfg['proposal_feature_levels'],
                                    proposal_in_stride      = cfg['out_stride'],
                                    proposal_tgt_strides    = cfg['proposal_tgt_strides'],
                                    )


# ----------------- Dencoder for Detection task -----------------
## PlainDETR's Transformer for Detection task
class PlainDETRTransformer(nn.Module):
    def __init__(self,
                 # Decoder layer params
                 d_model        :int   = 256,
                 num_heads      :int   = 8,
                 mlp_ratio      :float = 4.0,
                 dropout        :float = 0.1,
                 act_type       :str   = "relu",
                 pre_norm       :bool  = False,
                 rpe_hidden_dim :int   = 512,
                 feature_stride :int   = 16,
                 num_layers     :int   = 6,
                 # Decoder params
                 return_intermediate :bool = False,
                 use_checkpoint      :bool = False,
                 num_queries_one2one :int = 300,
                 num_queries_one2many :int = 1500,
                 proposal_feature_levels :int = 3,
                 proposal_in_stride      :int = 16,
                 proposal_tgt_strides    :int = [8, 16, 32],
                 ):
        super().__init__()
        # ------------ Basic setting ------------
        ## Model
        self.d_model = d_model
        self.num_heads = num_heads
        self.rpe_hidden_dim = rpe_hidden_dim
        self.mlp_ratio = mlp_ratio
        self.act_type = act_type
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        ## Trick
        self.use_checkpoint = use_checkpoint
        self.num_queries_one2one = num_queries_one2one
        self.num_queries_one2many = num_queries_one2many
        self.proposal_feature_levels = proposal_feature_levels
        self.proposal_tgt_strides = proposal_tgt_strides
        self.proposal_in_stride = proposal_in_stride
        self.proposal_min_size = 50

        # --------------- Network setting ---------------
        ## Global Decoder
        self.decoder = GlobalDecoder(d_model, num_heads, mlp_ratio, dropout, act_type, pre_norm,
                                     rpe_hidden_dim, feature_stride, num_layers, return_intermediate,
                                     use_checkpoint,)
        
        ## Two stage
        self.enc_output = nn.Linear(d_model, d_model)
        self.enc_output_norm = nn.LayerNorm(d_model)
        self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
        self.pos_trans_norm = nn.LayerNorm(d_model * 2)

        ## Expand layers
        if proposal_feature_levels > 1:
            assert len(proposal_tgt_strides) == proposal_feature_levels

            self.enc_output_proj = nn.ModuleList([])
            for stride in proposal_tgt_strides:
                if stride == proposal_in_stride:
                    self.enc_output_proj.append(nn.Identity())
                elif stride > proposal_in_stride:
                    scale = int(math.log2(stride / proposal_in_stride))
                    layers = []
                    for _ in range(scale - 1):
                        layers += [
                            nn.Conv2d(d_model, d_model, kernel_size=2, stride=2),
                            LayerNorm2D(d_model),
                            nn.GELU()
                        ]
                    layers.append(nn.Conv2d(d_model, d_model, kernel_size=2, stride=2))
                    self.enc_output_proj.append(nn.Sequential(*layers))
                else:
                    scale = int(math.log2(proposal_in_stride / stride))
                    layers = []
                    for _ in range(scale - 1):
                        layers += [
                            nn.ConvTranspose2d(d_model, d_model, kernel_size=2, stride=2),
                            LayerNorm2D(d_model),
                            nn.GELU()
                        ]
                    layers.append(nn.ConvTranspose2d(d_model, d_model, kernel_size=2, stride=2))
                    self.enc_output_proj.append(nn.Sequential(*layers))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if hasattr(self.decoder, '_reset_parameters'):
            print('decoder re-init')
            self.decoder._reset_parameters()

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = self.d_model // 2
        temperature = 10000
        scale = 2 * torch.pi

        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device
        )
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack(
            (pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4
        ).flatten(2)

        return pos

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)

        return valid_ratio

    def expand_encoder_output(self, memory, memory_padding_mask, spatial_shapes):
        assert spatial_shapes.size(0) == 1, f'Get encoder output of shape {spatial_shapes}, not sure how to expand'

        bs, _, c = memory.shape
        h, w = spatial_shapes[0]

        _out_memory = memory.view(bs, h, w, c).permute(0, 3, 1, 2)
        _out_memory_padding_mask = memory_padding_mask.view(bs, h, w)

        out_memory, out_memory_padding_mask, out_spatial_shapes = [], [], []
        for i in range(self.proposal_feature_levels):
            mem = self.enc_output_proj[i](_out_memory)
            mask = F.interpolate(
                _out_memory_padding_mask[None].float(), size=mem.shape[-2:]
            ).to(torch.bool)

            out_memory.append(mem)
            out_memory_padding_mask.append(mask.squeeze(0))
            out_spatial_shapes.append(mem.shape[-2:])

        out_memory = torch.cat([mem.flatten(2).transpose(1, 2) for mem in out_memory], dim=1)
        out_memory_padding_mask = torch.cat([mask.flatten(1) for mask in out_memory_padding_mask], dim=1)
        out_spatial_shapes = torch.as_tensor(out_spatial_shapes, dtype=torch.long, device=out_memory.device)
        
        return out_memory, out_memory_padding_mask, out_spatial_shapes

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        if self.proposal_feature_levels > 1:
            memory, memory_padding_mask, spatial_shapes = self.expand_encoder_output(
                memory, memory_padding_mask, spatial_shapes
            )
        N_, S_, C_ = memory.shape
        # base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            stride = self.proposal_tgt_strides[lvl]

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) * stride
            wh = torch.ones_like(grid) * self.proposal_min_size * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += H_ * W_
        output_proposals = torch.cat(proposals, 1)

        H_, W_ = spatial_shapes[0]
        stride = self.proposal_tgt_strides[0]
        mask_flatten_ = memory_padding_mask[:, :H_*W_].view(N_, H_, W_, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1, keepdim=True) * stride
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1, keepdim=True) * stride
        img_size = torch.cat([valid_W, valid_H, valid_W, valid_H], dim=-1)
        img_size = img_size.unsqueeze(1) # [BS, 1, 4]

        output_proposals_valid = (
            (output_proposals > 0.01 * img_size) & (output_proposals < 0.99 * img_size)
        ).all(-1, keepdim=True)
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1).repeat(1, 1, 1),
            max(H_, W_) * stride,
        )
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid,
            max(H_, W_) * stride,
        )

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))

        max_shape = (valid_H[:, None, :], valid_W[:, None, :])
        return output_memory, output_proposals, max_shape
    
    def get_reference_points(self, memory, mask_flatten, spatial_shapes):
        output_memory, output_proposals, max_shape = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )

        # hack implementation for two-stage Deformable DETR
        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
        enc_outputs_delta = self.decoder.bbox_embed[self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = self.decoder.box_xyxy_to_cxcywh(self.decoder.delta2bbox(
            output_proposals,
            enc_outputs_delta,
            max_shape
        ))

        topk = self.two_stage_num_proposals
        topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        )
        topk_coords_unact = topk_coords_unact.detach()
        reference_points = topk_coords_unact
        
        return (reference_points, max_shape, enc_outputs_class,
                enc_outputs_coord_unact, enc_outputs_delta, output_proposals)

    def forward(self, src, mask, pos_embed, query_embed=None, self_attn_mask=None):
        # Prepare input for encoder
        bs, c, h, w = src.shape
        src_flatten = src.flatten(2).transpose(1, 2)
        mask_flatten = mask.flatten(1)
        pos_embed_flatten = pos_embed.flatten(2).transpose(1, 2)
        spatial_shapes = torch.as_tensor([(h, w)], dtype=torch.long, device=src_flatten.device)

        # Prepare input for decoder
        memory = src_flatten
        bs, _, c = memory.shape
       
        # Two stage trick
        if self.training:
            self.two_stage_num_proposals = self.num_queries_one2one + self.num_queries_one2many
        else:
            self.two_stage_num_proposals = self.num_queries_one2one
        (reference_points, max_shape, enc_outputs_class,
        enc_outputs_coord_unact, enc_outputs_delta, output_proposals) \
            = self.get_reference_points(memory, mask_flatten, spatial_shapes)
        init_reference_out = reference_points
        pos_trans_out = torch.zeros((bs, self.two_stage_num_proposals, 2*c), device=init_reference_out.device)
        pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(reference_points)))

        # Mixed selection trick
        tgt = query_embed.unsqueeze(0).expand(bs, -1, -1)
        query_embed, _ = torch.split(pos_trans_out, c, dim=2)

        # Decoder
        hs, inter_references = self.decoder(tgt,
                                            reference_points,
                                            memory,
                                            pos_embed_flatten,
                                            spatial_shapes,
                                            query_embed,
                                            mask_flatten,
                                            self_attn_mask,
                                            max_shape
                                            )
        inter_references_out = inter_references

        return (hs,
                init_reference_out,
                inter_references_out,
                enc_outputs_class,
                enc_outputs_coord_unact,
                enc_outputs_delta,
                output_proposals,
                max_shape
                )


# ----------------- Dencoder for Segmentation task -----------------
## PlainDETR's Transformer for Segmentation task
class SegTransformerDecoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        # TODO: design seg-decoder

    def forward(self, x):
        return


# ----------------- Dencoder for Pose estimation task -----------------
## PlainDETR's Transformer for Pose estimation task
class PosTransformerDecoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        # TODO: design seg-decoder

    def forward(self, x):
        return


if __name__ == '__main__':
    import time
    from thop import profile
    from basic_modules.basic import MLP
    from basic_modules.transformer import get_clones

    cfg = {
        'out_stride': 16,
        # Transformer Decoder
        'transformer': 'plain_detr_transformer',
        'hidden_dim': 256,
        'num_queries': 300,
        'de_num_heads': 8,
        'de_num_layers': 6,
        'de_mlp_ratio': 4.0,
        'de_dropout': 0.1,
        'de_act': 'gelu',
        'de_pre_norm': True,
        'rpe_hidden_dim': 512,
        'use_checkpoint': False,
        'proposal_feature_levels': 3,
        'proposal_tgt_strides': [8, 16, 32],
    }
    feat = torch.randn(1, cfg['hidden_dim'], 40, 40)
    mask = torch.zeros(1, 40, 40)
    pos_embed = torch.randn(1, cfg['hidden_dim'], 40, 40)
    query_embed = torch.randn(cfg['num_queries'], cfg['hidden_dim'])

    model = build_transformer(cfg, True)

    class_embed = nn.Linear(cfg['hidden_dim'], 80)
    bbox_embed = MLP(cfg['hidden_dim'], cfg['hidden_dim'], 4, 3)
    class_embed = get_clones(class_embed, cfg['de_num_layers'] + 1)
    bbox_embed = get_clones(bbox_embed, cfg['de_num_layers'] + 1)

    model.decoder.bbox_embed = bbox_embed
    model.decoder.class_embed = class_embed

    model.train()
    t0 = time.time()
    outputs = model(feat, mask, pos_embed, query_embed)
    (hs,
     init_reference_out,
     inter_references_out,
     enc_outputs_class,
     enc_outputs_coord_unact,
     enc_outputs_delta,
     output_proposals,
     max_shape
     ) = outputs
    t1 = time.time()
    print('Time: ', t1 - t0)
    print(hs.shape)
    print(init_reference_out.shape)
    print(inter_references_out.shape)
    print(enc_outputs_class.shape)
    print(enc_outputs_coord_unact.shape)
    print(enc_outputs_delta.shape)
    print(output_proposals.shape)

    print('==============================')
    model.eval()
    flops, params = profile(model, inputs=(feat, mask, pos_embed, query_embed, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
