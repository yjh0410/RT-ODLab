import math
import torch
import torch.nn as nn

from .rtrdet_basic import get_clones, TREncoderLayer, TRDecoderLayer, MLP


class RTRDetTransformer(nn.Module):
    def __init__(self, cfg, num_classes, return_intermediate):
        super().__init__()
        # -------------------- Basic Parameters ---------------------
        self.d_model = round(cfg['d_model']*cfg['width'])
        self.num_classes = num_classes
        self.num_encoder = cfg['num_encoder']
        self.num_deocder = cfg['num_decoder']
        self.num_queries = cfg['decoder_num_queries']
        self.num_pattern = cfg['decoder_num_pattern']
        self.stop_layer_id = cfg['num_decoder'] if cfg['stop_layer_id'] == -1 else cfg['stop_layer_id']
        self.return_intermediate = return_intermediate
        self.scale = 2 * 3.141592653589793

        # -------------------- Network Parameters ---------------------
        ## Transformer Encoder
        encoder_layer = TREncoderLayer(
            self.d_model, cfg['encoder_num_head'], cfg['encoder_mlp_ratio'], cfg['encoder_dropout'], cfg['encoder_act'])
        self.encoder_layers = get_clones(encoder_layer, cfg['num_encoder'])

        ## Transformer Decoder
        decoder_layer = TRDecoderLayer(
            self.d_model, cfg['decoder_num_head'], cfg['decoder_mlp_ratio'], cfg['decoder_dropout'], cfg['decoder_act'])
        self.decoder_layers = get_clones(decoder_layer, cfg['num_decoder'])

        ## Pattern embed
        self.pattern = nn.Embedding(cfg['decoder_num_pattern'], self.d_model)

        ## Position embed
        self.position = nn.Embedding(cfg['decoder_num_queries'], 2)

        ## Adaptive PosEmbed
        self.adapt_pos2d = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        ## Output head
        self.class_embed = nn.Linear(self.d_model, self.num_classes)
        self.bbox_embed  = MLP(self.d_model, self.d_model, 4, 3)

        self._reset_parameters()

    def _reset_parameters(self):
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value

        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
        nn.init.uniform_(self.position.weight.data, 0, 1)

        self.class_embed = nn.ModuleList([self.class_embed for _ in range(self.num_deocder)])
        self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(self.num_deocder)])

    def generate_posembed(self, x, temperature=10000):
        num_pos_feats, hs, ws = x.shape[1]//2, x.shape[2], x.shape[3]
        # generate xy coord mat
        y_embed, x_embed = torch.meshgrid(
            [torch.arange(1, hs+1, dtype=torch.float32),
             torch.arange(1, ws+1, dtype=torch.float32)])
        y_embed = y_embed / (hs + 1e-6) * self.scale
        x_embed = x_embed / (ws + 1e-6) * self.scale
    
        # [H, W] -> [1, H, W]
        y_embed = y_embed[None, :, :].to(x.device)
        x_embed = x_embed[None, :, :].to(x.device)

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t_ = torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats
        dim_t = temperature ** (2 * dim_t_)

        pos_x = torch.div(x_embed[..., None], dim_t)
        pos_y = torch.div(y_embed[..., None], dim_t)
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)

        # [B, H, W, C] -> [B, C, H, W]
        pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        return pos_embed
        
    def pos2posemb2d(self, pos, temperature=10000):
        scale = 2 * math.pi
        num_pos_feats = self.d_model // 2
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t_ = torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats
        dim_t = temperature ** (2 * dim_t_)
        pos_x = pos[..., 0, None] / dim_t
        pos_y = pos[..., 1, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        posemb = torch.cat((pos_y, pos_x), dim=-1)
        
        return posemb

    def inverse_sigmoid(self, x):
        x = x.clamp(min=0, max=1)
        return torch.log(x.clamp(min=1e-5)/(1 - x).clamp(min=1e-5))

    def forward(self, src1=None, src2=None):
        """
        Input:
            src1: C4-level feature -> [B, C4, H4, W4]
            sec2: C5-level feature -> [B, C5, H5, W5]
        Output:

        """
        bs, c, h, w = src2.size()

        # ------------------------ Transformer Encoder ------------------------
        ## Generate pos_embed for src2
        pos2d_embed_2 = self.generate_posembed(src2)
        
        ## Reshape: [B, C, H, W] -> [B, N, C], N = HW
        src2 = src2.flatten(2).permute(0, 2, 1).contiguous()
        pos2d_embed_2 = self.adapt_pos2d(pos2d_embed_2.flatten(2).permute(0, 2, 1).contiguous())
        
        ## Encoder layer
        for layer_id, encoder_layer in enumerate(self.encoder_layers):
            src2 = encoder_layer(src2, pos2d_embed_2)
        
        ## Feature fusion
        src2 = src2.permute(0, 2, 1).reshape(bs, c, h, w)
        if src1 is not None:
            src1 = src1 + nn.functional.interpolate(src2, scale_factor=2.0)
        else:
            src1 = src2
        
        # ------------------------ Transformer Decoder ------------------------
        ## Generate pos_embed for src1
        pos2d_embed_1 = self.generate_posembed(src1)

        ## Reshape memory: [B, C, H, W] -> [B, N, C], N = HW
        src1 = src1.flatten(2).permute(0, 2, 1).contiguous()
        pos2d_embed_1 = self.adapt_pos2d(pos2d_embed_1.flatten(2).permute(0, 2, 1).contiguous())

        ## Reshape tgt: [Na, C] -> [1, Na, 1, C] -> [1, Na, Np, C] -> [1, Nq, C], Nq = Na*Np
        tgt = self.pattern.weight.reshape(1, self.num_pattern, 1, c).repeat(bs, 1, self.num_queries, 1)
        tgt = tgt.reshape(bs, self.num_pattern * self.num_queries, c)
        
        ## Prepare reference points
        reference_points = self.position.weight.unsqueeze(0).repeat(bs, self.num_pattern, 1)

        ## Decoder layer
        output_classes = []
        output_coords = []
        for layer_id, decoder_layer in enumerate(self.decoder_layers):
            ## query embed
            query_pos = self.adapt_pos2d(self.pos2posemb2d(reference_points))
            tgt = decoder_layer(tgt, query_pos, src1, pos2d_embed_1)
            reference = self.inverse_sigmoid(reference_points)
            ## class
            outputs_class = self.class_embed[layer_id](tgt)
            ## bbox
            tmp = self.bbox_embed[layer_id](tgt)
            tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()

            output_classes.append(outputs_class)
            output_coords.append(outputs_coord)

            if layer_id == self.stop_layer_id:
                break

        return torch.stack(output_classes), torch.stack(output_coords)

    
# build detection head
def build_transformer(cfg, num_classes, return_intermediate=False):
    if cfg['transformer'] == "RTRDetTransformer":
        transoformer = RTRDetTransformer(cfg, num_classes, return_intermediate) 

    return transoformer
