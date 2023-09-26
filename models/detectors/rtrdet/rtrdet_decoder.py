import torch
import torch.nn as nn
import math

from .rtrdet_basic import get_clones, TRDecoderLayer, MLP


# Transformer Decoder Module
class TransformerDecoder(nn.Module):
    def __init__(self, cfg, num_classes, return_intermediate=False):
        super().__init__()
        # -------------------- Basic Parameters ---------------------
        self.d_model = round(cfg['d_model'] * cfg['width'])
        self.num_queries = cfg['decoder_num_queries']
        self.num_pattern = cfg['decoder_num_pattern']
        self.num_deocder = cfg['num_decoder']
        self.num_classes = num_classes
        self.stop_layer_id = cfg['num_decoder'] if cfg['stop_layer_id'] == -1 else cfg['stop_layer_id']
        self.return_intermediate = return_intermediate
        self.scale = 2 * 3.141592653589793

        # -------------------- Network Parameters ---------------------
        ## Decoder
        decoder_layer = TRDecoderLayer(d_model   = self.d_model,
                                       num_heads = cfg['decoder_num_head'],
                                       mlp_ratio = cfg['decoder_mlp_ratio'],
                                       dropout   = cfg['decoder_dropout'],
                                       act_type  = cfg['decoder_act']
                                       )
        self.decoder_layers = get_clones(decoder_layer, self.num_deocder)
        ## Pattern embed
        self.pattern = nn.Embedding(self.num_pattern, self.d_model)
        ## Spatial embed
        self.position = nn.Embedding(self.num_queries, 2)
        ## Output head
        self.class_embed = nn.Linear(self.d_model, self.num_classes)
        self.bbox_embed  = MLP(self.d_model, self.d_model, 4, 3)
        # Adaptive pos_embed
        self.adapt_pos2d = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )

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


    def pos2posemb2d(self, pos, temperature=10000):
        pos = pos * self.scale
        num_pos_feats = self.d_model // 2
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        pos_x = pos[..., 0, None] / dim_t
        pos_y = pos[..., 1, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        posemb = torch.cat((pos_y, pos_x), dim=-1)
        
        return posemb
    

    def forward(self, memory, memory_pos):
        # reshape: [B, C, H, W] -> [B, N, C], N = HW
        memory = memory.flatten(2).permute(0, 2, 1).contiguous()
        memory_pos = memory_pos.flatten(2).permute(0, 2, 1).contiguous()
        memory_pos = self.adapt_pos2d(memory_pos)
        bs, _, channels = memory.size()

        # reshape: [Na, C] -> [1, Na, 1, C] -> [1, Na, Np, C] -> [1, Nq, C], Nq = Na*Np
        tgt = self.pattern.weight.reshape(1, self.num_pattern, 1, channels).repeat(bs, 1, self.num_queries, 1)
        tgt = tgt.reshape(bs, self.num_pattern * self.num_queries, channels)
        
        # Reference points
        reference_points = self.position.weight.unsqueeze(0).repeat(bs, self.num_pattern, 1)

        # Decoder
        output_classes = []
        output_coords = []
        for layer_id, layer in enumerate(self.decoder_layers):
            # query embed
            query_pos = self.adapt_pos2d(self.pos2posemb2d(reference_points))
            tgt = layer(tgt, memory, query_pos, memory_pos)
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


    def inverse_sigmoid(self, x):
        x = x.clamp(min=0, max=1)
        return torch.log(x.clamp(min=1e-5)/(1 - x).clamp(min=1e-5))

    
# build detection head
def build_decoder(cfg, num_classes, return_intermediate=False):
    decoder = TransformerDecoder(cfg, num_classes, return_intermediate) 

    return decoder
