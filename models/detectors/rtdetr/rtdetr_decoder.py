import torch
import torch.nn as nn

from .rtdetr_basic import get_clones, TRDecoderLayer, MLP


# Transformer Decoder Module
class TransformerDecoder(nn.Module):
    def __init__(self, cfg, in_dim, return_intermediate=False):
        super().__init__()
        # -------------------- Basic Parameters ---------------------
        self.d_model = in_dim
        self.query_dim = 4  # For RefPoint head
        self.scale = 2 * 3.141592653589793
        self.num_queries = cfg['num_queries']
        self.num_deocder_layers = cfg['num_decoder_layers']
        self.return_intermediate = return_intermediate
        self.ffn_dim = round(cfg['de_dim_feedforward']*cfg['width'])

        # -------------------- Network Parameters ---------------------
        ## Decoder
        decoder_layer = TRDecoderLayer(
            d_model=in_dim,
            dim_feedforward=self.ffn_dim,
            num_heads=cfg['de_num_heads'],
            dropout=cfg['de_dropout'],
            act_type=cfg['de_act']
        )
        self.decoder_layers = get_clones(decoder_layer, cfg['num_decoder_layers'])
        ## RefPoint Embed
        self.refpoint_embed = nn.Embedding(cfg['num_queries'], 4)
        self.ref_point_head = MLP(self.query_dim // 2 * in_dim, in_dim, in_dim, 2)
        ## Object Query Embed
        self.object_query = nn.Embedding(cfg['num_queries'], in_dim)
        nn.init.normal_(self.object_query.weight.data)
        ## TODO: Group queries


        self.bbox_embed = None
        self.class_embed = None


    def inverse_sigmoid(self, x):
        x = x.clamp(min=0, max=1)
        return torch.log(x.clamp(min=1e-5)/(1 - x).clamp(min=1e-5))


    def query_sine_embed(self, num_feats, reference_points):
        dim_t = torch.arange(num_feats, dtype=torch.float32, device=reference_points.device)
        dim_t_ = torch.div(dim_t, 2, rounding_mode='floor') / num_feats
        dim_t = 10000 ** (2 * dim_t_)

        x_embed = reference_points[:, :, 0] * self.scale
        y_embed = reference_points[:, :, 1] * self.scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        w_embed = reference_points[:, :, 2] * self.scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = reference_points[:, :, 3] * self.scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)
        query_sine_embed = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)

        return query_sine_embed
    

    def forward(self, memory, memory_pos):
        bs, _, channels = memory.size()
        num_feats = channels // 2

        # prepare tgt & refpoint
        tgt = self.object_query.weight[None].repeat(bs, 1, 1)
        refpoint_embed = self.refpoint_embed.weight[None].repeat(bs, 1, 1)

        intermediate = []
        reference_points = refpoint_embed.sigmoid()
        ref_points = [reference_points]

        # main process
        output = tgt
        for layer_id, layer in enumerate(self.decoder_layers):
            # Conditional query
            query_sine_embed = self.query_sine_embed(num_feats, reference_points)
            query_pos = self.ref_point_head(query_sine_embed) # [B, N, C]

            # Decoder
            output = layer(
                    # input for decoder
                    tgt = output,
                    tgt_query_pos = query_pos,
                    # input from encoder
                    memory = memory,
                    memory_pos = memory_pos,
                )

            # Iter update
            if self.bbox_embed is not None:
                delta_unsig = self.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + self.inverse_sigmoid(reference_points)
                new_reference_points = outputs_unsig.sigmoid()

                reference_points = new_reference_points.detach()
                ref_points.append(new_reference_points)

            intermediate.append(output)

        return torch.stack(intermediate), torch.stack(ref_points)


# build detection head
def build_decoder(cfg, in_dim, return_intermediate=False):
    decoder = TransformerDecoder(cfg, in_dim, return_intermediate=return_intermediate) 

    return decoder
