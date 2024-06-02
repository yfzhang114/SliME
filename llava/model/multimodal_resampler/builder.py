import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
import numpy as np
import math
from torch.nn import functional as F
from llava.model.multimodal_resampler.sampler import Resampler, ResamplerWithText

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        return F.interpolate(
            abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
            size=(tgt_size, tgt_size),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)
    else:
        return abs_pos

def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C

    # Reshape the absolute positions to one-dimensional encoding
    src_size = abs_pos.size(0)
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        abs_pos = F.interpolate(
            abs_pos.float().unsqueeze(0).unsqueeze(0),
            size=(tgt_size, 1),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).squeeze(0)

    return abs_pos.to(dtype=dtype)

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

class TextGuidedRouterAttention(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """
    def __init__(
            self,
            grid_size,
            embed_dim,
            num_heads,
            kv_dim=None,
            temp=1.0,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.num_queries = 1
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.temp = temp

        # self.pos_embed = nn.Parameter(
        #     torch.from_numpy(get_1d_sincos_pos_embed_from_grid(embed_dim, grid_size)).float()
        # ).requires_grad_(False)

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query, std=.02)

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)
        self.ln_post = norm_layer(embed_dim)
        self.prob_proj = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim // 4, 1))
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, text, attn_mask=None):
        # size(2) is the patch num        
        x = self.ln_q(x)
        text = self.ln_kv(text)
        attn_mask = attn_mask.bool()
        # query = self.ln_q(query)
        out, weight = self.cross_attn(
            x, 
            text,
            text,
            key_padding_mask=~attn_mask)
        out = self.ln_post(out)
        out = self.prob_proj(out).squeeze(-1)
        return softmax_with_temperature(out, self.temp) 
        
    def _repeat(self, query, N: int, dim: int = 1):
        if dim == 1:
            return query.unsqueeze(1).repeat(1, N, 1)
        elif dim == 0:
            return query.unsqueeze(0).repeat(N, 1, 1)
        elif dim == 2:
            return query.unsqueeze(2).repeat(1, 1, N)


def softmax_with_temperature(logits, temperature=1.0):
    scaled_logits = logits / temperature
    return F.softmax(scaled_logits, dim=-1)

class TextGuidedRouterCosine(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """
    def __init__(self, pad_token_id, temp=1.0, embed_dim=4096):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.temp = temp

    def forward(self, image, text_embedding, attn_mask=None):        

        similarity_matrix = torch.nn.functional.cosine_similarity(image.unsqueeze(1), text_embedding.unsqueeze(0), dim=-1)

        if attn_mask is not None:
            attn_mask = attn_mask == False
            similarity_matrix = similarity_matrix.masked_fill(attn_mask.unsqueeze(0), float(0))

            similarity_matrix = similarity_matrix.sum(dim=-1)
        else:
            similarity_matrix = similarity_matrix.mean(dim=-1)
            
        return similarity_matrix #softmax_with_temperature(similarity_matrix, self.temp) 

class SpatialMap(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_std = 1 / torch.sqrt(torch.tensor(config.hidden_size))
        self.image_newline = nn.Parameter(
            torch.randn(config.hidden_size) * embed_std
        )

    def forward(self, x, *args, **kwargs):
        x = torch.cat((
                                x,
                                self.image_newline[:, None, None].expand(*x.shape[:-1], 1).to(x.device)
                            ), dim=-1)
        return x

    @property
    def config(self):
        return {"mm_resampler_type": 'spatial'}
    
class TextGuidedSampler(nn.Module):

    def __init__(self, projector_type, config):
        super().__init__()
        self.num_queries = config.mm_resampler_dim
        self.topp = config.mm_resampler_topp
        self.temp = config.mm_resampler_temp
        self.grid_size = int(math.sqrt(self.num_queries))
        if projector_type == 'cosine':
            self.selector = TextGuidedRouterCosine(pad_token_id=config.pad_token_id, temp=self.temp, embed_dim = config.hidden_size,)
        elif projector_type == 'qformer':
            self.selector = TextGuidedRouterAttention(
                    grid_size=1, # patch tokens H*W
                    embed_dim=config.hidden_size,
                    num_heads=config.hidden_size // 128,
                    temp=self.temp
                )
        self.post_qformer = Resampler(
                        grid_size=self.grid_size,
                        embed_dim = config.mm_hidden_size,
                        num_heads = config.mm_hidden_size // 128,
                        kv_dim=config.mm_hidden_size,
                        llm_hidden_size=config.hidden_size,
                    )
        
        
    def forward(self, local_f, text_embedding, attn_mask=None):
        
        local_probs = self.selector(local_f, text_embedding, attn_mask)
        local_probs = softmax_with_temperature(local_probs, temperature=0.5)
                
        # Add gumbel noise
        if self.training:
            gumbel_noise = torch.randn_like(local_probs) * 0.1
            local_probs = (local_probs + gumbel_noise)

        local_probs = softmax_with_temperature(local_probs, temperature=self.temp)
        # Sort the probabilities in descending order and get the corresponding indices
        sorted_probs, sorted_indices = torch.sort(local_probs, descending=True)

        # Calculate the cumulative sum of probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)

        # Find the indices where the cumulative sum exceeds the threshold
        selected_indices = (cumulative_probs <= self.topp).nonzero(as_tuple=True)[0]

        # Include one more index to ensure the sum exceeds the threshold
        if selected_indices.numel() < sorted_indices.numel():
            selected_indices = sorted_indices[:selected_indices.numel() + 1]

        selected_indices, _ = selected_indices.sort(descending=False)
        
        # mask = torch.zeros_like(local_probs).scatter_(-1, selected_indices, 1)
        # mask_topp = (mask - local_probs).detach() + local_probs
        # selected_probs = local_f * mask_topp.unsqueeze(-1)
        
        local_f = local_f[selected_indices]
        # print(selected_probs.shape[0], local_f.shape[0], end='\t')
        # print(local_probs[:10], end='\t')
        return local_f


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_resampler_type": 'identity'}
    
def build_vision_sampler(config, delay_load=False, **kwargs):
    mm_resampler_type = getattr(config, 'mm_resampler_type', None)
    
    if mm_resampler_type == 'identity' or mm_resampler_type is None:
        return IdentityMap()
    
    return TextGuidedSampler(mm_resampler_type, config)