import torch 
import torch.nn as nn
import math
from typing import Optional
from dataclasses import dataclass

# Label and timestep
# From https://github.com/facebookresearch/DiT/blob/main/models.py

@dataclass
class DiTConfig:
    hidden_size: int = 768 
    patch_size: int = 8
    n_emb: int = 768       
    num_heads: int = 12    
    emb_dim: int = 768      


class TimestepEmbedder(nn.Module):
    def __init__(self, frequency_embedding_size=256):
        super().__init__()
        self.mlp = MLP(DiTConfig())
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, config, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = config.emb_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if config.emb_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, DiTConfig())
        t_emb = self.mlp(t_freq)
        return t_emb


def get_1d_sincos_pos_embed_from_grid(config, pos):
    assert config.n_emb % 2 == 0
    omega = torch.arange(config.n_emb // 2, dtype=torch.float32)
    omega /= config.n_emb / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb


def get_1d_sincos_pos_embed(config, length):
    grid = torch.arange(length, dtype=torch.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(config, grid)
    return pos_embed.unsqueeze(0)


def get_2d_sincos_pos_embed(config, length, interpolation_scale=1.0, base_size=16, device: Optional[torch.device] = None):
    grid_size = int(length**0.5)
    assert grid_size * grid_size == length
    grid_h = (
        torch.arange(grid_size, device=device, dtype=torch.float32)
        / (grid_size / base_size)
        / interpolation_scale
    )
    grid_w = (
        torch.arange(grid_size, device=device, dtype=torch.float32)
        / (grid_size / base_size)
        / interpolation_scale
    )
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(config, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(config, grid):
    assert config.n_emb % 2 == 0
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(DiTConfig(n_emb=config.n_emb // 2), grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(DiTConfig(n_emb=config.n_emb // 2), grid[1])  # (H*W, D/2)
    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert (config.n_emb % config.num_heads == 0), "n_emb must be divisible by num_heads"
        self.head_dim = config.n_emb // config.num_heads
        self.num_heads = config.num_heads
        self.n_emb = config.n_emb
        self.dropout = nn.Dropout(0.1)
        self.q = nn.Linear(config.n_emb, config.n_emb, bias=False)
        self.k = nn.Linear(config.n_emb, config.n_emb, bias=False)
        self.v = nn.Linear(config.n_emb, config.n_emb, bias=False)
        self.proj = nn.Linear(config.n_emb, config.n_emb)
        self.register_buffer("mask", torch.tril(torch.ones(1024, 1024)))

    def forward(self, x):
        B, num_tokens, d_in = x.shape
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        q = query.view(B, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, N, head_dim)
        k = key.view(B, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = value.view(B, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # calculate attention score
        attention_score = q @ k.transpose(2, 3)
        mask = self.mask.bool()[:num_tokens, :num_tokens]
        attention_score.masked_fill_(~mask, -torch.inf)

        attention_weights = torch.softmax(attention_score / (self.head_dim ** 0.5), dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector = (attention_weights @ v).transpose(1, 2)
        context_vector = context_vector.contiguous().view(B, num_tokens, self.n_emb)
        context_vector = self.proj(context_vector)

        return context_vector


class PatchEmbed(nn.Module):
    def __init__(self, config, in_channels=4, img_size: int = 32):
        super().__init__()
        self.dim = config.n_emb
        self.patch_size = config.patch_size
        self.num_patches = (img_size // self.patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, config.n_emb, kernel_size=(self.patch_size, self.patch_size), 
            stride=self.patch_size, bias=False
        )

    def forward(self, x):
        B, C, H, W = x.shape 
        x = self.proj(x)  # (B, hidden_size, num_patches_side, num_patches_side)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_size)
        return x


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.Linear(config.n_emb, 4 * config.n_emb)
        self.ln2 = nn.Linear(4 * config.n_emb, config.n_emb)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True))
        self.norm = nn.LayerNorm(config.n_emb)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        x = self.ln1(x)
        x = torch.nn.functional.gelu(x)
        x = self.ln2(x)
        return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.layernorm1 = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(config)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)   
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.layernorm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.layernorm2(x), shift_mlp, scale_mlp))
        return x

class Feedforward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.Linear(config.n_emb, config.n_emb * 4)
        self.ln2 = nn.Linear(config.n_emb * 4, config.n_emb)
        self.gelu = nn.GELU() 
        self.dropout = nn.Dropout(0.1) 

    def forward(self, x):
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.ln2(x)
        return x