import torch 
import torch.nn as nn
import math

#Label and timestep
# From https://github.com/facebookresearch/DiT/blob/main/models.py

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = MLP()
        self.frequency_embedding_size = frequency_embedding_size
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
# From https://github.com/young-geng/m3ae_public/blob/master/m3ae/model.py

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    grid = torch.arange(length, dtype=torch.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed.unsqueeze(0)

def get_2d_sincos_pos_embed(embed_dim, length,interpolation_scale=1.0,base_size=16,device: Optional[torch.device] = None):
    # example: embed_dim = 256, length = 16*16
    grid_size = int(length**0.5)
    assert grid_size * grid_size == length
    grid_h = (
        torch.arange(grid_size[0], device=device, dtype=torch.float32)
        / (grid_size[0] / base_size)
        / interpolation_scale
    )
    grid_w = (
        torch.arange(grid_size[1], device=device, dtype=torch.float32)
        / (grid_size[1] / base_size)
        / interpolation_scale
    )
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # here w goes first
    grid = torch.stack(grid, dim=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = torch.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb
    


#Multi head attention 
class MultiHeadAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert (config.n_emb % config.num_heads==0), "n_emb must be divisible by num_heads"
        self.head_dim  = config.n_emb//config.num_heads
        self.num_heads = config.num_heads
        self.n_emb = config.n_emb
        self.dropout = nn.Dropout()
        self.q = nn.Linear(config.n_emb,config.n_emb,bias=False)
        self.k = nn.Linear(config.n_emb,config.n_emb,bias=False)
        self.v = nn.Linear(config.n_emb,config.n_emb,bias=False)
        # scale = config.n_emb ** -0.5
        self.proj = nn.Linear(config.n_emb,config.n_emb)
    def forward(self,x):
        B,num_tokens,d_in = x.shape
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        #change the dimension of QKV
        q = self.q(x).view(B, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, N, head_dim)
        k = self.k(x).view(B, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        #transpose
        query = query.transpose(1,2)
        key = key.transpose(1,2)
        key = query.transpose(1,2)

        #calculate attention score
        attention_score = query @ key.transpose(2,3)
        mask = self.mask.bool()[:num_tokens,:num_tokens]
        attention_score.masked_fill_(mask,-torch.inf)

        attention_weights = torch.softmax(attention_score/key.shape[-1]**0.5,dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector = (attention_weights @ value).transpose(1,2)
        
        context_vector = context_vector.contiguous().view(B,num_tokens,self.n_emb)
        context_vector = self.proj(context_vector)

        return context_vector
    


# Patch embedding
class PatchEmbed():
    def __init__(self,config, in_channels=4, img_size: int=32, dim=1024, patch_size: int = 2):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        patch_tuple = (patch_size, patch_size)
        self.num_patches = (img_size // self.patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, config.n_emb, kernel_size=(patch_size, patch_size), stride=patch_size, bias=False
        )

    def forward(self, x):
        B, H, W, C = x.shape
        num_patches_side = (H // self.patch_size)
        x = self.conv_project(x) # (B, P, P, hidden_size)
        x = x.view(num_patches_side, num_patches_side)
        return x



#Multi layer perceptron
class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.silu = nn.SiLU()
        self.ln1 = nn.Linear(config.n_emb,4 * config.n_emb)
        self.ln2 = nn.Linear(4 * config.n_emb,config.n_emb)
        self.dropout = nn.Dropout()
    def forward(self,x):
        x = self.ln1(x)
        x = self.silu(x)
        x = self.dropout(x)
        x = self.ln2(x)
        x = self.dropout(x)
        return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBLock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.layernorm1 = nn.LayerNorm(config.hidden_size,elementwise_affine=False,eps=1e-6)
        self.layernomr2 = nn.LayerNorm(config.hidden_size,elementwise_affine=False,eps=1e-6)
        self.mlp = MLP()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True)
        )
    def forward(self,x,c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)   
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class DiT(nn.Module):
    def __init__(self,config):
        super().__init__()
