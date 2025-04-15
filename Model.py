import torch 
import torch.nn as nn
#Multi head attention 
class MultiHead(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert (config.n_emb % num_heads==0), "n_emb must be divisible by num_heads"
        self.num_heads = config.num_heads
        self.n_emb = config.n_emb
        self