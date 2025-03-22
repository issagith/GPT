import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
n_embd = 32
block_size = 8

class Head(nn.Module):
    """One head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # tril isn't a param of the model, it's called a buffer and we initialize it using register_buffer like so
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute "affinities" called attention score
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) --> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        # perform the weighted aggregation
        v = self.value(x)
        out = wei @ v # (B,T,T) @ (B,T,C) --> (B,T,C)
        return out 

