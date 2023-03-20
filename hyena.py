import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import copy
from positional_encoding import PositionalEncoding

class HyenaBlock(nn.Module):
    def __init__(self, len: int, dim: int, dim_pos: int, dropout: float):
        super().__init__()
        self.len = len
        self.dim = dim
        self.a = nn.Parameter(torch.randn(1))
        self.pos = PositionalEncoding(len, dim_pos)
        self.linear_pos = nn.Linear(dim_pos, dim, bias=False)
        self.linear_1 = nn.Linear(dim, dim*2, bias=False)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(dim*2, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)
    # (batch, len, dim), (batch, len, dim) -> (batch, len, dim)
    def forward(self, z, x):
        z = self.layer_norm(z)
        fz = fft.rfft(z,n=self.len*2,dim=1) # (batch, len*2, dim)
        expa = torch.exp(self.a)
        window = torch.exp(-torch.arange(self.len, device='cuda')*expa) # (len)
        h = self.linear_pos(self.pos.pe) # (len, dim)
        wfh = fft.rfft(window.unsqueeze(-1)*h,n=self.len*2,dim=0) # (len*2, dim)
        wfhfz = wfh*fz
        cwhz = fft.irfft(wfhfz,dim=1).narrow(1,0,self.len)
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x*cwhz + z

class Hyena(nn.Module):
    def __init__(self, len: int, dim: int, depth: int, dim_pos: int, dropout: float):
        super().__init__()
        block = HyenaBlock(len, dim, dim_pos, dropout)
        self.block_list = nn.ModuleList([copy.deepcopy(block) for _ in range(depth)])
        self.linear = nn.Linear(dim, dim, bias=False)
    # (batch, len, dim) -> (batch, len, dim)
    def forward(self, v):
        z = self.linear(v)
        for block in self.block_list:
            z = block(z, v)
        return z

