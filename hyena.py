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
        self.gelu = nn.GELU()
        self.linear_pos_1 = nn.Linear(dim_pos, dim_pos+dim, bias=True)
        self.linear_pos_2 = nn.Linear(dim_pos+dim, dim, bias=True)
        self.linear_1 = nn.Linear(dim, dim*2, bias=True)
        self.linear_2 = nn.Linear(dim*2, dim, bias=True)
        self.linear_3 = nn.Linear(dim, dim*2, bias=True)
        self.linear_4 = nn.Linear(dim*2, dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)
    # (batch, len, dim), (batch, len, dim) -> (batch, len, dim)
    def forward(self, z, x):
        zn = self.layer_norm(z)
        fz = fft.rfft(zn,n=self.len*2,dim=1) # (batch, len*2, dim)
        expa = torch.exp(self.a)
        window = torch.exp(-torch.arange(self.len, device='cuda')*expa) # (len)
        h = self.linear_pos_1(self.pos.pe) # (len, dim)
        h = self.gelu(h)
        h = self.linear_pos_2(h) # (len, dim)
        h = self.dropout(h)
        wfh = fft.rfft(window.unsqueeze(-1)*h,n=self.len*2,dim=0) # (len*2, dim)
        wfhfz = wfh*fz
        cwhz = fft.irfft(wfhfz,dim=1).narrow(1,0,self.len)
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        x = x*cwhz
        x = x + z
        y = self.layer_norm(x)
        y = self.linear_3(y)
        y = self.gelu(y)
        y = self.linear_4(y)
        return y + x

class Hyena(nn.Module):
    def __init__(self, len: int, dim: int, depth: int, dim_pos: int, dropout: float):
        super().__init__()
        block = HyenaBlock(len, dim, dim_pos, dropout)
        self.block_list = nn.ModuleList([copy.deepcopy(block) for _ in range(depth)])
        self.linear_1 = nn.Linear(dim, dim, bias=True)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(dim, dim, bias=True)
    # (batch, len, dim) -> (batch, len, dim)
    def forward(self, v):
        z = self.linear_1(v)
        z = self.gelu(z)
        z = self.linear_2(v)
        for block in self.block_list:
            z = block(z, v)
        return z

