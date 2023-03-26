import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import copy
from positional_encoding import PositionalEncoding

class ResidualLayerNormFFN(nn.Module):
    def __init__(self, dim: int, dim_ff_scale: float, dropout: float):
        self.layer_norm = nn.LayerNorm(dim)
        self.linear_1 = nn.Linear(dim, dim*dim_ff_scale, bias=True)
        self.linear_2 = nn.Linear(dim*dim_ff_scale*2, dim, bias=True)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        y = self.layer_norm(x)
        y = self.linear_1(y)
        y = self.act(y)
        y = self.linear_2(y)
        x = y + x
        x = self.dropout(x)
        return x

class HyenaBlock(nn.Module):
    def __init__(self, len: int, dim: int, dim_pos: int, dropout: float):
        super().__init__()
        self.len = len
        self.dim = dim
        self.a = nn.Parameter(torch.randn(1))
        self.pos = PositionalEncoding(len, dim_pos)
        self.act = nn.SiLU()
        self.linear_pos_1 = nn.Linear(dim_pos, dim_pos+dim, bias=True)
        self.linear_pos_2 = nn.Linear(dim_pos+dim, dim, bias=True)
        self.rlnffn_1 = ResidualLayerNormFFN(dim, 2, dropout)
        self.rlnffn_2 = ResidualLayerNormFFN(dim, 2, dropout)
        self.layer_norm = nn.LayerNorm(dim)
    # (batch, len, dim), (batch, len, dim) -> (batch, len, dim)
    def forward(self, z, x):
        zn = self.layer_norm(z)
        fz = fft.rfft(zn,n=self.len*2,dim=1) # (batch, len*2, dim)
        expa = torch.exp(self.a)
        window = torch.exp(-torch.arange(self.len, device='cuda')*expa) # (len)
        h = self.linear_pos_1(self.pos.pe) # (len, dim)
        h = self.act(h)
        h = self.linear_pos_2(h) # (len, dim)
        h = self.dropout(h)
        wfh = fft.rfft(window.unsqueeze(-1)*h,n=self.len*2,dim=0) # (len*2, dim)
        wfhfz = wfh*fz
        cwhz = fft.irfft(wfhfz,dim=1).narrow(1,0,self.len)
        x = self.rlnffn_1(x)
        y = x*cwhz
        y = self.rlnffn_2(y)
        return y

class Hyena(nn.Module):
    def __init__(self, len: int, dim: int, depth: int, dim_pos: int, dropout: float):
        super().__init__()
        block = HyenaBlock(len, dim, dim_pos, dropout)
        self.block_list = nn.ModuleList([copy.deepcopy(block) for _ in range(depth)])
        self.linear_1 = nn.Linear(dim, dim, bias=True)
        self.act = nn.act()
        self.linear_2 = nn.Linear(dim, dim, bias=True)
    # (batch, len, dim) -> (batch, len, dim)
    def forward(self, v):
        z = self.linear_1(v)
        z = self.act(z)
        z = self.linear_2(v)
        for block in self.block_list:
            z = block(z, v)
        return z

class HyenaEncoderBlock(nn.Module):
    def __init__(self, len_in: int, len_out: int, dim_in: int, dim_out: int, dim_pos: int, dropout: float):
        super().__init__()
        self.linear_in = nn.Linear(dim_in, dim_out, bias=True)
        self.pos = PositionalEncoding(len_in, dim_pos)
        self.linear_pos_1 = nn.Linear(dim_pos, dim_pos+dim_out, bias=True)
        self.linear_pos_2 = nn.Linear(dim_pos+dim_out, dim_out, bias=True)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.a = nn.Parameter(torch.randn(1))
        self.layer_norm = nn.LayerNorm(dim_out)
        self.len_in = len_in
        self.len_out = len_out
        self.rlnffn_1 = ResidualLayerNormFFN(dim_out, 2, dropout)
        self.rlnffn_2 = ResidualLayerNormFFN(dim_out, 2, dropout)
    def forward(self, z, x):
        z = self.linear_in(z)
        zn = self.layer_norm(z)
        fz = fft.rfft(zn,n=self.len_in*2,dim=1) # (batch, len_in*2, dim_out)
        h = self.linear_pos_1(self.pos.pe)
        h = self.act(h)
        h = self.linear_pos_2(h)
        h = self.dropout(h)
        expa = torch.exp(self.a)
        window = torch.exp(-torch.arange(self.len_in, device='cuda')*expa) # (len_in)
        wfh = fft.rfft(window.unsqueeze(-1)*h,n=self.len_in*2,dim=0) # (len_in*2, dim_out)
        wfhfz = wfh*fz
        cwhz = fft.irfft(wfhfz,dim=1).narrow(1,0,self.len_in) # (batch, len_in, dim_out)
        dcwhz = F.interpolate(cwhz.transpose(-2,-1), self.len_out).transpose(-2,-1) # (batch, len_out, dim_out)
        x = self.rlnffn_1(x)
        y = x * dcwhz
        y = self.rlnffn_2(y)
        return y
 