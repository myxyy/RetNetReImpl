import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import copy
from positional_encoding import PositionalEncoding

class ResidualLayerNormFFN(nn.Module):
    def __init__(self, dim: int, dim_ff_scale: float, dropout: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.linear_1 = nn.Linear(dim, dim*dim_ff_scale, bias=True)
        self.linear_2 = nn.Linear(dim*dim_ff_scale, dim, bias=True)
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

class HyenaBaseBlock(nn.Module):
    def __init__(self, len_in: int, len_out: int, dim_in: int, dim_out: int, dim_pos: int, dropout: float, z_residual=False):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.z_residual = z_residual
        if not (dim_in == dim_out and len_in == len_out) and z_residual:
            assert("z_residual can be True only if dim_in == dim_outand len_in == len_out")
        if (self.dim_in != self.dim_out):
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
        zn = self.layer_norm(z)
        if (self.dim_in != self.dim_out):
            zn = self.linear_in(zn)
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
        if self.len_in == self.len_out:
            dcwhz = cwhz
        else:
            dcwhz = F.interpolate(cwhz.transpose(-2,-1), self.len_out).transpose(-2,-1) # (batch, len_out, dim_out)
        x = self.rlnffn_1(x)
        y = x * dcwhz
        y = self.rlnffn_2(y)
        if self.z_residual:
            y = y + z
        return y
 
class HyenaBlock(HyenaBaseBlock):
    def __init__(self, len: int, dim: int, dim_pos: int, dropout: float):
        super().__init__(len, len, dim, dim, dim_pos, dropout, z_residual=True)

class Hyena(nn.Module):
    def __init__(self, len: int, dim: int, depth: int, dim_pos: int, dropout: float):
        super().__init__()
        block = HyenaBlock(len, dim, dim_pos, dropout)
        self.block_list = nn.ModuleList([copy.deepcopy(block) for _ in range(depth)])
        self.rlnffn = ResidualLayerNormFFN(dim, 2, dropout)
    # (batch, len, dim) -> (batch, len, dim)
    def forward(self, v):
        z = self.rlnffn(v)
        for block in self.block_list:
            z = block(z, v)
        return z
