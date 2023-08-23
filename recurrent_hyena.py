import torch
import torch.nn as nn
import torch.fft as fft
from positional_encoding import PositionalEncoding

class FFN(nn.Module):
    def __init__(self, dim: int, dim_ff_scale: float, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim*dim_ff_scale, bias=True)
        self.linear_2 = nn.Linear(dim*dim_ff_scale, dim, bias=True)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

class RecurrentHyenaBlock(nn.Module):
    def __init__(self, len: int, dim: int, dim_ff_scale: float, dropout: float, positional_encoding=None):
        super().__init__()
        self.dim = dim
        if positional_encoding is None:
            positional_encoding = PositionalEncoding(len, dim)
        self.pos = positional_encoding
        self.ffn_pos = FFN(dim, dim_ff_scale, dropout)
        self.a = nn.Parameter(torch.randn(1))
        self.len = len
        self.ffn = FFN(dim, dim_ff_scale, dropout)
        self.layer_norm = nn.LayerNorm(dim)
        self.hidden = None
        self.hidden_init = nn.Parameter(torch.randn(len, dim))
        self.is_refresh = True

    # (batch, len, dim) -> (batch, len, dim)
    def forward(self, z):
        batch = z.shape[0]
        if self.hidden is None:
            self.hidden = self.hidden_init.unsqueeze(0).expand(batch, self.len, self.dim)
        fz = fft.rfft(self.layer_norm(z),n=self.len*2,dim=1) # (batch, ?, dim)
        h = self.ffn_pos(self.pos()) # (len, dim)
        expa = torch.exp(self.a)
        window = torch.exp(-torch.arange(self.len, device=z.device)*expa) # (len)
        wfh = fft.rfft(window.unsqueeze(-1)*h,n=self.len*2,dim=0) # (?, dim)
        z_hidden = fft.irfft(wfh.unsqueeze(0)*fz,dim=1) # (batch, len, dim)
        z = z_hidden.narrow(1,0,self.len) + self.hidden.detach() #+ z # (batch, len, dim)
        if self.is_refresh:
            self.hidden = z_hidden.narrow(1,self.len,self.len)
        z = self.ffn(self.layer_norm(z))+z
        return z
 
    def clear_hidden(self):
        self.last_hidden = None

    def set_is_refresh(self, is_refresh):
        self.is_refresh = is_refresh

class RecurrentHyena(nn.Module):
    def __init__(self, len: int, depth: int, dim: int, dim_ff_scale: float, dropout: float):
        super().__init__()
        self.block_list = nn.ModuleList([RecurrentHyenaBlock(len, dim, dim_ff_scale, dropout) for _ in range(depth)])

    def forward(self, x):
        for block in self.block_list:
            x = block(x)
        return x 

    def clear_hidden(self):
        for block in self.block_list:
            block.clear_hidden()

    def set_is_refresh(self, is_refresh):
        for block in self.block_list:
            block.set_is_refresh(is_refresh)
 