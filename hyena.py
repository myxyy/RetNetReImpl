import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import copy
import math
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

class HyenaBaseBlock(nn.Module):
    def __init__(self, len_in: int, len_out: int, dim_in: int, dim_out: int, dim_pos: int, dim_ff_scale: float, dropout: float, positional_encoding=None):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        if dim_in != dim_out:
            self.linear_in = nn.Linear(dim_in, dim_out, bias=False)
        if positional_encoding is None:
            positional_encoding = PositionalEncoding(len_in, dim_pos)
        self.pos = positional_encoding
        self.pos_linear_1 = nn.Linear(dim_pos, dim_pos * dim_ff_scale)
        self.pos_linear_2 = nn.Linear(dim_pos * dim_ff_scale, dim_out)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
        self.a = nn.Parameter(torch.randn(1))
        self.len_in = len_in
        self.len_out = len_out
        self.ffn = FFN(dim_out, dim_ff_scale, dropout)
        self.ffn_x = FFN(dim_out, dim_ff_scale, dropout)
        self.layer_norm = nn.LayerNorm(dim_out)
    def forward(self, z, x):
        if self.dim_in != self.dim_out:
            z = self.linear_in(z)
        fz = fft.rfft(self.layer_norm(z),n=self.len_in*2,dim=1) # (batch, ?, dim_out)
        h = self.dropout(self.pos_linear_2(self.act(self.pos_linear_1(self.pos())))) # (len_in, dim_out)
        expa = torch.exp(self.a)
        window = torch.exp(-torch.arange(self.len_in, device='cuda')*expa) # (len_in)
        wfh = fft.rfft(window.unsqueeze(-1)*h,n=self.len_in*2,dim=0) # (?, dim_out)
        cwhz = fft.irfft(wfh*fz,dim=1).narrow(1,0,self.len_in) # (batch, len_in, dim_out)
        if self.len_in != self.len_out:
            cwhz = F.interpolate(cwhz.transpose(-2,-1), self.len_out).transpose(-2,-1) # (batch, len_out, dim_out)
        cwhz = cwhz + z
        y = self.ffn_x(x) * self.layer_norm(cwhz)
        yn = self.layer_norm(y)
        y = self.ffn(yn)+y
        return y
 
class HyenaBlock(HyenaBaseBlock):
    def __init__(self, len: int, dim: int, dim_pos: int, dim_ff_scale: float, dropout: float, positional_encoding=None):
        super().__init__(len, len, dim, dim, dim_pos, dim_ff_scale, dropout, positional_encoding=positional_encoding)

class Hyena(nn.Module):
    def __init__(self, len: int, dim: int, depth: int, dim_pos: int, dim_ff_scale: float, dropout: float, positional_encoding=None):
        super().__init__()
        if positional_encoding is None:
            positional_encoding = PositionalEncoding(len, dim_pos)
        block = HyenaBlock(len, dim, dim_pos, dim_ff_scale, dropout, positional_encoding=positional_encoding)
        self.block_list = nn.ModuleList([copy.deepcopy(block) for _ in range(depth)])
    # (batch, len, dim) -> (batch, len, dim)
    def forward(self, x):
        z = x
        for block in self.block_list:
            z = block(z, x)
        return z

class HyenaCross(nn.Module):
    def __init__(self, len_in: int, len_out: int, dim_in: int, dim_out: int, depth: int, dim_pos: int, dim_ff_scale: float, dropout: float, positional_encoding=None):
        super().__init__()
        if positional_encoding is None:
            positional_encoding = PositionalEncoding(len_in, dim_pos)
        block = HyenaBaseBlock(len_in, len_out, dim_in, dim_out, dim_pos, dim_ff_scale, dropout, positional_encoding=positional_encoding)
        self.block_list = nn.ModuleList([copy.deepcopy(block) for _ in range(depth)])
    def forward(self, z, x):
        for block in self.block_list:
            x = block(z, x)
        return x

class HyenaUet(nn.Module):
    def __init__(self, len: int, downsample_rate: float, depth_unet: int, depth_hyena: int, dim: int, dim_scale: float, dim_pos: int, dim_ff_scale: float, dropout: int, enable_pre=True, enable_middle=True, enable_post=True):
        super().__init__()
        self.depth_unet = depth_unet
        self.enable_pre = enable_pre
        self.enable_middle = enable_middle
        self.enable_post = enable_post
        def level_i_dim(i):
            return (int)(math.ceil(dim*(dim_scale**i)/2)*2)
        def level_i_len(i):
            return (int)(len*(downsample_rate**i))
        self.positional_encoding_hyena_list = nn.ModuleList([PositionalEncoding(level_i_len(i),dim_pos,requires_grad=True) for i in range(depth_unet+1)])
        self.positional_encoding_in_list = nn.ModuleList([PositionalEncoding(level_i_len(i),level_i_dim(i),requires_grad=True) for i in range(depth_unet+1)])
        self.encoder_list = nn.ModuleList([HyenaCross(level_i_len(i),level_i_len(i+1),level_i_dim(i),level_i_dim(i+1),depth_hyena,dim_pos,dim_ff_scale,dropout,self.positional_encoding_hyena_list[i]) for i in range(depth_unet)])
        self.decoder_list = nn.ModuleList([HyenaCross(level_i_len(i+1),level_i_len(i),level_i_dim(i+1),level_i_dim(i),depth_hyena,dim_pos,dim_ff_scale,dropout,self.positional_encoding_hyena_list[i+1]) for i in range(depth_unet)])
        if enable_pre:
            self.hyena_pre_list = nn.ModuleList([Hyena(level_i_len(i),level_i_dim(i),depth_hyena,dim_pos,dim_ff_scale,dropout,self.positional_encoding_hyena_list[i]) for i in range(depth_unet+1)])
        if enable_middle:
            self.hyena_middle_list = nn.ModuleList([Hyena(level_i_len(i),level_i_dim(i),depth_hyena,dim_pos,dim_ff_scale,dropout,self.positional_encoding_hyena_list[i]) for i in range(depth_unet+1)])
        if enable_post:
            self.hyena_post_list = nn.ModuleList([Hyena(level_i_len(i),level_i_dim(i),depth_hyena,dim_pos,dim_ff_scale,dropout,self.positional_encoding_hyena_list[i]) for i in range(depth_unet+1)])
    def unet_rec(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        batch = x.shape[0]
        if self.enable_pre:
            x = self.hyena_pre_list[depth](x)
        if depth < self.depth_unet:
            y = self.encoder_list[depth](x, self.positional_encoding_in_list[depth+1]().repeat(batch,1,1))
            y = self.unet_rec(y, depth+1)
        if self.enable_middle:
            x = self.hyena_middle_list[depth](x)
        if depth < self.depth_unet:
            x = self.decoder_list[depth](y, x)
        if self.enable_post:
            x = self.hyena_post_list[depth](x)
        return x
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet_rec(x, 0)
        

class ModHyenaBlock(nn.Module):
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
    def forward(self, z):
        fz = fft.rfft(self.layer_norm(z),n=self.len*2,dim=1) # (batch, ?, dim_out)
        h = self.ffn_pos(self.pos()) # (len_in, dim_out)
        expa = torch.exp(self.a)
        window = torch.exp(-torch.arange(self.len, device='cuda')*expa) # (len_in)
        wfh = fft.rfft(window.unsqueeze(-1)*h,n=self.len*2,dim=0) # (?, dim_out)
        z = fft.irfft(wfh*fz,dim=1).narrow(1,0,self.len) + z # (batch, len_in, dim_out)
        z = self.ffn(self.layer_norm(z))+z
        return z
 
class ModHyena(nn.Module):
    def __init__(self, len: int, dim: int, depth: int, dim_ff_scale: float, dropout: float, positional_encoding=None):
        super().__init__()
        if positional_encoding is None:
            positional_encoding = PositionalEncoding(len, dim)
        block = ModHyenaBlock(len, dim, dim_ff_scale, dropout, positional_encoding=positional_encoding)
        self.block_list = nn.ModuleList([copy.deepcopy(block) for _ in range(depth)])
    # (batch, len, dim) -> (batch, len, dim)
    def forward(self, z):
        for block in self.block_list:
            z = block(z)
        return z

