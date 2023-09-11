import torch
import torch.nn as nn

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

class SpiralConvConvBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.phazor = nn.Parameter(torch.randn(dim, dtype=torch.cfloat)) # log(-log(gamma))
        self.phazor_init = nn.Parameter(torch.randn(dim, dtype=torch.cfloat)) # log(-log(gamma))
        self.last_conv = None # (batch, dim)
        self.last_conv_init = nn.Parameter(torch.randn(dim, dtype=torch.cfloat)) # (dim)
        self.is_refresh = True

    # (batch, len, dim) -> (batch, len, dim)
    def forward(self, x):
        batch = x.shape[0]
        len = x.shape[1]
        if self.last_conv is None:
            self.last_conv = self.last_conv_init.expand(batch, self.dim) 
        phazor = self.phazor / self.phazor.abs() * torch.exp(-self.phazor.abs())
        phazor_progression = torch.pow(phazor.unsqueeze(0), torch.arange(len, device=x.device).unsqueeze(1)) # (len, dim)
        filter = phazor_progression * self.phazor_init.unsqueeze(0)
        filter_fft = torch.fft.fft(filter, n=len*2, dim=0) # (len*2, dim)
        x_fft = torch.fft.fft(x, n=len*2, dim=1) # (batch, len*2, dim)
        conv_filter_x = torch.fft.ifft(filter_fft.unsqueeze(0) * x_fft, dim=1).narrow(1,0,len) # (batch, len, dim)
        conv_with_past = conv_filter_x + self.last_conv.detach().unsqueeze(1)*phazor_progression.unsqueeze(0)*phazor.unsqueeze(0).unsqueeze(0)
        if self.is_refresh:
            self.last_conv = conv_with_past[:,-1,:]
        
        return conv_with_past.real

    def reset_hidden(self):
        self.last_conv = None

    def randomize_init(self):
        self.last_conv_init.value = torch.randn(self.dim, dtype=torch.cfloat)

    def set_is_refresh(self, is_refresh):
        self.is_refresh = is_refresh

class SpiralConvBlock(nn.Module):
    def __init__(self, dim: int, dim_ff_scale: float, dropout: float):
        super().__init__()
        self.spiral_conv = SpiralConvConvBlock(dim)
        self.ffn = FFN(dim, dim_ff_scale, dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x_ = x
        x = self.layer_norm(x)
        x = self.spiral_conv(x)
        x = x + x_

        x_ = x
        x = self.layer_norm(x)
        x = self.ffn(x)
        x = x + x_

        return x

    def reset_hidden(self):
        self.spiral_conv.reset_hidden()

    def randomize_init(self):
        self.spiral_conv.randomize_init()

    def set_is_refresh(self, is_refresh):
        self.spiral_conv.set_is_refresh(is_refresh)

class SpiralConv(nn.Module):
    def __init__(self, depth: int, dim: int, dim_ff_scale: float, dropout: float):
        super().__init__()
        self.block_list = nn.ModuleList([SpiralConvBlock(dim, dim_ff_scale, dropout) for _ in range(depth)])

    def forward(self, x):
        for block in self.block_list:
            x = block(x)
        return x 

    def reset_hidden(self):
        for block in self.block_list:
            block.reset_hidden()

    def randomize_init(self):
        for block in self.block_list:
            block.randomize_init()

    def set_is_refresh(self, is_refresh):
        for block in self.block_list:
            block.set_is_refresh(is_refresh)
    