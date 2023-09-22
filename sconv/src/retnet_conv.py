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

class RetentionConv(nn.Module):
    def __init__(self, dim: int, dim_qkv: int, num_head: int):
        super().__init__()
        self.dim = dim
        self.dim_qkv = dim_qkv
        self.num_head = num_head
        self.phazor = nn.Parameter(torch.randn((num_head, dim_qkv, dim_qkv), dtype=torch.cfloat)) # log(-log(gamma))
        self.phazor_init = nn.Parameter(torch.randn((num_head, dim_qkv, dim_qkv), dtype=torch.cfloat)) # log(-log(gamma))
        self.last_conv = None # (batch, dim)
        self.last_conv_init = nn.Parameter(torch.randn((num_head, dim_qkv, dim_qkv), dtype=torch.cfloat)) # (dim)
        self.wq = nn.Linear(dim, num_head * dim_qkv)
        self.wk = nn.Linear(dim, num_head * dim_qkv)
        self.wv = nn.Linear(dim, num_head * dim_qkv)
        self.wout = nn.Linear(num_head * dim_qkv, dim)
        self.is_refresh = True

    # (batch, len, dim) -> (batch, len, dim)
    def forward(self, x):
        batch = x.shape[0]
        len = x.shape[1]
        query = self.wq(x).view(batch, len, self.num_head, self.dim_qkv) # (batch, len, num_head, dim_qkv)
        key = self.wk(x).view(batch, len, self.num_head, self.dim_qkv) # (batch, len, num_head, dim_qkv)
        value = self.wv(x).view(batch, len, self.num_head, self.dim_qkv) # (batch, len, num_head, dim_qkv)
        kv = torch.matmul(key.unsqueeze(4), value.unsqueeze(3)) # (batch, len, num_head, dim_qkv, dim_qkv)
        if self.last_conv is None:
            self.last_conv = self.last_conv_init.unsqueeze(0).expand(batch, self.num_head, self.dim_qkv, self.dim_qkv) 
        phazor = self.phazor / self.phazor.abs() * torch.exp(-self.phazor.abs())
        phazor_progression = torch.pow(phazor.unsqueeze(0), torch.arange(len, device=x.device).unsqueeze(1).unsqueeze(1).unsqueeze(1)) # (len, num_head, dim_qkv, dim_qkv)
        filter = phazor_progression * self.phazor_init.unsqueeze(0) # (len, num_head, dim_qkv, dim_qkv)
        filter_fft = torch.fft.fft(filter, n=len*2, dim=0) # (len*2, num_head, dim_qkv, dim_qkv)
        kv_fft = torch.fft.fft(kv, n=len*2, dim=1) # (batch, len*2, num_head, dim_qkv, dim_qkv)
        conv_filter_kv = torch.fft.ifft(filter_fft.unsqueeze(0) * kv_fft, dim=1).narrow(1,0,len) # (batch, len, num_head, dim_qkv, dim_qkv)
        conv_with_past = conv_filter_kv + self.last_conv.detach().unsqueeze(1)*phazor_progression.unsqueeze(0)*phazor.unsqueeze(0).unsqueeze(0)
        if self.is_refresh:
            self.last_conv = conv_with_past[:,-1,:,:,:]
        
        return self.wout(torch.matmul(query.unsqueeze(3), conv_with_past.real).view(batch, len, self.num_head * self.dim_qkv))

    def reset_hidden(self):
        self.last_conv = None

    def randomize_init(self):
        self.last_conv_init.value = torch.randn(self.dim, dtype=torch.cfloat)

    def set_is_refresh(self, is_refresh):
        self.is_refresh = is_refresh

class RetentionConvBlock(nn.Module):
    def __init__(self, dim: int, dim_ff_scale: float, dropout: float, dim_qkv: int, num_head: int):
        super().__init__()
        self.retention_conv = RetentionConv(dim, dim_qkv, num_head)
        self.ffn = FFN(dim, dim_ff_scale, dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x_ = x
        x = self.layer_norm(x)
        x = self.retention_conv(x)
        x = x + x_

        x_ = x
        x = self.layer_norm(x)
        x = self.ffn(x)
        x = x + x_

        return x

    def reset_hidden(self):
        self.retention_conv.reset_hidden()

    def randomize_init(self):
        self.retention_conv.randomize_init()

    def set_is_refresh(self, is_refresh):
        self.retention_conv.set_is_refresh(is_refresh)

class RetNetConv(nn.Module):
    def __init__(self, depth: int, dim: int, dim_ff_scale: float, dropout: float, dim_qkv: int, num_head: int):
        super().__init__()
        self.block_list = nn.ModuleList([RetentionConvBlock(dim, dim_ff_scale, dropout, dim_qkv, num_head) for _ in range(depth)])

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
    