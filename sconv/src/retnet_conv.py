import torch
import torch.nn as nn
import numpy as np

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
        self.phazor = nn.Parameter(torch.randn((num_head, dim_qkv), dtype=torch.cfloat)) # log(-log(gamma))
        self.amplitude = nn.Parameter(torch.randn(num_head))
        #self.phazor_init = nn.Parameter(torch.randn((num_head, dim_qkv, dim_qkv), dtype=torch.cfloat)) # log(-log(gamma))
        self.last_conv = None # (batch, num_head, dim_qkv, dim_qkv)
        self.last_conv_init = nn.Parameter(torch.randn((num_head, dim_qkv, dim_qkv), dtype=torch.cfloat)) # (num_head, dim_qkv, dim_qkv)
        self.wq = nn.Linear(dim, num_head * dim_qkv)
        self.wk = nn.Linear(dim, num_head * dim_qkv)
        self.wv = nn.Linear(dim, num_head * dim_qkv)
        self.wout = nn.Linear(num_head * dim_qkv, dim)
        self.is_refresh = True
        self.sigmoid = nn.Sigmoid()

    # (batch, len, dim) -> (batch, len, dim)
    def forward(self, x):
        batch = x.shape[0]
        len = x.shape[1]
        num_head = self.num_head
        dim_qkv = self.dim_qkv

        query = self.wq(x).view(batch, len, self.num_head, self.dim_qkv) # (batch, len, num_head, dim_qkv)
        key = self.wk(x).view(batch, len, self.num_head, self.dim_qkv) # (batch, len, num_head, dim_qkv)
        value = self.wv(x).view(batch, len, self.num_head, self.dim_qkv) # (batch, len, num_head, dim_qkv)

        kv = torch.matmul(key.unsqueeze(4), value.unsqueeze(3)) # (batch, len, num_head, dim_qkv, dim_qkv)

        if self.last_conv is None:
            self.last_conv = self.last_conv_init.unsqueeze(0).expand(batch, self.num_head, self.dim_qkv, self.dim_qkv) 

        phase = self.phazor / self.phazor.abs()
        amplitude = self.sigmoid(self.amplitude) # (num_head,)
        phazor = phase * amplitude.unsqueeze(-1).expand(num_head, dim_qkv) # (num_head, dim_qkv)
        phase_progression = torch.pow(phase.unsqueeze(0), (torch.arange(len, device=x.device)).unsqueeze(1).unsqueeze(1)).expand(len, num_head, dim_qkv) # (len, num_head, dim_qkv)
        phase_progression_inverse = torch.pow(phase.unsqueeze(0), (-torch.arange(len, device=x.device)).unsqueeze(1).unsqueeze(1)).expand(len, num_head, dim_qkv) # (len, num_head, dim_qkv)
        phazor_progression = torch.pow(phazor.unsqueeze(0), (torch.arange(len, device=x.device)).unsqueeze(1).unsqueeze(1)).expand(len, num_head, dim_qkv) # (len, num_head, dim_qkv)
        phazor_progression_inverse = torch.pow(phazor.unsqueeze(0), (len - 1 - torch.arange(len, device=x.device)).unsqueeze(1).unsqueeze(1)).expand(len, num_head, dim_qkv) # (len, num_head, dim_qkv)

        cross_chunk = torch.matmul(query.unsqueeze(3) * (phazor_progression * phazor.unsqueeze(0)).unsqueeze(0).unsqueeze(3), self.last_conv.detach().unsqueeze(1)).view(batch, len, num_head, dim_qkv) # (batch, len, num_head, dim_qkv)
        if self.is_refresh:
            self.last_conv = self.last_conv.detach() * torch.pow(phazor, len).unsqueeze(0).unsqueeze(-1) + (kv * phazor_progression_inverse.unsqueeze(0).unsqueeze(-1)).sum(1)

        mask_mask = torch.full((len, len), np.inf, device=x.device).triu(1) # (len, len)
        #mask_mask_2 = torch.ones((len, len), device=x.device).tril()
        mask_exp = (torch.arange(len, device=x.device).unsqueeze(1) - torch.arange(len, device=x.device).unsqueeze(0)) + mask_mask # (len, len)
        mask = torch.pow(amplitude.unsqueeze(-1).unsqueeze(-1), mask_exp.unsqueeze(0)).detach() # (num_head, len, len) # 勾配計算するとnanになりよくわからず
        qk = torch.matmul(
            (query * phase_progression.unsqueeze(0)).permute(0,2,1,3), # (batch, num_head, len, dim_qkv)
            (key * phase_progression_inverse.unsqueeze(0)).permute(0,2,3,1) # (batch, num_head, dim_qkv, len)
        ).view(batch, num_head, len, len) # (batch, num_head, len, len)
        qk_mask = qk * mask.unsqueeze(0).expand(batch, num_head, len, len) # (batch, num_head, len, len)
        inner_chunk = torch.matmul(qk_mask, value.permute(0,2,1,3).to(torch.cfloat)).permute(0,2,1,3) # (batch, len, num_head, dim_qkv)

        out = self.wout((inner_chunk + cross_chunk).real.reshape(batch, len, num_head * dim_qkv))
        #print(f'test:{out.isinf().any()}')
        #print(f'test:{out.isnan().any()}')
        #print(amplitude)
        #print(mask[7])
        #print(out[0])
        return out

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
    