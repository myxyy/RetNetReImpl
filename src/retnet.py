import torch
import torch.nn as nn
import numpy as np

class FFN(nn.Module):
    def __init__(self, dim: int, dim_hidden: float, dtype):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim_hidden, dtype=dtype)
        self.linear_2 = nn.Linear(dim_hidden, dim, dtype=dtype)
        self.act = nn.SiLU()
    def forward(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x

class Retention(nn.Module):
    def __init__(self, dim: int, dim_qkv: int, num_head: int, dtype, out_weight=False):
        super().__init__()
        self.dim = dim
        self.dim_qkv = dim_qkv
        self.num_head = num_head
        self.phazor = nn.Parameter(torch.randn((num_head, dim_qkv), dtype=torch.cfloat)) # log(-log(gamma))
        self.amplitude = nn.Parameter(torch.randn(num_head))
        #self.phazor_init = nn.Parameter(torch.randn((num_head, dim_qkv, dim_qkv), dtype=torch.cfloat)) # log(-log(gamma))
        self.last_conv = None # (batch, num_head, dim_qkv, dim_qkv)
        self.last_conv_init = nn.Parameter(torch.randn((num_head, dim_qkv, dim_qkv), dtype=torch.cfloat)) # (num_head, dim_qkv, dim_qkv)
        self.wq = nn.Linear(dim, num_head * dim_qkv, dtype=dtype)
        self.wk = nn.Linear(dim, num_head * dim_qkv, dtype=dtype)
        self.wv = nn.Linear(dim, num_head * dim_qkv, dtype=dtype)
        self.act = nn.SiLU()
        self.out_weight = out_weight
        if out_weight:
            self.wout = nn.Linear(num_head * dim_qkv, dim, dtype=dtype)
        else:
            assert dim == dim_qkv * num_head, "Retentin dim_qkv * num_head must be dim if out_weight==False"
        self.is_refresh = True
        self.sigmoid = nn.Sigmoid()

    # (batch, len, dim) -> (batch, len, dim)
    def forward(self, x):
        batch = x.shape[0]
        len = x.shape[1]
        dtype = x.dtype
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
            self.last_conv = self.last_conv.detach() * torch.pow(phazor, len).unsqueeze(0).unsqueeze(-1) + (kv * phazor_progression_inverse.unsqueeze(0).unsqueeze(-1)).sum(1) # (batch, num_head, dim_qkv, dim_qkv)

        mask_mask = torch.ones((len, len), device=x.device).tril()
        mask_exp = (torch.arange(len, device=x.device).unsqueeze(1) - torch.arange(len, device=x.device).unsqueeze(0)) * mask_mask # (len, len)
        mask = torch.pow(amplitude.unsqueeze(-1).unsqueeze(-1), mask_exp.unsqueeze(0)) * mask_mask # (num_head, len, len)

        qk = torch.matmul(
            (query * phase_progression.unsqueeze(0)).permute(0,2,1,3), # (batch, num_head, len, dim_qkv)
            (key * phase_progression_inverse.unsqueeze(0)).permute(0,2,3,1) # (batch, num_head, dim_qkv, len)
        ).view(batch, num_head, len, len) # (batch, num_head, len, len)
        qk_mask = qk * mask.unsqueeze(0).expand(batch, num_head, len, len) # (batch, num_head, len, len)
        inner_chunk = torch.matmul(qk_mask.to(torch.cfloat), value.permute(0,2,1,3).to(torch.cfloat)).permute(0,2,1,3) # (batch, len, num_head, dim_qkv)

        if self.out_weight:
            out = self.wout(self.act((inner_chunk + cross_chunk).real).to(dtype).reshape(batch, len, num_head * dim_qkv))
        else:
            out = (inner_chunk + cross_chunk).real.to(dtype).reshape(batch, len, num_head * dim_qkv)
        #print(f'test:{out.isinf().any()}')
        #print(f'test:{out.isnan().any()}')
        #print(mask_exp)
        #print(amplitude)
        #print(mask[0])
        #print(qk[0,0])
        #print(qk_mask[0,0])
        #print(out[0])
        return out

    def reset_hidden(self):
        self.last_conv = None

    def set_is_refresh(self, is_refresh):
        self.is_refresh = is_refresh

class RetNetBlock(nn.Module):
    def __init__(self, dim: int, dim_hidden: float, dropout: float, dim_qkv: int, num_head: int, dtype):
        super().__init__()
        self.retention = Retention(dim, dim_qkv, num_head, dtype, out_weight=True)
        self.ffn = FFN(dim, dim_hidden, dtype)
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_ = x
        x = self.layer_norm(x)
        x = self.retention(x)
        x = self.dropout(x)
        x = x + x_

        x_ = x
        x = self.layer_norm(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + x_

        return x

    def reset_hidden(self):
        self.retention.reset_hidden()

    def set_is_refresh(self, is_refresh):
        self.retention.set_is_refresh(is_refresh)

class RetNet(nn.Module):
    def __init__(self, depth: int, dim: int, dim_hidden: float, dropout: float, dim_qkv: int, num_head: int, vocab_size: int, devices, dtype=torch.bfloat16):
        super().__init__()
        self.devices = devices
        self.dtype = dtype
        self.token_in = nn.Linear(vocab_size, dim, device=devices[0], dtype=dtype)
        self.token_out = nn.Linear(dim, vocab_size, device=devices[-1], dtype=dtype)
        self.block_list = nn.ModuleList([RetNetBlock(dim, dim_hidden, dropout, dim_qkv, num_head, dtype) for _ in range(depth)])
        for i, block in enumerate(self.block_list):
            block.to(devices[self.device_index(i)])

    def device_index(self, i):
        return (len(self.devices) * i) // len(self.block_list)

    def forward(self, x):
        x = self.token_in(x)
        for i, block in enumerate(self.block_list):
            if i > 0 and self.device_index(i) != self.device_index(i-1):
                x = x.to(self.devices[self.device_index(i)])
            x = block(x)
        x = self.token_out(x)
        return x 

    def reset_hidden(self):
        for block in self.block_list:
            block.reset_hidden()

    def set_is_refresh(self, is_refresh):
        for block in self.block_list:
            block.set_is_refresh(is_refresh)
    
    def module_list(self):
        blistlist = []
        for _ in self.devices:
            blistlist.append([])
        for i, block in enumerate(self.block_list):
            blistlist[self.device_index(i)].append(block)
        mlist = []
        for blist in blistlist:
            mlist.append(nn.Sequential(*blist))
        mlist[0] = nn.Sequential(self.token_in, mlist[0])
        mlist[-1] = nn.Sequential(mlist[-1], self.token_out)
        return mlist
        