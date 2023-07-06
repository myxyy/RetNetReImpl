from hyena import HyenaUet, ModHyena

import pytorch_lightning as pl
from timm.models.layers import trunc_normal_
import torchvision.transforms as transforms
import torch
from torchmetrics import MeanMetric
import torch.nn as nn

class Lang(pl.LightningModule):
    def __init__(self, model, len=1024, downsample_rate=0.5, depth_unet=10, depth_hyena=4, dropout=0.1, vocab_size=256, dim=256, dim_scale=1, dim_pos=256, dim_ff_scale=2, batch_size=16, enable_pre=True, enable_middle=True, enable_post=True, enable_profiling=False, text_load_mode='slice'):
        super().__init__()
        self.text_load_mode = text_load_mode
        self.enable_profiling=enable_profiling
        self.len = len
        self.vocab_size = vocab_size
        self.hyena = model(len, downsample_rate, depth_unet, depth_hyena, dim, dim_scale, dim_pos, dim_ff_scale, dropout, enable_pre=enable_pre, enable_middle=enable_middle, enable_post=enable_post)
        self.token_in = nn.Linear(vocab_size, dim)
        self.token_out = nn.Linear(dim, vocab_size)
        self.batch_size = batch_size
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.apply(self._init_weights)
        self.save_hyperparameters()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.num_parameters**-0.5)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def training_step(self, batch, batch_idx):
        data, next = batch
        x = nn.functional.one_hot(data.long(), self.vocab_size).float()
        x_next = next
        if self.enable_profiling:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA
                ]
            ) as p:
                x_hat = self.token_out(self.hyena(self.token_in(x)))
            print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
        else:
            x_hat = self.token_out(self.hyena(self.token_in(x)))
        loss = nn.CrossEntropyLoss()(x_hat.view(-1,self.vocab_size), x_next.view(-1).long())
        self.log("train_loss", loss, on_epoch=False, prog_bar=True)
        return loss

    def forward(self, x):
        x = nn.functional.one_hot(x.long(), self.vocab_size).float()
        x_hat = self.token_out(self.hyena(self.token_in(x)))
        x_hat = x_hat.softmax(2)
        return x_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

class ModHyenaLang(pl.LightningModule):
    def __init__(self, model, len=1024, depth=4, dropout=0.1, vocab_size=256, dim=256, dim_ff_scale=2, batch_size=16, enable_profiling=False, text_load_mode='slice'):
        super().__init__()
        self.text_load_mode = text_load_mode
        self.enable_profiling=enable_profiling
        self.len = len
        self.vocab_size = vocab_size
        self.hyena = model(len, dim, depth, dim_ff_scale, dropout)
        self.token_in = nn.Linear(vocab_size, dim)
        self.token_out = nn.Linear(dim, vocab_size)
        self.batch_size = batch_size
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.apply(self._init_weights)
        self.save_hyperparameters()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.num_parameters**-0.5)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def training_step(self, batch, batch_idx):
        data, next = batch
        x = nn.functional.one_hot(data.long(), self.vocab_size).float()
        x_next = next
        if self.enable_profiling:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA
                ]
            ) as p:
                x_hat = self.token_out(self.hyena(self.token_in(x)))
            print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
        else:
            x_hat = self.token_out(self.hyena(self.token_in(x)))
        loss = nn.CrossEntropyLoss()(x_hat.view(-1,self.vocab_size), x_next.view(-1).long())
        self.log("train_loss", loss, on_epoch=False, prog_bar=True)
        return loss

    def forward(self, x):
        x = nn.functional.one_hot(x.long(), self.vocab_size).float()
        x_hat = self.token_out(self.hyena(self.token_in(x)))
        x_hat = x_hat.softmax(2)
        return x_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

class ModHyenaDownsampleRecurrentLang(pl.LightningModule):
    def __init__(self, model, len=1024, enc_depth=4, dec_depth=4, dropout=0.1, vocab_size=256, dim=256, dim_ff_scale=2, batch_size=16, enable_profiling=False, text_load_mode='slice'):
        super().__init__()
        self.text_load_mode = text_load_mode
        self.enable_profiling=enable_profiling
        self.dim = dim
        self.len = len
        self.vocab_size = vocab_size
        self.enc = model(len*2, dim, enc_depth, dim_ff_scale, dropout)
        self.dec = model(len*2, dim, dec_depth, dim_ff_scale, dropout)
        self.token_in = nn.Linear(vocab_size, dim)
        self.token_out = nn.Linear(dim, vocab_size)
        self.batch_size = batch_size
        self.hidden = None
        self.hidden_init = nn.Parameter(torch.randn(batch_size, len, dim, device='cuda'))
        self.pos_emb_hidden = nn.Parameter(torch.randn(batch_size, len, dim, device='cuda'))
        self.pos_emb_text = nn.Parameter(torch.randn(batch_size, len, dim, device='cuda'))
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.downsample = nn.Conv1d(self.dim, self.dim, 2, stride=2)
        self.apply(self._init_weights)
        self.save_hyperparameters()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.num_parameters**-0.5)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def model_step(self, text, hidden_prev):
        hidden_prev = hidden_prev + self.pos_emb_hidden
        text = text + self.pos_emb_text

        hidden_before_enc = torch.cat([hidden_prev, text], dim=1)
        hidden = self.enc(hidden_before_enc)
        hidden_after_dec = self.dec(hidden)

        hidden_prev = hidden_after_dec.narrow(1,0,self.len-1)
        text = hidden_after_dec.narrow(1,self.len-1,self.len+1)

        hidden = self.downsample(hidden.transpose(2,1)).transpose(2,1)

        return text, hidden_prev, hidden


    def training_step(self, batch, batch_idx):
        data, next = batch
        if batch_idx % 128 == 0:
            self.hidden = self.hidden_init
        data_head = data[:,0].unsqueeze(1)
        x = nn.functional.one_hot(data.long(), self.vocab_size).float()
        x_next = torch.cat([data_head, next], dim=1)
        x = self.token_in(x)
        if self.hidden is None:
            self.hidden = self.hidden_init
        with torch.no_grad():
            hidden = self.hidden
        hidden_next = hidden[:,1:self.len,:]
        x_hat, hidden_hat, hidden = self.model_step(x, hidden)
        x_hat = self.token_out(x_hat)
        self.hidden.weight = hidden
        loss_hidden = nn.MSELoss()(hidden_hat, hidden_next)
        loss_text = nn.CrossEntropyLoss()(x_hat.view(-1,self.vocab_size), x_next.view(-1).long())
        loss = loss_hidden + loss_text
        self.log("loss_hidden", loss_hidden, on_epoch=False, prog_bar=True)
        self.log("loss_text", loss_text, on_epoch=False, prog_bar=True)
        self.log("train_loss", loss, on_epoch=False, prog_bar=True)
        return loss

    def forward(self, x, hidden):
        x = nn.functional.one_hot(x.long(), self.vocab_size).float()
        x = self.token_in(x)

        #x = torch.cat([hidden, x], dim=1)

        #hidden = self.enc(x)
        #x = self.dec(hidden)
        #hidden = self.downsample(hidden.transpose(2,1)).transpose(2,1)
        #x = x.narrow(1,self.len,self.len)

        x, _, hidden = self.model_step(x, hidden)
        x = x.narrow(1,1,self.len)

        x_hat = self.token_out(x)
        x_hat = x_hat.softmax(2)
        return x_hat, hidden

    def reset_hidden(self, hidden):
        self.hidden = hidden

    def clear_hidden(self):
        self.hidden = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer



model = ModHyenaDownsampleRecurrentLang(
    ModHyena,
    len=256,
    dim=1024,
    dim_ff_scale=2,
    enc_depth=32,
    dec_depth=32,
    batch_size=1,
    text_load_mode='cut',
)

#model = Lang(
    #HyenaUet,
    #len=4096,
    #dim=512,
    #dim_scale=1,
    #dim_ff_scale=2,
    #depth_unet=0,
    #depth_hyena=64,
    #batch_size=1,
    #text_load_mode='cut',
    #enable_pre=False,
    #enable_middle=True,
    #enable_post=False,
#)