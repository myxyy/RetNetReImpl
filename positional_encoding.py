import torch
import torch.nn as nn
import math

# 10000うんちゃらのPositionalEncodingが気に入らないので独自実装
# TAU / lenを基本周波数としてTAU / len * len^((d-1)/d)まで上げる
class PositionalEncoding(nn.Module):
    def __init__(self, len: int, d: int, requires_grad=False):
        super().__init__()
        if d % 2 != 0:
            raise ValueError("PEError")
        half_d = d // 2
        position = torch.arange(len).unsqueeze(1)
        div_term = torch.pow(len, 1-torch.arange(half_d)/half_d)
        self.pe = torch.zeros(len, d)
        self.pe[:, 0::2] = torch.cos(2 * math.pi * position / div_term)
        self.pe[:, 1::2] = torch.sin(2 * math.pi * position / div_term)
        self.pe = nn.Parameter(self.pe, requires_grad=requires_grad)
        self.shape = (len, d)

    def forward(self, x: torch.Tensor=None) -> torch.Tensor:
        return self.pe if x is None else x + self.pe

    def __call__(self):
        return self.pe