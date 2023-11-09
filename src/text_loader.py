import torch
import torch.nn as nn
import numpy
from torch.utils.data import Dataset
from typing import Tuple, Literal

class TextDataset(Dataset):
    def __init__(self, path: str, size: int, transforms=None) -> None:
        super().__init__()
        self.size = size
        self.transforms = transforms
        self.text = numpy.array([i for i in open(path, 'r', encoding='utf-8').read().encode(encoding='utf-8')])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.text[index*self.size:(index+1)*self.size]
        data_next = self.text[index*self.size+1:(index+1)*self.size+1]

        if self.transforms is not None:
            data = self.transforms(data)
            data_next = self.transforms(data_next)
        return data, data_next


    def __len__(self) -> int:
        return (len(self.text)-1)//self.size