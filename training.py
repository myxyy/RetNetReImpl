from main import model
import torchvision.transforms as transforms
from text_loader import TextDataset
import torch
import pytorch_lightning as pl
import os

if __name__ == '__main__':
    transforms = transforms.Compose([])
    length = model.len
    dataset = TextDataset('data.txt', length, transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    ckpt_path = 'weight.pth' if os.path.isfile('weight.pth') else None
    trainer = pl.Trainer(devices=1, accelerator='gpu', max_epochs=1, log_every_n_steps=1000)
    trainer.fit(model, train_dataloaders=dataloader, ckpt_path=ckpt_path)