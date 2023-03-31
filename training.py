from main import model
import torchvision.transforms as transforms
from text_loader import TextDataset
import torch
import pytorch_lightning as pl
import os

if __name__ == '__main__':
    transforms = transforms.Compose([])
    length = model.len
    dataset = TextDataset('data.txt', length, transforms, model.text_load_mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=model.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    ckpt_path = 'weight.ckpt' if os.path.isfile('weight.ckpt') else None
    trainer = pl.Trainer(devices=1, accelerator='gpu', max_epochs=4, log_every_n_steps=1000, precision=16)
    trainer.fit(model, train_dataloaders=dataloader, ckpt_path=ckpt_path)