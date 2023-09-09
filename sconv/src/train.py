import torchvision
from text_loader import TextDataset
import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from hydra.utils import instantiate

@hydra.main(version_base=None, config_path="../configs/", config_name="config")
def main(cfg):
    transforms = torchvision.transforms.Compose([])
    dataset = TextDataset(cfg.train.text, cfg.train.length, transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count(), pin_memory=True, drop_last=True)
    ckpt_path = cfg.train.weight
    ckpt_path = ckpt_path if os.path.isfile(ckpt_path) else None
    trainer = pl.Trainer(devices=1, accelerator='gpu', max_epochs=cfg.train.max_epochs, log_every_n_steps=cfg.train.log_every_n_steps, logger=[TensorBoardLogger('./')])
    model = instantiate(cfg.model)
    trainer.fit(model, train_dataloaders=dataloader, ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()