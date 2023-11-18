import torchvision
from text_loader import TextDataset
import torch
import torch.nn as nn
import os
import hydra
from hydra.utils import instantiate
from tqdm import tqdm
from torch.distributed.pipeline.sync import Pipe

@hydra.main(version_base=None, config_path="../configs/", config_name="config")
def main(cfg):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)
    transforms = torchvision.transforms.Compose([])
    dataset = TextDataset(cfg.train_pipeline.text, cfg.train_pipeline.length, transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.train_pipeline.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True, drop_last=True)
    ckpt_path = cfg.train_pipeline.weight
    ckpt_path = ckpt_path if os.path.isfile(ckpt_path) else None
    dtype = torch.bfloat16
    #trainer = pl.Trainer(devices=1, accelerator='gpu', max_epochs=cfg.train.max_epochs, log_every_n_steps=cfg.train.log_every_n_steps, logger=[TensorBoardLogger('./')])
    model = instantiate(cfg.model)
    devices = cfg.train_pipeline.devices
    model = model(devices=devices)
    model = model.to(dtype)
    epochs = 0
    steps = 0
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['state_dict'])
        epochs = ckpt['epochs']
        steps = ckpt['steps']

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"#parameter:{num_parameters}")

    model_pipe = nn.Sequential(*model.module_list())
    model_pipe = Pipe(model_pipe, chunks=cfg.train_pipeline.batch_size)
    model_pipe.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
    def save():
        torch.save({'state_dict': model.state_dict(), 'steps': steps, 'epochs': epochs}, cfg.train_pipeline.weight)
    try:
        for _ in range(cfg.train_pipeline.max_epochs - epochs):
            pbar = tqdm(dataloader, initial=steps)
            for batch in pbar:
                optimizer.zero_grad()

                model.reset_hidden()
                text, text_next = batch
                text = text.to(devices[0])
                text_next = text_next.to(devices[-1])
                text = nn.functional.one_hot(text.long(), 256).to(dtype)

                text_hat = model_pipe(text).local_value()

                loss = nn.CrossEntropyLoss()(text_hat.view(-1,256), text_next.view(-1).long())
 
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())
                steps += 1
            save()
            steps = 0
            epochs += 1
    except KeyboardInterrupt:
        save()
        print(f'KeyboardInterrupted')
        print(f'steps:{steps}/{len(dataloader)} epochs:{epochs}/{cfg.train_pipeline.max_epochs}')


if __name__ == '__main__':
    main()