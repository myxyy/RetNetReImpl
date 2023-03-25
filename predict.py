from main import Lang
import torchvision.transforms as transforms
import torch
import pytorch_lightning as pl
import numpy as np

np.set_printoptions(threshold=np.inf)

model = Lang.load_from_checkpoint('weight.ckpt')
length = model.len
vocab_size = model.vocab_size
model = model.cuda()

print(f"#parameter:{sum(p.numel() for p in model.parameters() if p.requires_grad)}")

def predict(prompt):
    prompt = torch.from_numpy(np.array([i for i in prompt.encode('utf-8')]).astype(int)).clone().cuda()
    prompt_len = len(prompt)
    prompt = torch.nn.functional.pad(prompt, (0,length-prompt_len),'constant',0)

    beam_width = 1
    predict_init = model(prompt.view(1,length))
    _, predict_init_i = predict_init.view(length, vocab_size)[prompt_len-1].topk(beam_width)
    prompt_beam = prompt.repeat(beam_width, 1)
    prompt_beam[:,prompt_len] = predict_init_i
    prompt_len = prompt_len + 1

    while prompt_len < length:
        predict_beam = model(prompt_beam)
        _, predict_beam_i = predict_beam[:,prompt_len-1,:].reshape(beam_width * vocab_size).topk(beam_width)
        prompt_beam = prompt_beam[torch.div(predict_beam_i, vocab_size, rounding_mode='floor')]
        prompt_beam[:,prompt_len] = predict_beam_i % vocab_size 
        prompt_len = prompt_len + 1

    predict = prompt_beam[0]
    predict = predict.cpu().numpy().astype(dtype='uint8')
    predict = predict.tobytes().decode('utf-8', 'replace')
    return predict

while True:
    prompt = input('prompt:')
    print(predict(prompt))