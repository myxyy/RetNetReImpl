from main import Lang 
import torchvision.transforms as transforms
import torch
import pytorch_lightning as pl
import numpy as np

np.set_printoptions(threshold=np.inf)

model = Lang.load_from_checkpoint('weight.ckpt', strict=False)
length = model.len * 10
vocab_size = model.vocab_size
for p in model.parameters():
    p.requires_grad = False
model = model.cuda()

print(f"#parameter:{model.num_parameters}")

def predict(prompt):
    prompt = torch.from_numpy(np.array([i for i in prompt.encode('utf-8')]).astype(int)).clone().cuda()
    prompt_len = len(prompt)
    prompt = torch.nn.functional.pad(prompt, (0,length-prompt_len),'constant',0)

    beam_width = 2
    model.clear_hidden()
    model.set_is_refresh(False)
    predict_init = model(prompt.view(1,length)[:,0:model.len])
    _, predict_init_i = predict_init.view(model.len, vocab_size)[prompt_len-1].topk(beam_width)
    prompt_beam = prompt.repeat(beam_width, 1)
    prompt_beam[:,prompt_len] = predict_init_i
    prompt_len = prompt_len + 1

    start = 0
    while prompt_len < length:
        print(f"{prompt_len} {start}")
        #print(prompt_beam[:,start:start+model.len])
        model.set_is_refresh(prompt_len % model.len == 0)
        predict_beam = model(prompt_beam[:,start:start+model.len])
        _, predict_beam_i = predict_beam[:,prompt_len-1-start,:].reshape(beam_width * vocab_size).topk(beam_width)
        prompt_beam = prompt_beam[torch.div(predict_beam_i, vocab_size, rounding_mode='floor')]
        prompt_beam[:,prompt_len] = predict_beam_i % vocab_size 
        prompt_len = prompt_len + 1

        predict = prompt_beam[0]
        predict = predict.cpu().numpy().astype(dtype='uint8')
        predict = predict.tobytes().decode('utf-8', 'replace')
        print(predict)

        if prompt_len % model.len == 1 and prompt_len > 1:
            start = start + model.len

    predict = prompt_beam[0]
    predict = predict.cpu().numpy().astype(dtype='uint8')
    predict = predict.tobytes().decode('utf-8', 'replace')
    return predict

while True:
    prompt = input('prompt:')
    print(predict(prompt))