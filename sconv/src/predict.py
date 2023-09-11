from main import Lang 
import torchvision.transforms as transforms
import torch
import pytorch_lightning as pl
import numpy as np
import hydra

np.set_printoptions(threshold=np.inf)

@hydra.main(version_base=None, config_path="../configs/", config_name="config")
def main(cfg):
    device = 'cuda'
    model = Lang.load_from_checkpoint(cfg.predict.weight, strict=False)
    context_len = cfg.predict.context_len
    length = cfg.predict.max_len
    vocab_size = model.vocab_size
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)

    print(f"#parameter:{model.num_parameters}")

    def predict(prompt):
        prompt = torch.from_numpy(np.array([i for i in prompt.encode('utf-8')]).astype(int)).clone().to(device)
        prompt_len = len(prompt)
        prompt = torch.nn.functional.pad(prompt, (0, length-prompt_len), 'constant', 0)

        beam_width = 1
        model.randomize_init()
        model.reset_hidden()

        current_len = 0
        start = 0
        model.set_is_refresh(True)
        while prompt_len - current_len > context_len:
            model(prompt[current_len:current_len+context_len].view(1,context_len))
            current_len += context_len
            start += context_len
        model.set_is_refresh(False)

        predict_init = model(prompt[current_len:current_len+context_len].view(1,context_len))
        _, predict_init_i = predict_init.view(context_len, vocab_size)[prompt_len - current_len -1].topk(beam_width)
        prompt_beam = prompt.repeat(beam_width, 1)
        prompt_beam[:,prompt_len] = predict_init_i
        current_len = prompt_len

        previous_predict = None
        while current_len < length:
            #print(f"{current_len} {start}")
            #print(prompt_beam[:,start:start+context_len])
            model.set_is_refresh(current_len % context_len == 0)
            predict_beam = model(prompt_beam[:,start:start+context_len])
            _, predict_beam_i = predict_beam[:,current_len-1-start,:].reshape(beam_width * vocab_size).topk(beam_width)
            prompt_beam = prompt_beam[torch.div(predict_beam_i, vocab_size, rounding_mode='floor')]
            prompt_beam[:,current_len] = predict_beam_i % vocab_size 

            predict = prompt_beam[0]
            predict = predict.cpu().numpy().astype(dtype='uint8')
            predict = predict.tobytes().decode('utf-8', 'replace')

            current_len += 1

            print(f'{current_len} Bytes')
            print(predict)

            if current_len % context_len == 1 or context_len == 1:
                start = start + context_len

        predict = prompt_beam[0]
        predict = predict.cpu().numpy().astype(dtype='uint8')
        predict = predict.tobytes().decode('utf-8', 'replace')
        return predict

    while True:
        prompt = input('prompt:')
        predict(prompt)

if __name__ == '__main__':
    main()