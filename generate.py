# generate.py

import torch
import torch.nn.functional as F
from model import MiniGPT, encode, decode, device, vocab_size

# Config
block_size = 64

# Load model
model = MiniGPT(vocab_size=vocab_size, block_size=block_size).to(device)
model.load_state_dict(torch.load('minigpt.pth'))
model.eval()

@torch.no_grad()
def generate(idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]  # last time step
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
    return idx

# Start generation
prompt = "i like "
start = torch.tensor([encode(prompt)], device=device)
out = generate(start, 50)
for i in out:
    print(decode(i.tolist()))
