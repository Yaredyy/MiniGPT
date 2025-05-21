# train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import MiniGPT, get_batch, encode, decode, device, vocab_size, data

# Config
batch_size = 64
block_size = 64
max_iters = 2000
eval_interval = 100
learning_rate = 5e-3
eval_iters = 20

model = MiniGPT(vocab_size=vocab_size, block_size=block_size, n_embd=64, n_head=4, n_layer=4).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, block_size, batch_size)
        logits = model(X)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        Y = Y.view(B * T)
        loss = F.cross_entropy(logits, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean()

best_loss = float('inf')  # track best validation loss

# Training
for step in range(max_iters):
    if step % eval_interval == 0:
        val_loss = estimate_loss()
        print(f"Step {step}, loss = {val_loss:.4f}")
        
        # Save best model if validation loss improved
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'minigpt_best.pth')
            print(f"Saved new best model at step {step} with loss {val_loss:.4f}")
    
    x_batch, y_batch = get_batch(data, block_size, batch_size)
    logits = model(x_batch)
    B, T, C = logits.shape
    loss = F.cross_entropy(logits.view(B * T, C), y_batch.view(B * T))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Optionally save final model as well
torch.save(model.state_dict(), 'minigpt_final.pth')
