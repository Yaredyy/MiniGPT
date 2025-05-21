# model.py

import torch
import torch.nn as nn

# Sample data and vocab
text = (
    "i like pizza and you like pasta. "
    "we all enjoy tasty food like sushi, ramen, burgers, and fries. "
    "sometimes i eat spicy curry or sweet desserts. "
    "food makes us happy and full. "
    "he prefers salads with grilled chicken or avocado. "
    "breakfast often includes pancakes, eggs, and orange juice. "
    "ice cream on a hot day is the best treat. "
    "she loves cooking homemade lasagna for dinner. "
    "we tried dumplings and noodles from the new restaurant. "
    "healthy snacks like apples and almonds are great. "
    "everyone has a favorite dish that reminds them of home. "
    "cheese melts perfectly on top of hot nachos. "
    "drinking water and staying hydrated is important. "
    "baking cookies with chocolate chips is fun. "
    "every meal tells a story and brings people together. "
    "some days we crave salty popcorn or buttery toast."
)

chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, 64)
        self.position_embed = nn.Embedding(block_size, 64)
        self.l1 = nn.Linear(64, 64)
        self.lm_head = nn.Linear(64, vocab_size)

    def forward(self, x):
        tok = self.token_embed(x)
        pos = self.position_embed(torch.arange(x.size(1), device=x.device))
        out = tok + pos
        out = torch.relu(self.l1(out))
        return self.lm_head(out)