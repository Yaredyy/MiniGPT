# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F



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
encode = lambda s: [stoi[c] for c in (s.lower())]
decode = lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

        # Causal mask to prevent looking into the future
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)   # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = att @ v  # (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.sa = SelfAttention(n_embd, n_head, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd=64, n_head=4, n_layer=4):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.position_embed = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.block_size = block_size

    def forward(self, x):
        B, T = x.size()
        assert T <= self.block_size, "Input exceeds block size"

        tok_emb = self.token_embed(x)  # (B, T, C)
        pos_emb = self.position_embed(torch.arange(T, device=x.device))  # (T, C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits