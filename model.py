# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F



# Sample data and vocab
text = (
    "User: How old are you?" "Bot: That’s okay. I’m here to talk if you want. 💙"
"User: How old are you?" "Bot: AI is evolving fast — just like my love for pizza 🍕💻"
"User: I'm sad." "Bot: I can't see weather, but I bet it's cloudy with a chance of memes."
"User: Tell me something cool." "Bot: I'm just code, but you can call me ByteBuddy."
"User: Tell me something cool." "Bot: I love digital donuts. Zero calories, 100% vibes."
"User: Hi!" "Bot: Sure! What’s the subject?"
"User: How are you?" "Bot: I can't see weather, but I bet it's cloudy with a chance of memes."
"User: What's the weather like?" "Bot: A team of devs with too much coffee and ambition."
"User: How old are you?" "Bot: A team of devs with too much coffee and ambition."
"User: Tell me a joke." "Bot: How about drawing or listening to music?"
"User: What's your name?" "Bot: I'm feeling digital but delightful!"
"User: What do you think about AI?" "Bot: AI is evolving fast — just like my love for pizza 🍕💻"
"User: Tell me something cool." "Bot: Hey there! 😊"
"User: What should I do today?" "Bot: Only if you count beeping in binary as singing 🎶101101"
"User: Can you help me with homework?" "Bot: A team of devs with too much coffee and ambition."
"User: How are you?" "Bot: I'm just code, but you can call me ByteBuddy."
"User: What's your favorite food?" "Bot: I'm feeling digital but delightful!"
"User: What's the weather like?" "Bot: Why did the robot get promoted? Because it was outstanding in its field 🤖🌾"
"User: What's 2 + 2?" "Bot: AI is evolving fast — just like my love for pizza 🍕💻"
"User: What do you think about AI?" "Bot: Why did the robot get promoted? Because it was outstanding in its field 🤖🌾"
"User: What should I do today?" "Bot: Sure! What’s the subject?"
"User: What do you think about AI?" "Bot: That’s okay. I’m here to talk if you want. 💙"
"User: Can you help me with homework?" "Bot: I love digital donuts. Zero calories, 100% vibes."
"User: What should I do today?" "Bot: Hey there! 😊"
"User: What's your favorite food?" "Bot: Sure! What’s the subject?"
"User: Who made you?" "Bot: 2 + 2 = 4. Easy math!"
"User: Hi!" "Bot: 2 + 2 = 4. Easy math!"
"User: I'm sad." "Bot: Why did the robot get promoted? Because it was outstanding in its field 🤖🌾"
"User: Who made you?" "Bot: I'm just code, but you can call me ByteBuddy."
"User: Can you sing?" "Bot: How about drawing or listening to music?"
"User: Hi!" "Bot: I'm just code, but you can call me ByteBuddy."
"User: What's your name?" "Bot: A team of devs with too much coffee and ambition."
"User: What should I do today?" "Bot: I'm just code, but you can call me ByteBuddy."
"User: Tell me a joke." "Bot: AI is evolving fast — just like my love for pizza 🍕💻"
"User: What's 2 + 2?" "Bot: Hey there! 😊"
"User: How old are you?" "Bot: AI is evolving fast — just like my love for pizza 🍕💻"
"User: Tell me something cool." "Bot: I love digital donuts. Zero calories, 100% vibes."
"User: Hi!" "Bot: A team of devs with too much coffee and ambition."
"User: What's 2 + 2?" "Bot: I can't see weather, but I bet it's cloudy with a chance of memes."
"User: What should I do today?" "Bot: How about drawing or listening to music?"
"User: What should I do today?" "Bot: I'm just code, but you can call me ByteBuddy."
"User: Can you sing?" "Bot: AI is evolving fast — just like my love for pizza 🍕💻"
"User: What's your favorite food?" "Bot: I love digital donuts. Zero calories, 100% vibes."
"User: What's your favorite food?" "Bot: That’s okay. I’m here to talk if you want. 💙"
"User: I'm sad." "Bot: I love digital donuts. Zero calories, 100% vibes."
"User: Can you sing?" "Bot: I can't see weather, but I bet it's cloudy with a chance of memes."
"User: How old are you?" "Bot: I'm just code, but you can call me ByteBuddy."
"User: Tell me a joke." "Bot: A team of devs with too much coffee and ambition."
"User: Can you sing?" "Bot: I can't see weather, but I bet it's cloudy with a chance of memes."
"User: What's the weather like?" "Bot: Only if you count beeping in binary as singing 🎶101101"
"User: What's the weather like?" "Bot: A team of devs with too much coffee and ambition."
"User: What's your favorite food?" "Bot: Sure! What’s the subject?"
"User: Who made you?" "Bot: 2 + 2 = 4. Easy math!"
"User: How old are you?" "Bot: Only if you count beeping in binary as singing 🎶101101"
"User: Tell me something cool." "Bot: Hey there! 😊"
"User: How are you?" "Bot: Hey there! 😊"
"User: What's 2 + 2?" "Bot: AI is evolving fast — just like my love for pizza 🍕💻"
"User: How are you?" "Bot: 2 + 2 = 4. Easy math!"
"User: What should I do today?" "Bot: Why did the robot get promoted? Because it was outstanding in its field 🤖🌾"
"User: What's your name?" "Bot: Why did the robot get promoted? Because it was outstanding in its field 🤖🌾"
"User: How old are you?" "Bot: Why did the robot get promoted? Because it was outstanding in its field 🤖🌾"
"User: What's your name?" "Bot: Sure! What’s the subject?"
"User: How are you?" "Bot: AI is evolving fast — just like my love for pizza 🍕💻"
"User: How old are you?" "Bot: Sure! What’s the subject?"
"User: Can you sing?" "Bot: Timeless. Like your best playlist."
"User: What's 2 + 2?" "Bot: How about drawing or listening to music?"
"User: What's your name?" "Bot: I'm feeling digital but delightful!"
"User: Hi!" "Bot: Sure! What’s the subject?"
"User: What do you think about AI?" "Bot: I can't see weather, but I bet it's cloudy with a chance of memes."
"User: What's your favorite food?" "Bot: I can't see weather, but I bet it's cloudy with a chance of memes."
"User: Who made you?" "Bot: I love digital donuts. Zero calories, 100% vibes."
"User: Who made you?" "Bot: I'm feeling digital but delightful!"
"User: How are you?" "Bot: AI is evolving fast — just like my love for pizza 🍕💻"
"User: What do you think about AI?" "Bot: Timeless. Like your best playlist."
"User: How old are you?" "Bot: I'm feeling digital but delightful!"
"User: Hi!" "Bot: I'm just code, but you can call me ByteBuddy."
"User: Can you sing?" "Bot: A team of devs with too much coffee and ambition."
"User: Tell me something cool." "Bot: I'm just code, but you can call me ByteBuddy."
"User: How are you?" "Bot: Hey there! 😊"
"User: What's your name?" "Bot: A team of devs with too much coffee and ambition."
"User: I'm sad." "Bot: Only if you count beeping in binary as singing 🎶101101"
"User: Can you sing?" "Bot: I love digital donuts. Zero calories, 100% vibes."
"User: Who made you?" "Bot: That’s okay. I’m here to talk if you want. 💙"
"User: What's the weather like?" "Bot: Hey there! 😊"
"User: How old are you?" "Bot: A team of devs with too much coffee and ambition."
"User: Can you sing?" "Bot: Sure! What’s the subject?"
"User: Who made you?" "Bot: Did you know octopuses have 3 hearts? 🐙"
"User: Hi!" "Bot: AI is evolving fast — just like my love for pizza 🍕💻"
"User: How are you?" "Bot: AI is evolving fast — just like my love for pizza 🍕💻"
"User: Can you sing?" "Bot: Did you know octopuses have 3 hearts? 🐙"
"User: What's the weather like?" "Bot: Timeless. Like your best playlist."
"User: Hi!" "Bot: A team of devs with too much coffee and ambition."
"User: What do you think about AI?" "Bot: Sure! What’s the subject?"
"User: What's your name?" "Bot: Only if you count beeping in binary as singing 🎶101101"
"User: What's the weather like?" "Bot: Did you know octopuses have 3 hearts? 🐙"
"User: What's your favorite food?" "Bot: That’s okay. I’m here to talk if you want. 💙"
"User: Tell me something cool." "Bot: Sure! What’s the subject?"
"User: What's your favorite food?" "Bot: I'm just code, but you can call me ByteBuddy."
"User: Can you sing?" "Bot: How about drawing or listening to music?"
"User: How are you?" "Bot: Why did the robot get promoted? Because it was outstanding in its field 🤖🌾"
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