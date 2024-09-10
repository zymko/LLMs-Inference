import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
import csv



class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_group == 0

        # key, query, value projections for all heads, but in a batch
        self.kv_attn = nn.Linear(config.n_embd, 2 * config.n_embd // config.n_group)

        self.q_attn = nn.Linear(config.n_embd, config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.n_group = config.n_group

        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                                                .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        
        B, T, C  = x.size()
        kv = self.kv_attn(x) # B T 3*C
        q = self.q_attn(x)

        k, v = kv.split(self.n_embd // self.n_group, dim = 2)

        k = k.view(B, T, self.n_head // self.n_group, C // self.n_head).unsqueeze(2) \
            .expand(B, T, self.n_group, self.n_head // self.n_group, C // self.n_head).reshape(B, T, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head // self.n_group, C // self.n_head).unsqueeze(2) \
            .expand(B, T, self.n_group, self.n_head // self.n_group, C // self.n_head).reshape(B, T, self.n_head, C // self.n_head)
        k = k.transpose(1, 2) # (B, nh, T, hs)
        v = v.transpose(1, 2) # (B, nh, T, hs)


        # k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)


        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim = -1)
        y = att @ v
        # y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    n_group: int = 12 # group attention

class GPT(nn.Module):
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.n_embd),
                wpe = nn.Embedding(num_embeddings=config.block_size, embedding_dim=config.n_embd),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(normalized_shape=config.n_embd)
                )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_embd = self.transformer.wpe(pos)
        tok_embd = self.transformer.wte(idx)
        x = pos_embd + tok_embd
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

class Text(Dataset):

    def __init__(self, B, T, text='None'):
        self.B = B
        self.T = T

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'load {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')

        self.tokens_reshape = self.tokens[: (len(self.tokens) // (self.T + 1)) * (self.T + 1)].view(len(self.tokens) // (self.T + 1), (self.T + 1))
    
    def __len__(self):
        return len(self.tokens) // (self.T + 1)

    def __getitem__(self, idx):

        # if (idx + 1) * self.T + 1 > self.tokens.size(0):
        return  self.tokens_reshape[idx][: self.T], self.tokens_reshape[idx][1 : self.T + 1]


if __name__ == '__main__':

    master_process=False
    # model=GPT.from_pretrained('gpt2')
    # print('Model loading successfully!')
    # model.eval()

    if torch.cuda.is_available():
        print("CUDA is available!")
        device = torch.device("cuda")  # You can now use CUDA
    else:
        print("CUDA is not available.")
        device = torch.device("cpu")  # Fallback to CPU

    model_type='gpt2'
    config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
    
    config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
    config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
    # create a from-scratch initialized minGPT model
    config = GPTConfig(**config_args)

    model=GPT(config)
    model.to(device)
    
    batch_size = 32
    seq_length = 1024

    with open('dataset/text_split_2.txt', 'r', encoding='utf-8') as file:
        text = file.read()


    text_dataset = Text(B=batch_size, T=seq_length, text=text)
    train_loader = DataLoader(text_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    start_time = time.time()
    for x, y in train_loader:
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
    end_time = time.time()
    running_time = end_time - start_time
    print(f'Total time: {running_time} seconds')
    memory=torch.cuda.max_memory_allocated()/ 1024 ** 2
    print(f'Peak memory is {memory} GB')


    results =[
        [model_type, batch_size, seq_length, running_time, memory, config.n_group]
    ]
    with open("gpt_results/output.csv", "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(results)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # for epoch in range(50):
    #     for x, y in train_loader:
    #         x, y = x.to(device), y.to(device)
    #         optimizer.zero_grad()
    #         logits, loss = model(x, y)
    #         loss.backward()
    #         optimizer.step()
    #         print(f"Epoch {epoch}, loss {loss.item()}:")


    # num_return_sequences = 5
    # max_length = 30

    # import tiktoken
    # enc = tiktoken.get_encoding('gpt2')
    # tokens = enc.encode("Hello, I'm a language model,")
    # tokens = torch.tensor(tokens, dtype=torch.long)
    # tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    # x = tokens.to(device)
    # torch.manual_seed(42)
    # torch.cuda.manual_seed(42)

    # while x.size(1) < max_length:
        
    #     with torch.no_grad():
    #         logits = model(x)
    #         logits = logits[:, -1, :]
    #         probs = F.softmax(logits, dim=1)
    #         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
    #         ix = torch.multinomial(topk_probs, 1)
    #         xcol = torch.gather(topk_indices, -1, ix)
    #         x = torch.cat((x, xcol), dim=1)

    # for i in range(num_return_sequences):
    #     tokens = x[i, :max_length].tolist()
    #     decoded = enc.decode(tokens)
    #     print('>', decoded)