from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.register_buffer("bias",
                             torch.tril(torch.ones(config.block_size, config.block_size)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, C = x.shape
        x = self.c_attn(x)
        q, k, v = x.split(self.config.n_embd, dim=-1)
        q = q.view(B, T, self.config.n_head, -1).transpose(1, 2)
        k = k.view(B, T, self.config.n_head, -1).transpose(1, 2)
        v = v.view(B, T, self.config.n_head, -1).transpose(1, 2)
        attention = q @ k.transpose(2, 3) / np.sqrt(q.shape[-1])
        masked_attention = attention.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        y = F.softmax(masked_attention, dim=-1) @ v
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.NANO_GPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        return self.c_proj(x)


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


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self.init_weights)

    def init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
                nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=std)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        pos_embd = self.transformer.wpe(pos)
        token_embd = self.transformer.wte(idx)
        x = pos_embd + token_embd
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained GPT: {model_type}")

        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600)
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [sd_key for sd_key in sd.keys() if not sd_key.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [sd_key for sd_key in sd_hf.keys() if
                      not sd_key.endswith(".attn.masked_bias") and not sd_key.endswith(".attn.bias")]

        assert len(sd_keys_hf) == len(sd_keys), f"{len(sd_keys_hf)} != {len(sd_keys)}"

        conv1d2linear = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]

        for k in sd_keys:
            if any([k.endswith(m) for m in conv1d2linear]):
                sd[k].copy_(sd_hf[k].t())
            else:
                assert sd[k].shape == sd_hf[k].shape, f"{k}: {sd[k].shape} != {sd_hf[k].shape}"
                sd[k].copy_(sd_hf[k])
        return model

    # TODO: 写个函数自动计算参数的数量


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print("using device: ", device)

import tiktoken


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        enc = tiktoken.get_encoding("gpt2")
        with open("input.txt", "r") as f:
            text = f.read()
        self.tokens = torch.tensor(enc.encode(text))
        print(f"total tokens: {len(self.tokens)}")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_pos = 0

    def next_batch(self):
        start_pos = self.current_pos * (self.B * self.T)
        end_pos = start_pos + (self.B * self.T)
        buf = self.tokens[start_pos:end_pos + 1]
        x = torch.tensor(buf[:-1]).view(self.B, self.T)
        y = torch.tensor(buf[1:]).view(self.B, self.T)
        if start_pos > len(self.tokens):
            self.current_pos = 0
        else:
            self.current_pos += 1
        return x, y


# model = GPT.from_pretrained("gpt2")
model = GPT(GPTConfig())
model.eval()
model.to(device)

dataloader = DataLoaderLite(4, 32)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    optimizer.zero_grad()
    x, y = dataloader.next_batch()
    x, y = x.to(device), y.to(device)
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step: {i}, loss: {loss.item()}")
