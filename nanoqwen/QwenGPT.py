"""
Implementation of QwenGPT, a large language model developed by Alibaba. This module provides an interface to interact with the QwenGPT model, allowing users to generate text based on given prompts.
Embedding dims: 1024
# of layers: 28
# of attention heads: 16
Context window size: 32k
Vocab size: 151k
FFN inner dim: 3072

# Features:
- RoPE
- GQA
- QK norm
- RMS norm
- SLU activation
"""

# lets import the HF transformers library to load the QwenGPT model

from transformers import AutoTokenizer, AutoModelForCausalLM


model_name = "Qwen/Qwen3-0.6B"

# load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # load model
# model = AutoModelForCausalLM.from_pretrained(
#     model_name, 
#     torch_dtype="auto",
#     device_map="auto")

# # prompt
# prompt = "Tell me about the Qwen 3 600M model in English."
# message = [{
#     "role": "user", 
#     "content": prompt}]


# print("\n=== Modules + Params ===")
# total_params = 0

# for module_path, module in model.named_modules():
#     direct_params = list(module.named_parameters(recurse=False))

#     # modules like Dropout/activation may have no parameters
#     if not direct_params:
#         continue

#     for param_name, param in direct_params:
#         full_name = f"{module_path}.{param_name}" if module_path else param_name
#         print(
#             f"{full_name:90s} "
#             f"{module.__class__.__name__:25s} "
#             f"{str(tuple(param.shape)):20s} "
#             f"{str(param.dtype):15s}"
#         )
#         total_params += param.numel()

# print(f"\nTotal parameters: {total_params:,}")


# print each parameter: name + shape + dtype + device
# print("\n=== Parameters (name -> shape) ===")
# total_params = 0
# for name, param in model.named_parameters():
#     print(f"{name:80s} {tuple(param.shape)!s:20s} dtype={param.dtype} device={param.device}")
#     total_params += param.numel()

# print(f"\nTotal parameters: {total_params:,}")


# print("\n=== Modules ===")
# for name, module in model.named_modules():
#     print(f"{name:80s} {module.__class__.__name__}")


# text = tokenizer.apply_chat_template(message, 
#                                      tokenize=False,
#                                      add_generation_prompt=True,
#                                      enable_thinking=True)

# print(f"Formatted input text: {text}")
# model_inputs = tokenizer([text], return_tensors="pt").to(model.device)  # 'pt' for PyTorch tensor

# print(f"Model inputs: {model_inputs}")

# generated_ids = model.generate(**model_inputs, max_new_tokens=32678)

# print(f"Shape of generated_ids: {generated_ids.shape}")
# print(f"EoS token ID: {tokenizer.eos_token_id}")

# # the generated ids will contain the input ids followed by the generated ids. We need to extract only the generated ids.
# # [0] to get the first (and only) sequence in the batch, and then we slice from the length of the input ids to get only the generated ids.
# output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()


# # not a nice menthod to find the index of the thinking token, but it works. 
# # We reverse the output ids and find the index of the thinking token (151668) from the end, 
# # then we subtract that index from the total length to get the index from the start.
# try:
#     index = len(output_ids) - output_ids[::-1].index(151668) - 1
# except ValueError:
#     index = 0

# print(f"Index of thinking token: {index}")

# # entire_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
# thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
# content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")


# # print(f"Entire output: {entire_output}")

# print(f"Thinking content: {thinking_content}")
# print(f"Content: {content}")


#%  
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect

@dataclass
class QwenConfig:
    block_size: int = 1024
    vocab_size: int = 151936
    n_embed: int = 1024
    n_head: int = 16
    n_layer: int = 28
    n_kv_head: int = 8
    head_dim: int = 128
    rope_theta: float = 1000000

class Qwen3RMSNorm(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embed))

    def forward(self, x):
        eps = 1e-6
        norm_x = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return norm_x * self.weight


def rotate_half(x):
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


def precompute_rope_cache(max_seq_len, head_dim, theta):
    position_ids = torch.arange(max_seq_len, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    sinusoid_inp = torch.einsum("i,j->ij", position_ids, inv_freq)
    freqs = torch.cat([sinusoid_inp, sinusoid_inp], dim=-1)
    cos_cache = torch.cos(freqs)[None, None, :, :]
    sin_cache = torch.sin(freqs)[None, None, :, :]
    return cos_cache, sin_cache

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_kv_head > 0 and config.n_head % config.n_kv_head == 0, "n_head must be divisible by n_kv_head"

        self.q_proj = nn.Linear(config.n_embed, config.n_head * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embed, config.n_kv_head * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embed, config.n_kv_head * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_head * config.head_dim, config.n_embed, bias=False)

        self.q_norm = Qwen3RMSNorm(config.head_dim)
        self.k_norm = Qwen3RMSNorm(config.head_dim)

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.group_size = config.n_head // config.n_kv_head

        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size), persistent=False)

        cos, sin = precompute_rope_cache(config.block_size, config.head_dim, config.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimension

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)      # (B, nh, T, hd)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)   # (B, nkv, T, hd)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)   # (B, nkv, T, hd)

        cos = self.cos[:, :, :T, :].to(dtype=q.dtype, device=q.device)
        sin = self.sin[:, :, :T, :].to(dtype=q.dtype, device=q.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.n_kv_head != self.n_head:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf')) # (B, nh, T, T)
        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        y = att @ v # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim) # (B, T, nh * hd)
        y = self.o_proj(y) # (B, T, C)

        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embed, 3*config.n_embed, bias=False)
        self.up_proj = nn.Linear(config.n_embed, 3*config.n_embed, bias=False)
        self.down_proj = nn.Linear(3*config.n_embed, config.n_embed, bias=False)

        # SiLU activation should be defined
        self.silu = nn.SiLU()

    def forward(self, x):
        x_gate = self.gate_proj(x)
        x_up = self.up_proj(x)
        x_intermediate = self.silu(x_gate) * x_up
        x_down = self.down_proj(x_intermediate)

        return x_down

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

        # define the norms
        self.input_layernorm = Qwen3RMSNorm(config.n_embed)
        self.post_attention_layernorm = Qwen3RMSNorm(config.n_embed)


    # need to fix it w.r.t to the norms
    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))

        return x


class Qwen(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = nn.ModuleDict(dict(
            embed_tokens = nn.Embedding(config.vocab_size, config.n_embed),
            layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            norm = Qwen3RMSNorm(config.n_embed)
        ))

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()

        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"

        tok_emb = self.model.embed_tokens(idx) # (B, T, n_embed)

        x = tok_emb

        for block in self.model.layers:
            x = block(x)
        
        x = self.model.norm(x) # (B, T, n_embed)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_name):
        # this method should load the pretrained weights from the Hugging Face model and initialize the Qwen model with those weights
        assert model_name == "Qwen/Qwen3-0.6B", "Only Qwen/Qwen3-0.6B is supported in this implementation"

        config_args = {
            'Qwen/Qwen3-0.6B': dict(n_layer=28, n_head=16, n_embed=1024)
        }[model_name]

        config_args['vocab_size'] = 151936
        config_args['block_size'] = 1024
        config_args['n_kv_head'] = 8

        config = QwenConfig(**config_args)
        model = Qwen(config)

        sd = model.state_dict()
        sd_keys = sd.keys()

        model_hf = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        sd_hf = model_hf.state_dict()
        sd_hf_keys = sd_hf.keys()

        missing = [k for k in sd_hf if k not in sd]
        extra = [k for k in sd if k not in sd_hf]
        shape_mismatch = [(k, sd_hf[k].shape, sd[k].shape) for k in sd_hf if k in sd and sd_hf[k].shape != sd[k].shape]

        print("missing:", len(missing))
        print("extra:", len(extra))
        print("shape_mismatch:", len(shape_mismatch))

        extra = sorted(set(sd_keys) - set(sd_hf_keys))
        print(len(extra))
        for k in extra:
            print(k)


        assert len(sd_keys) == len(sd_hf_keys), f"Number of parameters in the Qwen model ({len(sd_keys)}) does not match the Hugging Face model ({len(sd_hf_keys)})"


        from tabulate import tabulate

        # Collect all unique keys
        all_keys = sorted(set(sd_hf.keys()) | set(sd.keys()))

        rows = []

        for k in all_keys:
            shape_hf = tuple(sd_hf[k].shape) if k in sd_hf else None
            shape_sd = tuple(sd[k].shape) if k in sd else None
            rows.append([k, shape_hf, shape_sd])

        # Print comparison table
        print(tabulate(rows, headers=["Key", "HF Shape", "SD Shape"], tablefmt="grid"))

        # Separate unmatched keys
        only_hf = sorted(set(sd_hf.keys()) - set(sd.keys()))
        only_sd = sorted(set(sd.keys()) - set(sd_hf.keys()))

        print("\nKeys only in HuggingFace dict:")
        for k in only_hf:
            print(k, tuple(sd_hf[k].shape))

        print("\nKeys only in Current model dict:")
        for k in only_sd:
            print(k, tuple(sd[k].shape))

        for k in sd_hf_keys:

            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

        return model



# lets print the model architecture and the number of parameters in each module, similar to what we did with the Hugging Face model. We can also print the total number of parameters in the model.


print("Model Architecture:")

model = Qwen.from_pretrained('Qwen/Qwen3-0.6B')

qwen_config = QwenConfig()
qwen_model = Qwen(qwen_config)
total_params = 0

# print each parameter: name + shape + dtype + device
print("\n=== Parameters (name -> shape) ===")
total_params = 0
for name, param in qwen_model.named_parameters():
    print(f"{name:80s} {tuple(param.shape)!s:20s} dtype={param.dtype} device={param.device}")
    total_params += param.numel()

print(f"\nTotal parameters: {total_params:,}")


model.eval()
# model.to('cuda')
num_return_sequences = 3
max_length = 300

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokens = tokenizer("Hello, I am language model", return_tensors="pt").input_ids
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)

torch.manual_seed(36)
x = tokens
print(f"Initial tokens shape: {x.shape}")
x = x.repeat(num_return_sequences, 1)
print(f"Tokens shape after repeat: {x.shape}")
# x = x.view(1, -1)  # Ensure x has shape (1, T)
# x = torch.zeros((5,1), dtype=torch.long)
while x.size(1) < max_length:
  with torch.no_grad():
    logits, _ = model(x)
    logits = logits[:,-1,:]

    probs = F.softmax(logits, -1)

    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
    ix = torch.multinomial(topk_probs,1)

    x_new = torch.gather(topk_indices, -1, ix)
    x = torch.cat((x, x_new), axis=1)



for i in range(num_return_sequences):
  tokens = x[i, :max_length].tolist()
  decoded = tokenizer.decode(tokens)
  print(decoded)
  print('-'*50)
