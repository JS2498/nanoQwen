from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM


@dataclass
class QwenConfig:
    block_size: int = 1024
    vocab_size: int = 151936
    n_embed: int = 1024
    n_head: int = 16
    n_layer: int = 28
    n_kv_head: int = 8
    head_dim: int = 128
    rope_theta: float = 1_000_000.0


class Qwen3RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


def precompute_rope_cache(max_seq_len: int, head_dim: int, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    position_ids = torch.arange(max_seq_len, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    sinusoid_inp = torch.einsum("i,j->ij", position_ids, inv_freq)
    freqs = torch.cat([sinusoid_inp, sinusoid_inp], dim=-1)
    cos_cache = torch.cos(freqs)[None, None, :, :]
    sin_cache = torch.sin(freqs)[None, None, :, :]
    return cos_cache, sin_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        assert config.n_kv_head > 0 and config.n_head % config.n_kv_head == 0

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

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

        cos, sin = precompute_rope_cache(config.block_size, config.head_dim, config.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.size()

        q = self.q_proj(x).view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seqlen, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seqlen, self.n_kv_head, self.head_dim).transpose(1, 2)


        q = self.q_norm(q)
        k = self.k_norm(k)

        cos = self.cos[:, :, :seqlen, :].to(dtype=q.dtype, device=q.device)
        sin = self.sin[:, :, :seqlen, :].to(dtype=q.dtype, device=q.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)


        if self.n_kv_head != self.n_head:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :seqlen, :seqlen] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.n_head * self.head_dim)
        return self.o_proj(y)


class MLP(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        hidden = 3 * config.n_embed
        self.gate_proj = nn.Linear(config.n_embed, hidden, bias=False)
        self.up_proj = nn.Linear(config.n_embed, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, config.n_embed, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.self_attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.n_embed)
        self.post_attention_layernorm = Qwen3RMSNorm(config.n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.config = config
        self.model = nn.ModuleDict(
            {
                "embed_tokens": nn.Embedding(config.vocab_size, config.n_embed),
                "layers": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "norm": Qwen3RMSNorm(config.n_embed),
            }
        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        bsz, seqlen = idx.size()
        if seqlen > self.config.block_size:
            raise ValueError(f"Sequence length {seqlen} exceeds block size {self.config.block_size}")

        x = self.model.embed_tokens(idx)
        for block in self.model["layers"]:
            x = block(x)
        x = self.model["norm"](x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_name: str = "Qwen/Qwen3-0.6B", block_size: int = 1024):
        if model_name != "Qwen/Qwen3-0.6B":
            raise ValueError("Only Qwen/Qwen3-0.6B is supported in this implementation")

        config = QwenConfig(block_size=block_size)
        model = cls(config)

        model_hf = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        sd = model.state_dict()
        sd_hf = model_hf.state_dict()

        if set(sd.keys()) != set(sd_hf.keys()):
            only_hf = sorted(set(sd_hf.keys()) - set(sd.keys()))
            only_local = sorted(set(sd.keys()) - set(sd_hf.keys()))
            raise RuntimeError(
                "State dict key mismatch. only_hf="
                f"{len(only_hf)} only_local={len(only_local)}"
            )

        for key in sd_hf:
            if sd_hf[key].shape != sd[key].shape:
                raise RuntimeError(
                    f"Shape mismatch for {key}: hf={tuple(sd_hf[key].shape)} local={tuple(sd[key].shape)}"
                )
            with torch.no_grad():
                sd[key].copy_(sd_hf[key])

        model.load_state_dict(sd)
        return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
