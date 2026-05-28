from dataclasses import dataclass
import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM


KVCache = Tuple[torch.Tensor, torch.Tensor]
PastKeyValues = List[Optional[KVCache]]

@dataclass
class QwenConfig:
    max_position_embeddings: int = 1024
    vocab_size: int = 151936
    hidden_size: int = 1024
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

        self.q_proj = nn.Linear(config.hidden_size, config.n_head * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.n_kv_head * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.n_kv_head * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_head * config.head_dim, config.hidden_size, bias=False)

        self.q_norm = Qwen3RMSNorm(config.head_dim)
        self.k_norm = Qwen3RMSNorm(config.head_dim)

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.group_size = config.n_head // config.n_kv_head

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.max_position_embeddings, config.max_position_embeddings)).view(
                1, 1, config.max_position_embeddings, config.max_position_embeddings
            ),
            persistent=False,
        )

        cos, sin = precompute_rope_cache(config.max_position_embeddings, config.head_dim, config.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor,
                past_key_values: Optional[KVCache] = None,
                use_cache: bool = False,
                position_offset: int = 0,
                ) -> tuple[torch.Tensor, Optional[KVCache]]:
        bsz, seqlen, _ = x.size()

        q = self.q_proj(x).view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seqlen, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seqlen, self.n_kv_head, self.head_dim).transpose(1, 2)


        q = self.q_norm(q)
        k = self.k_norm(k)

        start = position_offset
        end = position_offset + seqlen

        if end > self.cos.size(2):
            raise ValueError(
                f"RoPE position out of range: end={end}, max={self.cos.size(2)}. "
                "Increase max_position_embeddings or enforce cache/window trimming."
            )

        cos = self.cos[:, :, start:end, :].to(dtype=q.dtype, device=q.device)
        sin = self.sin[:, :, start:end, :].to(dtype=q.dtype, device=q.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)


        if past_key_values is not None:
            past_k, past_v = past_key_values
            k_cat = torch.cat([past_k, k], dim=2)
            v_cat = torch.cat([past_v, v], dim=2)

            max_ctx = self.cos.size(2)
            if k_cat.size(2) > max_ctx:
                k_cat = k_cat[:, :, -max_ctx:, :].contiguous()
                v_cat = v_cat[:, :, -max_ctx:, :].contiguous()
        else:
            k_cat = k
            v_cat = v

       
        if self.n_kv_head != self.n_head:
            k_attn = k_cat.repeat_interleave(self.group_size, dim=1)
            v_attn = v_cat.repeat_interleave(self.group_size, dim=1)
        else:
            k_attn = k_cat
            v_attn = v_cat

        t_q = q.size(2)
        t_k = k_attn.size(2)

        past_len = t_k - t_q

        # not required for SPDA
        att = (q @ k_attn.transpose(-2, -1)) * (1.0 / math.sqrt(k_attn.size(-1)))
        
        k_idx = torch.arange(t_k, device=x.device).view(1, t_k)
        q_limit = past_len + torch.arange(t_q, device=x.device).view(t_q, 1)
        causal = k_idx <= q_limit

        #============== SDPA with masking ==============
        # didn't see improvement in the tokens throughput in my GPU. Might be my GPU issue
        # use-amp: 540 tokens/s with or without SDPA
        # no use-amp: 2050 tokens/s with or without SDPA
        # SDPA expects additive mask: 0 for allowed, -inf for blocked

        # attn_bias = torch.zeros((t_q, t_k), device=x.device, dtype=q.dtype)
        # attn_bias = attn_bias.masked_fill(~causal, float("-inf"))
        # attn_bias = attn_bias.view(1, 1, t_q, t_k)

        # y = F.scaled_dot_product_attention(
        #     q,                      # [B, H, Tq, D]
        #     k_attn,                 # [B, H, Tk, D]
        #     v_attn,                 # [B, H, Tk, D]
        #     attn_mask=attn_bias,    # additive mask
        #     dropout_p=0.0,          # set >0 only if you add training dropout intentionally
        #     is_causal=False,        # mask already encodes cache-aware causality
        # )
        # print("q", q.shape, "k_attn", k_attn.shape, "v_attn", v_attn.shape, "y_pre", y.shape)
        # print("y_for_o_proj", y.shape, "expected_last_dim", self.n_head * self.head_dim)
        #=============================================================================
        att = att.masked_fill(~causal.view(1,1, t_q, t_k), float("-inf"))
        # att = att.masked_fill(self.mask[:, :, :seqlen, :seqlen] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v_attn
        #===============================================================================
        
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.n_head * self.head_dim)
        
        out = self.o_proj(y)
        present_kv = (k_cat, v_cat) if use_cache else None
        
        return out, present_kv


class MLP(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        hidden = 3 * config.hidden_size
        self.gate_proj = nn.Linear(config.hidden_size, hidden, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, config.hidden_size, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.self_attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size)

    def forward(self, x: torch.Tensor,
                past_key_values: Optional[KVCache] = None,
                use_cache: bool = False,
                position_offset: int = 0,
                ) -> tuple[torch.Tensor, Optional[KVCache]]:
    
        attn_out, present_kv =self.self_attn(self.input_layernorm(x), past_key_values=past_key_values, use_cache=use_cache, position_offset=position_offset)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, present_kv


class Qwen(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.config = config
        self.model = nn.ModuleDict(
            {
                "embed_tokens": nn.Embedding(config.vocab_size, config.hidden_size),
                "layers": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "norm": Qwen3RMSNorm(config.hidden_size),
            }
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self, idx: torch.Tensor, 
                targets: Optional[torch.Tensor] = None,
                past_key_values: Optional[PastKeyValues] = None,
                use_cache: bool = False,
                position_offset: Optional[int] = None,):

        bsz, seqlen = idx.size()
        if seqlen > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seqlen} exceeds max_position_embeddings {self.config.max_position_embeddings}"
            )

        if past_key_values is None:
            past_key_values = [None] * self.config.n_layer
        elif len(past_key_values) != self.config.n_layer:
            raise ValueError(
                f"past_key_values length {len(past_key_values)} != n_layer {self.config.n_layer}"
            )
        
        x = self.model.embed_tokens(idx)
        present_key_values: PastKeyValues = []

        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
            past_len = int(past_key_values[0][0].size(2))

        past_len = min(past_len, self.config.max_position_embeddings - 1)
        
        for i, block in enumerate(self.model["layers"]):
            x, present_kv = block(x, past_key_values=past_key_values[i], use_cache=use_cache, position_offset=past_len)
            if use_cache:
                present_key_values.append(present_kv)
        x = self.model["norm"](x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        if use_cache:
            return logits, loss, present_key_values
        
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_name: str = "Qwen/Qwen3-0.6B", max_position_embeddings: int = 1024):
        if model_name != "Qwen/Qwen3-0.6B":
            raise ValueError("Only Qwen/Qwen3-0.6B is supported in this implementation")

        config = QwenConfig(max_position_embeddings=max_position_embeddings)
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
