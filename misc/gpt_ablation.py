from dataclasses import dataclass, asdict
from pathlib import Path
import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import wandb
except Exception:
    wandb = None


@dataclass
class TrainConfig:
    batch_size: int = 32
    block_size: int = 128
    n_embd: int = 192
    n_head: int = 6
    n_kv_head: int = 6
    n_layer: int = 6
    dropout: float = 0.1
    max_iters: int = 2500
    eval_interval: int = 20
    eval_iters: int = 100
    learning_rate: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu" 
    seed: int = 1337
    # Feature toggles
    use_rope: bool = False
    use_rmsnorm: bool = False
    use_qk_norm: bool = False
    use_relu2: bool = False
    use_no_bias_linear: bool = False
    use_gqa: bool = False
    use_untied_lm_head: bool = False
    use_logit_softcap: bool = False


def maybe_download_data(path: Path) -> str:
    if not path.exists():
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, path)
    return path.read_text(encoding="utf-8")


def build_char_tokenizer(text: str):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
    return len(chars), encode, decode


class Norm(nn.Module):
    def __init__(self, dim: int, use_rmsnorm: bool):
        super().__init__()
        self.use_rmsnorm = use_rmsnorm
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        if self.use_rmsnorm:
            return F.rms_norm(x, (x.size(-1),))
        return self.ln(x)


def apply_rope(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=-1)


def precompute_rope(block_size, head_dim, device):
    freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(block_size, device=device).float()
    angles = torch.outer(t, freqs)
    return angles.cos()[None, :, None, :], angles.sin()[None, :, None, :]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.cfg = cfg
        self.head_dim = cfg.n_embd // cfg.n_head
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head if cfg.use_gqa else cfg.n_head
        assert self.n_head % self.n_kv_head == 0
        bias = not cfg.use_no_bias_linear
        self.q = nn.Linear(cfg.n_embd, self.n_head * self.head_dim, bias=bias)
        self.k = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=bias)
        self.v = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=bias)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        if cfg.use_rope:
            cos, sin = precompute_rope(cfg.block_size, self.head_dim, cfg.device)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_head, self.head_dim)
        k = self.k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v(x).view(B, T, self.n_kv_head, self.head_dim)

        if self.cfg.use_rope:
            q = apply_rope(q, self.cos[:, :T], self.sin[:, :T])
            k = apply_rope(k, self.cos[:, :T], self.sin[:, :T])
        if self.cfg.use_qk_norm:
            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))

        if self.n_kv_head != self.n_head:
            reps = self.n_head // self.n_kv_head
            k = k.repeat_interleave(reps, dim=2)
            v = v.repeat_interleave(reps, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = self.dropout(att) @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        bias = not cfg.use_no_bias_linear
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=bias)
        self.dropout = nn.Dropout(cfg.dropout)
        self.use_relu2 = cfg.use_relu2

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x).square() if self.use_relu2 else F.relu(x)
        return self.dropout(self.proj(x))


class Block(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.ln1 = Norm(cfg.n_embd, cfg.use_rmsnorm)
        self.ln2 = Norm(cfg.n_embd, cfg.use_rmsnorm)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, cfg: TrainConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(vocab_size, cfg.n_embd)
        self.pos_emb = None if cfg.use_rope else nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = Norm(cfg.n_embd, cfg.use_rmsnorm)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size, bias=False)
        if not cfg.use_untied_lm_head:
            self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if self.cfg.use_logit_softcap:
            softcap = 15.0
            logits = softcap * torch.tanh(logits / softcap)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def run_experiment(exp_name: str, cfg: TrainConfig):
    
    torch.manual_seed(cfg.seed)
    text = maybe_download_data(Path("input.txt"))
    vocab_size, encode, decode = build_char_tokenizer(text)
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    def get_batch(split):
        source = train_data if split == "train" else val_data
        ix = torch.randint(len(source) - cfg.block_size, (cfg.batch_size,))
        x = torch.stack([source[i:i + cfg.block_size] for i in ix]).to(cfg.device)
        y = torch.stack([source[i + 1:i + cfg.block_size + 1] for i in ix]).to(cfg.device)
        return x, y

    @torch.no_grad()
    def estimate():
        model.eval()
        out = {}
        for split in ["train", "val"]:
            losses = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                xb, yb = get_batch(split)
                _, loss = model(xb, yb)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    model = GPTLanguageModel(cfg, vocab_size).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    if wandb is None:
        raise RuntimeError("wandb is not installed/importable. Install it in this environment to proceed.")

    # Make W&B logging robust by default.
    os.environ.pop("WANDB_DISABLED", None)
    os.environ.setdefault("WANDB_MODE", "online")
    wandb.login(anonymous="allow", relogin=False)

    run = None
    init_err = None
    for _ in range(2):
        try:
            run = wandb.init(
                project="vanilla-vs-nanochat-ablations",
                name=exp_name,
                config=asdict(cfg),
                mode="online",
                settings=wandb.Settings(start_method="thread"),
                reinit=True,
            )
            break
        except Exception as e:
            init_err = e
    if run is None:
        raise RuntimeError(f"Failed to initialize W&B run after retry: {init_err}")

    print(f"[wandb] run initialized: name={run.name} id={run.id}")
    print(f"[wandb] url: {run.url}")

    run.define_metric("iteration")
    run.define_metric("train/step_loss", step_metric="iteration")
    run.define_metric("train/loss", step_metric="iteration")
    run.define_metric("val/loss", step_metric="iteration")

    for step in range(cfg.max_iters):
        if step % cfg.eval_interval == 0 or step == cfg.max_iters - 1:
            losses = estimate()
            print(f"[{exp_name}] step {step}: train={losses['train']:.4f}, val={losses['val']:.4f}")
            run.log(
                {"iteration": step, "train/loss": losses["train"], "val/loss": losses["val"]},
            )
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        # Keep W&B chart active between eval checkpoints.
        if step % 5 == 0:
            run.log({"iteration": step, "train/step_loss": loss.item()})

    context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
    out_50 = model.generate(context, max_new_tokens=50)[0].tolist()
    out_100 = model.generate(context, max_new_tokens=100)[0].tolist()
    sample_50 = decode(out_50)
    sample_100 = decode(out_100)
    print(f"\n[{exp_name}] sample_50:\n{sample_50}\n")
    print(f"[{exp_name}] sample_100:\n{sample_100}\n")
    run.summary["samples/50_tokens"] = sample_50
    run.summary["samples/100_tokens"] = sample_100
    run.finish()


def feature_sweep():
    base = TrainConfig()
    experiments = [
        # ("00_baseline", {}),
        # ("01_rope", {"use_rope": True}),
        # ("02_rmsnorm", {"use_rope": True, "use_rmsnorm": True}),
        # ("03_qk_norm", {"use_rope": True, "use_rmsnorm": True, "use_qk_norm": True}),
        ("04_relu2", {"use_rope": True, "use_rmsnorm": True, "use_qk_norm": True, "use_relu2": True}),
        ("05_no_bias", {"use_rope": True, "use_rmsnorm": True, "use_qk_norm": True, "use_relu2": True, "use_no_bias_linear": True}),
        ("06_gqa", {"use_rope": True, "use_rmsnorm": True, "use_qk_norm": True, "use_relu2": True, "use_no_bias_linear": True, "use_gqa": True, "n_kv_head": 2}),
        ("07_untied_head", {"use_rope": True, "use_rmsnorm": True, "use_qk_norm": True, "use_relu2": True, "use_no_bias_linear": True, "use_gqa": True, "n_kv_head": 2, "use_untied_lm_head": True}),
        ("08_logit_softcap", {"use_rope": True, "use_rmsnorm": True, "use_qk_norm": True, "use_relu2": True, "use_no_bias_linear": True, "use_gqa": True, "n_kv_head": 2, "use_untied_lm_head": True, "use_logit_softcap": True}),
    ]
    for name, patch in experiments:
        cfg = TrainConfig(**{**asdict(base), **patch})
        run_experiment(name, cfg)


if __name__ == "__main__":
    feature_sweep()


"""
To Do:
1. Log the loss more frequently (e.g. every 10 steps) to get a smoother curve
2. Add a command line interface to run individual experiments or the full sweep
3. Add support for saving and loading model checkpoints
4. Add more features to ablate, such as different activation functions, normalization techniques or attention mechanisms
5. Add support for running on different datasets, such as Penn Treebank or WikiText-2
7. Add support for running on different hardware, such as CPU, GPU or TPU
10. Add support for logging additional metrics, such as perplexity, accuracy or training time, to get a more comprehensive view of the model performance    
13. Add support for analyzing the learned representations and attention patterns of the models to understand how each feature affects the internal workings of the model    
16. Add support for running on different evaluation metrics, such as BLEU, ROUGE or F1 score, to see how the ablations affect different aspects of the model performance
19. Add support for running on different regularization techniques, such as weight decay, dropout or label smoothing, to see how the ablations interact with different methods for preventing overfitting
21. Add support for running on different data preprocessing techniques, such as byte pair encoding, wordpiece or sentencepiece, to see how the ablations interact with different ways of tokenizing the input text
"""
