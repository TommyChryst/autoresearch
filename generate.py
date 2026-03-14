"""
Standalone inference script for the medical GPT model.
Uses standard PyTorch attention — no flash-attn3 required, runs on CPU or GPU.

Usage:
    python generate.py "SOAP NOTE:" --max_tokens 200 --temperature 0.8
    python generate.py "HISTORY:" --max_tokens 300 --temperature 0.9 --top_k 50
    python generate.py "CHIEF COMPLAINT:" --checkpoint /path/to/model.pt
"""

import argparse
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Model architecture (identical to train.py, but with standard attention)
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # Standard PyTorch attention — no FA3 needed, works on CPU/GPU
        # GQA: repeat k/v heads to match q heads
        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)

        q = q.transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2)  # (B, T, n_head, head_dim)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size=None):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.cos.size(1), f"Sequence length {T} exceeds max {self.cos.size(1)}"
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin)
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        return logits


# ---------------------------------------------------------------------------
# Tokenizer (same as prepare.py)
# ---------------------------------------------------------------------------

TOKENIZER_PATH = Path.home() / ".cache" / "autoresearch" / "tokenizer" / "tokenizer.pkl"


def load_tokenizer(path=None):
    path = Path(path) if path else TOKENIZER_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {path}.\n"
            "Run: uv run python prepare_medical.py"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path} ({checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB)")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config_dict = checkpoint["config"]
    config = GPTConfig(**config_dict)
    print(f"Model config: depth={config.n_layer}, dim={config.n_embd}, heads={config.n_head}, vocab={config.vocab_size}")

    model = GPT(config)

    # Strip _orig_mod. prefix added by torch.compile
    state_dict = checkpoint["model_state"]
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("_orig_mod.", "")] = v
    model.load_state_dict(cleaned, strict=True)

    model.to(device)
    model.eval()

    val_bpb = checkpoint.get("val_bpb", "unknown")
    step = checkpoint.get("step", "unknown")
    print(f"Loaded: val_bpb={val_bpb}, trained_steps={step}")
    return model


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens=200, temperature=0.8, top_k=None, device="cpu"):
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    seq_len = model.config.sequence_len

    for _ in range(max_tokens):
        # Crop context to sequence length
        idx_cond = idx if idx.size(1) <= seq_len else idx[:, -seq_len:]

        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature  # (1, vocab_size)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)

    generated_tokens = idx[0, len(tokens):].tolist()
    return tokenizer.decode(generated_tokens)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate text from medical GPT checkpoint")
    parser.add_argument("prompt", type=str, help="Prompt text to complete")
    parser.add_argument("--max_tokens", type=int, default=200, help="Max tokens to generate (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling (default: disabled)")
    parser.add_argument("--checkpoint", type=str, default="model.pt", help="Path to model.pt checkpoint")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to tokenizer.pkl (default: ~/.cache/autoresearch/...)")
    parser.add_argument("--device", type=str, default=None, help="Device: cpu, cuda, mps (default: auto)")
    args = parser.parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    tokenizer = load_tokenizer(args.tokenizer)
    model = load_model(args.checkpoint, device)

    print(f"\nPrompt: {args.prompt!r}")
    print(f"Generating {args.max_tokens} tokens (temp={args.temperature}, top_k={args.top_k})...\n")
    print("-" * 60)
    print(args.prompt, end="")

    output = generate(
        model, tokenizer, args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    print(output)
    print("-" * 60)


if __name__ == "__main__":
    main()
