"""Encoder-decoder transformer for request -> Python code.

Built to the locked config: 4 bidirectional encoder layers, 12 causal decoder
layers each carrying self-attention, cross-attention and a SwiGLU feed-forward,
with pre-norm LayerNorm, sinusoidal positions and the output projection tied to
the decoder embedding. ~29.5M parameters.

The encoder runs once per request. Its output is the K/V that every one of the
12 cross-attention layers reads -- which is why it gets a final norm too.
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from config import Config


def sinusoidal_table(max_len: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    idx = torch.arange(0, d_model, 2, dtype=torch.float32)
    div = torch.exp(-math.log(10000.0) * idx / d_model)
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class Attention(nn.Module):
    """One multi-head block, used for encoder self, decoder self and cross.

    Cross-attention differs only in where K and V come from, so it is the same
    module with a different `mem` argument.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head, self.hd = cfg.n_head, cfg.head_dim
        d = cfg.d_model
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def _heads(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.n_head, self.hd).transpose(1, 2)

    def forward(self, x, mem=None, key_mask=None, causal=False):
        """x: (B, Lq, d). mem: (B, Lk, d) for cross-attention, else x.
        key_mask: (B, Lk) bool, True where the key is a real token."""
        src = x if mem is None else mem
        q = self._heads(self.q_proj(x))
        k = self._heads(self.k_proj(src))
        v = self._heads(self.v_proj(src))

        attn_mask = None
        if key_mask is not None:
            # (B, 1, 1, Lk) broadcasts over heads and query positions.
            attn_mask = key_mask[:, None, None, :]

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=causal)
        B, _, Lq, _ = y.shape
        return self.o_proj(y.transpose(1, 2).reshape(B, Lq, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ffn, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ffn, bias=False)
        self.down_proj = nn.Linear(cfg.d_ffn, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class EncoderBlock(nn.Module):
    """Two sublayers. Self-attention is bidirectional -- the whole point of
    having an encoder is that every request token sees every other."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.n1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.n2 = nn.LayerNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, key_mask):
        x = x + self.attn(self.n1(x), key_mask=key_mask, causal=False)
        return x + self.ffn(self.n2(x))


class DecoderBlock(nn.Module):
    """Three sublayers: self -> cross -> ffn, each with its own residual."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.n1 = nn.LayerNorm(cfg.d_model)
        self.self_attn = Attention(cfg)
        self.n2 = nn.LayerNorm(cfg.d_model)
        self.cross_attn = Attention(cfg)
        self.n3 = nn.LayerNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, mem, mem_mask):
        x = x + self.self_attn(self.n1(x), causal=True)
        x = x + self.cross_attn(self.n2(x), mem=mem, key_mask=mem_mask)
        return x + self.ffn(self.n3(x))


class SLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.enc_embed = nn.Embedding(cfg.vocab_size, d)
        self.dec_embed = nn.Embedding(cfg.vocab_size, d)
        self.enc_embed_norm = nn.LayerNorm(d) if cfg.embed_norm else nn.Identity()
        self.dec_embed_norm = nn.LayerNorm(d) if cfg.embed_norm else nn.Identity()

        self.encoder = nn.ModuleList(EncoderBlock(cfg) for _ in range(cfg.enc_layers))
        self.decoder = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.dec_layers))

        # Pre-norm leaves the residual stream unnormalised all the way up, so
        # both stacks need a final norm. The encoder's matters twice over: its
        # output is consumed by all 12 cross-attention layers.
        self.enc_norm_f = nn.LayerNorm(d) if cfg.final_norm else nn.Identity()
        self.dec_norm_f = nn.LayerNorm(d) if cfg.final_norm else nn.Identity()

        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        if cfg.tie_output:
            self.lm_head.weight = self.dec_embed.weight

        max_len = max(cfg.enc_seq_len, cfg.dec_seq_len)
        self.register_buffer("pos", sinusoidal_table(max_len, d), persistent=False)

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                depth = cfg.enc_layers if name.startswith("encoder") else cfg.dec_layers
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * depth))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def _embed(self, idx, table, norm):
        x = table(idx)
        if self.cfg.scale_embeddings:
            # Sinusoids live in [-1, 1] while embeddings init at std 0.02.
            # Without this the positional signal swamps the token signal.
            x = x * math.sqrt(self.cfg.d_model)
        x = x + self.pos[: idx.size(1)]
        return norm(x)

    def encode(self, enc_input, enc_mask=None):
        """Run the encoder once. The result is the frozen memory every
        cross-attention layer reads."""
        x = self._embed(enc_input, self.enc_embed, self.enc_embed_norm)
        for block in self.encoder:
            x = block(x, enc_mask)
        return self.enc_norm_f(x)

    def decode(self, dec_input, mem, mem_mask=None):
        x = self._embed(dec_input, self.dec_embed, self.dec_embed_norm)
        for block in self.decoder:
            x = block(x, mem, mem_mask)
        return self.lm_head(self.dec_norm_f(x))

    def forward(self, enc_input, dec_input, enc_mask=None, dec_target=None):
        mem = self.encode(enc_input, enc_mask)
        logits = self.decode(dec_input, mem, enc_mask)
        loss = None
        if dec_target is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                dec_target.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.enc_embed.weight.numel() + self.dec_embed.weight.numel()
        return n

    def configure_optimizer(self, lr, weight_decay, betas, device_type):
        """Decay the matrices; leave norms, biases and embeddings alone."""
        no_decay_ids = {id(self.enc_embed.weight), id(self.dec_embed.weight)}
        decay, no_decay, seen = [], [], set()
        for p in self.parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            (no_decay if (p.dim() < 2 or id(p) in no_decay_ids) else decay).append(p)
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8,
                                 fused=(device_type == "cuda"))

    @torch.no_grad()
    def generate(self, enc_input, enc_mask, start_id, max_new_tokens,
                 temperature=0.8, top_k=50, eos_id=None):
        """Encode once, then decode token by token against that fixed memory."""
        self.eval()
        mem = self.encode(enc_input, enc_mask)
        idx = torch.full((enc_input.size(0), 1), start_id,
                         dtype=torch.long, device=enc_input.device)
        for _ in range(max_new_tokens):
            logits = self.decode(idx[:, -self.cfg.dec_seq_len:], mem, enc_mask)
            logits = logits[:, -1, :].float()
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    kth = logits.topk(min(top_k, logits.size(-1)), -1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = Config()
    m = SLM(cfg).eval()

    print(f"total params      {m.num_params():,}")
    print(f"non-embedding     {m.num_params(non_embedding=True):,}")
    print(f"config estimate   {cfg.est_params():,}  (excludes LayerNorm gains/biases)")

    B, Le, Ld = 2, 48, 32
    enc = torch.randint(200, cfg.vocab_size, (B, Le))
    enc_mask = torch.ones(B, Le, dtype=torch.bool)
    enc_mask[1, 30:] = False                      # second row is padded
    dec = torch.randint(200, cfg.vocab_size, (B, Ld + 1))

    logits, loss = m(enc, dec[:, :-1], enc_mask, dec[:, 1:])
    expected = math.log(cfg.vocab_size)
    print(f"\nlogits {tuple(logits.shape)}  loss {loss.item():.4f}  expected ~{expected:.4f}")
    assert abs(loss.item() - expected) < 0.3, "init loss is off"

    # The decoder must not see the future: editing the last input token can only
    # move the last position's logits.
    alt = dec.clone()
    alt[:, Ld - 1] = (alt[:, Ld - 1] + 1) % cfg.vocab_size
    l2, _ = m(enc, alt[:, :-1], enc_mask)
    same = torch.allclose(logits[:, :-1], l2[:, :-1], atol=1e-5)
    print(f"causal mask holds              {same}")
    assert same, "decoder is attending to future tokens"

    # Padded encoder positions must be invisible to cross-attention.
    enc2 = enc.clone()
    enc2[1, 30:] = torch.randint(200, cfg.vocab_size, (Le - 30,))
    l3, _ = m(enc2, dec[:, :-1], enc_mask)
    pad_ok = torch.allclose(logits, l3, atol=1e-5)
    print(f"encoder padding ignored        {pad_ok}")
    assert pad_ok, "padding is leaking into cross-attention"

    out = m.generate(enc, enc_mask, start_id=0, max_new_tokens=5, temperature=0.0)
    print(f"greedy generate                {tuple(out.shape)}")
    print("ok")
