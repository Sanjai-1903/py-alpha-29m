"""Locked configuration for the encoder-decoder Python SLM.

Single source of truth -- prepare.py, corruption.py and train.py all read it.

    params = 2*V*d + enc_layers*12d^2 + dec_layers*16d^2
           = 5.24M  +  4.83M          +  19.41M          = 29.5M
"""
from dataclasses import dataclass, field

PAD, EOS, IR_OPEN, IR_CLOSE = "<pad>", "<eos>", "<ir>", "</ir>"
N_SENTINELS = 100


def special_tokens(n_sentinels: int = N_SENTINELS):
    """Order matters: these take the lowest ids, and span corruption assumes
    the sentinels are one contiguous run."""
    return [PAD, EOS, IR_OPEN, IR_CLOSE] + [f"<extra_id_{i}>" for i in range(n_sentinels)]


@dataclass
class Config:
    # tokeniser
    vocab_size: int = 8192
    n_sentinels: int = N_SENTINELS

    # shapes
    d_model: int = 320
    n_head: int = 10          # head_dim = 32
    d_ffn: int = 832          # ~8/3 * d_model, multiple of 64
    enc_layers: int = 4
    dec_layers: int = 12
    enc_seq_len: int = 512
    dec_seq_len: int = 1024

    # architecture choices
    norm: str = "layernorm"   # layernorm | rmsnorm
    norm_placement: str = "pre"
    activation: str = "swiglu"
    positions: str = "sinusoidal"

    # Where norms sit outside the blocks. With pre-norm the residual stream is
    # never normalised on its way up, so both of these earn their place:
    #   embed_norm  tames the embedding + sinusoid sum before the stack
    #   final_norm  before the output projection, and before the encoder's
    #               output becomes the K/V that every cross-attention reads
    embed_norm: bool = True
    final_norm: bool = True
    # Sinusoids live in [-1, 1] but embeddings init at std 0.02 -- without this
    # scaling the positional signal drowns out the token signal early on.
    scale_embeddings: bool = True
    tie_output: bool = True   # output projection tied to the DECODER embedding
    separate_embeddings: bool = True

    # stage 1: span corruption
    corruption_rate: float = 0.15
    mean_span_len: int = 6    # longer than T5's 3 -- code is more redundant

    # optimisation
    lr: float = 6e-4
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_frac: float = 0.02

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def est_params(self) -> int:
        d, v = self.d_model, self.vocab_size
        emb = 2 * v * d if self.separate_embeddings else v * d
        enc = self.enc_layers * (4 * d * d + 3 * d * self.d_ffn)
        dec = self.dec_layers * (8 * d * d + 3 * d * self.d_ffn)
        return emb + enc + dec


if __name__ == "__main__":
    c = Config()
    assert c.d_model % c.n_head == 0, "d_model must divide by n_head"
    print(f"head_dim        {c.head_dim}")
    print(f"estimated params {c.est_params():,}")
