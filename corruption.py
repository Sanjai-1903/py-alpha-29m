"""T5-style span corruption: the stage-1 pretraining objective.

An encoder-decoder cannot be pretrained on raw code the way a decoder-only
model can -- there is no natural split into "input" and "target". Span
corruption manufactures one: mask runs of tokens, show the encoder the holes,
make the decoder reconstruct what was removed.

    tokens    def add(a, b): return a + b
    encoder   def add(a, <X> return a <Y> b
    decoder   <X> b): <Y> + <Z>

Runs on every batch, so each epoch sees different masks over the same code.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import Config


def raw_window_for(enc_seq_len, rate, mean_span):
    """How many raw tokens to sample so the corrupted encoder input lands near
    enc_seq_len. Masking removes `rate` of them and adds one sentinel per span."""
    keep = (1.0 - rate) + rate / mean_span
    return int(enc_seq_len / keep)


def _segment(total, parts, rng):
    """Split `total` into `parts` positive integers, uniformly at random."""
    if parts <= 1:
        return [total]
    cuts = rng.choice(np.arange(1, total), size=parts - 1, replace=False)
    cuts.sort()
    bounds = np.concatenate([[0], cuts, [total]])
    return np.diff(bounds).tolist()


def corrupt(tokens, cfg: Config, sentinel_start: int, rng, eos_id=None):
    """One sequence -> (encoder_input, decoder_target), both lists of ids.

    eos_id closes the target. Without it the model is never shown where a
    reconstruction ends, so generation runs until it hits the token limit and
    degenerates into a repeated sentinel.
    """
    n = len(tokens)
    n_noise = max(1, round(n * cfg.corruption_rate))
    n_spans = max(1, round(n_noise / cfg.mean_span_len))
    # Every span needs a sentinel, a token to mask, and a gap to sit between.
    n_spans = min(n_spans, cfg.n_sentinels - 1, n_noise, n - n_noise - 1)
    if n_spans < 1:
        tail = [sentinel_start] + ([eos_id] if eos_id is not None else [])
        return list(tokens), tail

    span_lens = _segment(n_noise, n_spans, rng)
    gap_lens = _segment(n - n_noise, n_spans + 1, rng)

    enc, dec, pos = [], [], 0
    for i in range(n_spans):
        enc.extend(tokens[pos:pos + gap_lens[i]])
        pos += gap_lens[i]
        enc.append(sentinel_start + i)
        dec.append(sentinel_start + i)
        dec.extend(tokens[pos:pos + span_lens[i]])
        pos += span_lens[i]
    enc.extend(tokens[pos:])
    dec.append(sentinel_start + n_spans)     # closing sentinel
    if eos_id is not None:
        dec.append(eos_id)                   # ...then stop
    return enc, dec


def uncorrupt(enc, dec, sentinel_start, n_sentinels):
    """Inverse of corrupt(). Only used by the self-test -- if this cannot
    rebuild the original, the masking logic is wrong."""
    lo, hi = sentinel_start, sentinel_start + n_sentinels

    def split(seq):
        chunks, cur = [], []
        for t in seq:
            if lo <= t < hi:
                chunks.append(cur); cur = []
            else:
                cur.append(t)
        chunks.append(cur)
        return chunks

    gaps = split(enc)             # n_spans + 1 chunks
    spans = split(dec)[1:]        # leading chunk is empty (dec starts on a sentinel)
    out = []
    for i, gap in enumerate(gaps):
        out.extend(gap)
        if i < len(spans):
            out.extend(spans[i])
    return out


def make_batch(windows, cfg: Config, sentinel_start, pad_id, rng, eos_id=None):
    """Corrupt a list of token windows and pad into rectangular arrays.

    Returns enc_input, enc_mask, dec_input, dec_target.
    dec_input is the target shifted right by one and started with pad, which is
    the teacher-forcing convention: position t predicts target[t].
    """
    encs, decs = [], []
    for w in windows:
        e, d = corrupt(w, cfg, sentinel_start, rng, eos_id)
        encs.append(e[:cfg.enc_seq_len])
        decs.append(d[:cfg.dec_seq_len])

    be, bd = max(len(e) for e in encs), max(len(d) for d in decs)
    n = len(windows)
    enc_input = np.full((n, be), pad_id, dtype=np.int64)
    enc_mask = np.zeros((n, be), dtype=bool)
    dec_input = np.full((n, bd), pad_id, dtype=np.int64)
    dec_target = np.full((n, bd), -100, dtype=np.int64)   # -100 = ignored by the loss

    for i, (e, d) in enumerate(zip(encs, decs)):
        enc_input[i, :len(e)] = e
        enc_mask[i, :len(e)] = True
        dec_input[i, 0] = pad_id
        dec_input[i, 1:len(d)] = d[:-1]
        dec_target[i, :len(d)] = d
    return enc_input, enc_mask, dec_input, dec_target


if __name__ == "__main__":
    cfg = Config()
    rng = np.random.default_rng(0)
    sentinel_start, pad_id = 4, 0

    window = raw_window_for(cfg.enc_seq_len, cfg.corruption_rate, cfg.mean_span_len)
    print(f"raw window {window} -> encoder input ~{cfg.enc_seq_len}")

    eos_id = 1
    enc_lens, dec_lens = [], []
    for _ in range(200):
        toks = rng.integers(200, cfg.vocab_size, size=window).tolist()
        e, d = corrupt(toks, cfg, sentinel_start, rng, eos_id)
        assert d[-1] == eos_id, "target does not end with EOS"
        # Drop the EOS before inverting; everything before it must rebuild the input.
        assert uncorrupt(e, d[:-1], sentinel_start, cfg.n_sentinels) == toks, "round-trip failed"
        enc_lens.append(len(e)); dec_lens.append(len(d))
    print(f"round-trip ok over 200 samples, every target EOS-terminated")
    print(f"encoder len {np.mean(enc_lens):.0f} +/- {np.std(enc_lens):.1f}")
    print(f"decoder len {np.mean(dec_lens):.0f} +/- {np.std(dec_lens):.1f}")

    batch = [rng.integers(200, cfg.vocab_size, size=window).tolist() for _ in range(4)]
    ei, em, di, dt = make_batch(batch, cfg, sentinel_start, pad_id, rng, eos_id)
    print(f"batch shapes enc {ei.shape} mask {em.shape} dec_in {di.shape} dec_tgt {dt.shape}")
    assert (dt[:, 0] != -100).all() and (di[:, 0] == pad_id).all()
    print("ok")
