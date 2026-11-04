"""Generate from a checkpoint.

    # stage 2 checkpoint: request -> code
    python sample.py --request "print the first n fibonacci numbers"

    # stage 1 checkpoint: mask a snippet and watch it reconstruct the spans
    python sample.py --mode denoise --checkpoint checkpoints/stage1_best.pt

A stage 1 model has only ever seen span corruption, so asking it for code will
produce noise. Use denoise mode to judge it -- that is the task it was trained on.
"""
import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from config import Config, EOS, PAD
from corruption import corrupt
from model import SLM

ROOT = Path(__file__).parent

SNIPPET = '''def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        print(a)
        a, b = b, a + b
'''


def load(checkpoint, device):
    ck = torch.load(ROOT / checkpoint, map_location=device)
    cfg = Config(**ck["config"])
    model = SLM(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = Tokenizer.from_file(str(ROOT / "data" / "tokenizer.json"))
    print(f"{checkpoint} | stage {ck.get('stage','?')} | step {ck['step']} "
          f"| val_loss {ck.get('val_loss', float('nan')):.4f}\n")
    return model, cfg, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["code", "denoise"], default="code")
    ap.add_argument("--request", default="print the first n fibonacci numbers")
    ap.add_argument("--snippet-file", default=None)
    ap.add_argument("--checkpoint", default="checkpoints/stage2_best.pt")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("-n", "--num-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="which validation window to mask")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    model, cfg, tok = load(args.checkpoint, device)
    pad_id, eos_id = tok.token_to_id(PAD), tok.token_to_id(EOS)

    if args.mode == "denoise":
        if args.snippet_file:
            ids = tok.encode(Path(args.snippet_file).read_text()).ids
        else:
            # A real validation window, not the toy snippet. The model trains on
            # ~585-token windows; corrupting 30 tokens is out of distribution and
            # makes a healthy model look like it degenerates.
            import numpy as _np
            from corruption import raw_window_for
            toks = _np.memmap(ROOT / "data" / "stage1_val.bin", dtype=_np.uint16, mode="r")
            w = raw_window_for(cfg.enc_seq_len, cfg.corruption_rate, cfg.mean_span_len)
            start = np.random.default_rng(args.seed).integers(0, len(toks) - w - 1)
            ids = toks[start:start + w].astype(int).tolist()
        enc_ids, target = corrupt(ids, cfg, tok.token_to_id("<extra_id_0>"),
                                  np.random.default_rng(0))
        print("MASKED INPUT:")
        print(tok.decode(enc_ids, skip_special_tokens=False))
        print("\nTRUE SPANS: ", tok.decode(target, skip_special_tokens=False))
        enc = torch.tensor([enc_ids[: cfg.enc_seq_len]], device=device)
    else:
        print(f"REQUEST: {args.request}")
        enc = torch.tensor([tok.encode(args.request).ids[: cfg.enc_seq_len]], device=device)

    mask = torch.ones_like(enc, dtype=torch.bool)
    for i in range(args.num_samples):
        out = model.generate(enc, mask, start_id=pad_id,
                             max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, top_k=args.top_k,
                             eos_id=eos_id)
        text = tok.decode(out[0].tolist(), skip_special_tokens=(args.mode == "code"))
        print(f"\n--- sample {i+1} " + "-" * 50)
        print(text)
        if args.mode == "code":
            # ast.parse('') succeeds, so an empty generation would otherwise
            # report as valid. Require the tree to hold something.
            try:
                tree = ast.parse(text)
                ok = bool(tree.body)
            except (SyntaxError, ValueError, RecursionError):
                ok = False
            print("[parses as valid Python]" if ok else "[does NOT parse / empty]")


if __name__ == "__main__":
    main()
