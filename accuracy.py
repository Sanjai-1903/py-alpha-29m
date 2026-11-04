"""Measure span-reconstruction accuracy for a stage 1 checkpoint.

    python accuracy.py
    python accuracy.py --checkpoint checkpoints/stage1_latest.pt --batches 20

Accuracy cannot be derived from loss -- two models with identical loss can have
very different accuracy, because loss weighs the whole probability distribution
while accuracy only asks which token ranked first. So it gets measured directly.

Measured under teacher forcing: each position is scored given the true preceding
tokens, so one early mistake does not poison everything after it.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from config import Config, EOS, PAD
from model import SLM
from train import Stage1Batches

ROOT = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/stage1_best.pt")
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(ROOT / args.checkpoint, map_location=device)
    cfg = Config(**ck["config"])
    model = SLM(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    tok = Tokenizer.from_file(str(ROOT / "data" / "tokenizer.json"))
    batches = Stage1Batches(ROOT / "data" / "stage1_val.bin", cfg, args.batch_size,
                            tok.token_to_id("<extra_id_0>"), tok.token_to_id(PAD),
                            device, tok.token_to_id(EOS))

    rng = np.random.default_rng(args.seed)
    hit = top5 = tot = 0
    nll = 0.0
    for _ in range(args.batches):
        ei, em, di, dt = batches(rng)
        with torch.no_grad():
            logits, loss = model(ei, di, em, dt)
        valid = dt != -100
        n = valid.sum().item()
        hit += (logits.argmax(-1)[valid] == dt[valid]).sum().item()
        k = logits.topk(5, -1).indices
        top5 += (k[valid] == dt[valid].unsqueeze(-1)).any(-1).sum().item()
        nll += loss.item() * n
        tot += n

    loss = nll / tot
    print(f"{args.checkpoint}  step {ck['step']}\n")
    print(f"  loss                          {loss:8.4f}")
    print(f"  perplexity                    {np.exp(loss):8.2f}")
    print(f"  avg prob on correct token     {np.exp(-loss):8.1%}   (= 1/ppl)")
    print(f"  exact accuracy (top-1)        {hit/tot:8.1%}   ({hit:,}/{tot:,} tokens)")
    print(f"  correct token in top-5        {top5/tot:8.1%}")


if __name__ == "__main__":
    main()
