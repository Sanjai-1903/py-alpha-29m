"""Strip a training checkpoint down to publishable weights.

    python release.py --checkpoint checkpoints/stage2aug_best.pt --name stage2aug

A training checkpoint stores the model plus AdamW's two momentum tensors per
parameter -- three copies in all, which is why 29.5M parameters occupy 338 MB.
Nobody loading the model needs the optimizer state, so this keeps the weights
and the config and nothing else.
"""
import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).parent
OUT = ROOT / "release"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", required=True, help="output stem, e.g. stage2aug")
    ap.add_argument("--fp16", action="store_true", help="halve the size again")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    ck = torch.load(ROOT / args.checkpoint, map_location="cpu")
    state = ck["model"]
    if args.fp16:
        state = {k: (v.half() if v.is_floating_point() else v) for k, v in state.items()}

    dst = OUT / f"{args.name}.pt"
    torch.save({"model": state, "config": ck["config"],
                "stage": ck.get("stage"), "step": ck["step"],
                "val_loss": ck.get("val_loss")}, dst)

    (OUT / f"{args.name}_config.json").write_text(json.dumps(ck["config"], indent=2))
    before = (ROOT / args.checkpoint).stat().st_size / 1048576
    after = dst.stat().st_size / 1048576
    n = sum(v.numel() for v in state.values())
    print(f"{args.checkpoint}  ->  release/{args.name}.pt")
    print(f"  {n:,} parameters | step {ck['step']} | val_loss {ck.get('val_loss', float('nan')):.4f}")
    print(f"  {before:.0f} MB -> {after:.0f} MB  ({'fp16' if args.fp16 else 'fp32'})")


if __name__ == "__main__":
    main()
