"""Split each pair into (request + signature) -> body.

The model currently has to infer the function name, the parameter list and the
logic all at once, and it gets the first two roughly right and the third wrong.
Handing it the signature removes those first two jobs, so what is left measures
one thing only: can it write a correct body?

    python augment_pairs.py --src stage2clean --out-prefix stage2aug

Encoder input:                      Decoder target:
    Sort a matrix in ascending order     M.sort()
    def sort_matrix(M):                  return M
"""
import argparse
import ast
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning)

ROOT = Path(__file__).parent
DATA = ROOT / "data"


def split_signature(code: str):
    """-> (signature_lines, body_lines) or None if it cannot be split."""
    try:
        fn = ast.parse(code).body[0]
    except Exception:
        return None
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or not fn.body:
        return None

    lines = code.splitlines()
    # Body starts at the first statement; everything above it is the signature
    # (which may wrap across several lines for long parameter lists).
    first = fn.body[0]
    start = first.lineno - fn.lineno
    if start < 1 or start >= len(lines):
        return None
    sig = "\n".join(lines[:start])
    body = "\n".join(lines[start:])
    if not sig.strip().endswith(":") or not body.strip():
        return None
    return sig, body


def convert(src: Path, dst: Path):
    kept = skipped = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            p = json.loads(line)
            parts = split_signature(p["code"])
            if parts is None:
                skipped += 1
                continue
            sig, body = parts
            fout.write(json.dumps({
                "request": f"{p['request']}\n{sig}",
                "code": body,
                # keep the originals so the pair can be reassembled for eval
                "signature": sig,
                "plain_request": p["request"],
            }) + "\n")
            kept += 1
    print(f"  {src.name} -> {dst.name}:  kept {kept:,}, skipped {skipped:,}")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="stage2clean")
    ap.add_argument("--out-prefix", default="stage2aug")
    args = ap.parse_args()
    for split in ("train", "val"):
        convert(DATA / f"{args.src}_{split}.jsonl",
                DATA / f"{args.out_prefix}_{split}.jsonl")


if __name__ == "__main__":
    main()
