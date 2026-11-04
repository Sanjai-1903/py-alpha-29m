"""Benchmark a checkpoint on MBPP and HumanEval.

    python eval.py --benchmark mbpp --limit 50
    python eval.py --benchmark both --checkpoint checkpoints/stage2_best.pt

Reports pass@1 (greedy) and parse rate. At 29.5M parameters expect pass@1 near
zero -- parse rate is the metric that will actually move, and it is the honest
measure of whether the model learned Python at all.

SAFETY: generated code is executed to check it. Each program runs in a separate
isolated interpreter with a timeout, which stops accidents and infinite loops.
It is not a security sandbox -- do not point this at a model trained on data
you do not trust.
"""
import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from config import Config, EOS, PAD
from model import SLM

ROOT = Path(__file__).parent


def valid_python(code: str) -> tuple[bool, bool]:
    """(parses, defines a function).

    An empty string parses cleanly -- ast.parse('') returns an empty body -- so
    a bare try/except around ast.parse scores silent models at 100%. Require
    the tree to actually contain something.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return False, False
    if not tree.body:
        return False, False
    has_fn = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 for n in ast.walk(tree))
    return True, has_fn


def run_program(src: str, timeout: float) -> bool:
    """True if the program runs to completion with exit status 0."""
    try:
        p = subprocess.run([sys.executable, "-I", "-c", src],
                           capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def mbpp_signature(test_list):
    """Recover `def name(a0, a1):` from MBPP's own assertions.

    Lets the model be given the signature instead of having to invent it, which
    separates "can it name a function" from "can it write the body".
    """
    for t in test_list:
        try:
            tree = ast.parse(t)
        except (SyntaxError, ValueError):
            continue
        call = None
        node = tree.body[0]
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
            left = node.test.left
            if isinstance(left, ast.Call) and isinstance(left.func, ast.Name):
                call = left
        if call is None:
            for n in ast.walk(tree):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                    call = n
                    break
        if call is not None:
            args = ", ".join(f"a{i}" for i in range(len(call.args)))
            return f"def {call.func.id}({args}):"
    return None


def load_problems(benchmark, limit):
    from datasets import load_dataset, disable_progress_bars
    disable_progress_bars()
    out = []
    if benchmark in ("mbpp", "both"):
        # MBPP is a natural fit: text is the request, test_list the assertions.
        for row in load_dataset("google-research-datasets/mbpp", split="test"):
            out.append({"bench": "mbpp", "request": row["text"],
                        "signature": mbpp_signature(row["test_list"]),
                        "suffix": "\n" + "\n".join(row["test_list"])})
    if benchmark in ("humaneval", "both"):
        # Pass the whole prompt so the model sees the required function name.
        for row in load_dataset("openai/openai_humaneval", split="test"):
            # The prompt already carries the signature and docstring, so for
            # --give-signature it IS the prefix: the model writes only the body.
            out.append({"bench": "humaneval", "request": row["prompt"],
                        "signature": row["prompt"].rstrip(),
                        "suffix": "\n" + row["test"] + f"\ncheck({row['entry_point']})"})
    if limit:
        keep = {}
        for p in out:
            keep.setdefault(p["bench"], []).append(p)
        out = [p for rows in keep.values() for p in rows[:limit]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["mbpp", "humaneval", "both"], default="both")
    ap.add_argument("--checkpoint", default="checkpoints/stage2_best.pt")
    ap.add_argument("--limit", type=int, default=0, help="problems per benchmark")
    ap.add_argument("--give-signature", action="store_true",
                    help="append the target signature to the request and have "
                         "the model generate only the body (MBPP only)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    # Greedy commits to one path and is prone to repetition loops. Drawing n
    # samples and keeping any that passes its tests measures whether a correct
    # answer exists in the distribution, even when it is not the argmax.
    ap.add_argument("--samples", type=int, default=1,
                    help="candidates per problem; >1 switches to sampling")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="used only when --samples > 1")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="out/eval.json")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(ROOT / args.checkpoint, map_location=device)
    cfg = Config(**ck["config"])
    model = SLM(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = Tokenizer.from_file(str(ROOT / "data" / "tokenizer.json"))
    pad_id, eos_id = tok.token_to_id(PAD), tok.token_to_id(EOS)
    print(f"{args.checkpoint} | stage {ck.get('stage','?')} | step {ck['step']}")

    problems = load_problems(args.benchmark, args.limit)
    print(f"{len(problems)} problems | device {device}\n")

    stats, records = {}, []
    skipped = 0
    for i, p in enumerate(problems):
        sig = p.get("signature") if args.give_signature else None
        if args.give_signature and p["bench"] == "mbpp" and not sig:
            skipped += 1          # no signature recoverable; excluded, not scored
            continue
        # For HumanEval the prompt and the signature are the same text, so it is
        # shown once; for MBPP the recovered `def` line is appended to the request.
        if sig and sig != p["request"].rstrip():
            request = f"{p['request']}\n{sig}"
        else:
            request = p["request"]
        enc = torch.tensor([tok.encode(request).ids[: cfg.enc_seq_len]], device=device)
        mask = torch.ones_like(enc, dtype=torch.bool)
        # All n candidates come from one batched call: the encoder input is
        # repeated and each row samples independently.
        n = max(1, args.samples)
        rows = []
        # Candidate 0 is always greedy, so best-of-n can never score below
        # pass@1. Sampling away from the argmax is strictly a bonus attempt.
        rows.append(model.generate(enc, mask, start_id=pad_id,
                                   max_new_tokens=args.max_new_tokens,
                                   temperature=0.0, top_k=0, eos_id=eos_id)[0])
        if n > 1:
            extra = model.generate(enc.repeat(n - 1, 1), mask.repeat(n - 1, 1),
                                   start_id=pad_id, max_new_tokens=args.max_new_tokens,
                                   temperature=args.temperature, top_k=args.top_k,
                                   eos_id=eos_id)
            rows.extend(extra)

        cands = []
        for row in rows:
            c = tok.decode(row.tolist(), skip_special_tokens=True)
            if sig:
                # the model wrote only the body; put the signature back on top
                c = sig + "\n" + c
            cands.append(c)

        # Report the first candidate's formatting, but let any candidate pass.
        parses, has_fn = valid_python(cands[0])
        passed = False
        for c in cands:
            ok, _ = valid_python(c)
            if ok and run_program(c + p["suffix"], args.timeout):
                passed = True
                code = c
                break
        else:
            code = cands[0]
        parses = parses or (passed and valid_python(code)[0])

        s = stats.setdefault(p["bench"], {"n": 0, "parses": 0, "has_fn": 0, "passed": 0})
        s["n"] += 1; s["parses"] += parses; s["has_fn"] += has_fn; s["passed"] += passed
        records.append({"bench": p["bench"], "parses": parses, "has_fn": has_fn,
                        "passed": passed, "code": code})

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(problems)}", flush=True)

    print()
    k = max(1, args.samples)
    label = "pass@1" if k == 1 else f"pass@{k}"
    for bench, s in stats.items():
        print(f"{bench:10s}  {label} {s['passed']/s['n']:6.1%}   "
              f"parse rate {s['parses']/s['n']:6.1%}   "
              f"defines a function {s['has_fn']/s['n']:6.1%}   ({s['n']} problems)")
    if k > 1:
        print(f"\n  note: pass@{k} means at least one of {k} sampled candidates passed.")
        print(f"  It is NOT comparable to a published pass@1 figure.")

    Path(ROOT / args.out).parent.mkdir(exist_ok=True)
    (ROOT / args.out).write_text(json.dumps({"stats": stats, "records": records}, indent=2))
    print(f"\nfull output -> {args.out}")


if __name__ == "__main__":
    main()
