"""Build the training data for the encoder-decoder Python SLM.

    python prepare.py all --source hf --target-tokens 600000000
    python prepare.py all --source local --target-tokens 2000000   # offline smoke test

Stages:
  tokenizer  byte-level BPE, with sentinels + <ir> reserved up front
  stage1     raw Python packed into flat uint16 .bin files.
             Span corruption happens at TRAINING time (see corruption.py), so
             what is stored here is plain Python -- every epoch sees fresh masks.
  stage2     (request, code) pairs mined from functions that carry a docstring.

Both stages drop anything that n-gram-overlaps the eval sets, so the benchmark
numbers mean something.
"""
import argparse
import ast
import hashlib
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

# ast.parse() on scraped code emits SyntaxWarning for things like "\s" written
# without a raw string. The parse still succeeds and the file is still usable;
# the warnings are about the corpus, not about us, and there are millions of them.
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Redirected to a file, Python block-buffers stdout and a multi-hour run looks
# dead for ages. Line buffering costs nothing here and keeps `tail -f` honest.
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent))
from config import Config, EOS, special_tokens

ROOT = Path(__file__).parent
DATA = ROOT / "data"
VAL_EVERY = 200
NGRAM = 13


# ---------------------------------------------------------------- corpora

def iter_hf(limit=None):
    from datasets import load_dataset
    ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    for i, row in enumerate(ds):
        if limit and i >= limit:
            return
        yield row["content"]


def iter_local(limit=None):
    """Offline fallback: .py files from the Python installs on this machine."""
    seen, n = set(), 0
    for root in (Path(sys.prefix), Path(sys.base_prefix)):
        for path in root.rglob("*.py"):
            rp = path.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            try:
                yield rp.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            n += 1
            if limit and n >= limit:
                return


def get_iter(source, limit=None):
    return iter_local(limit) if source == "local" else iter_hf(limit)


# ---------------------------------------------------------------- filtering

def quality_ok(text: str) -> bool:
    """Drop minified, generated, and non-code files before they reach the model."""
    if not 200 <= len(text) <= 100_000:
        return False
    lines = text.splitlines()
    if len(lines) < 5:
        return False
    if max(len(ln) for ln in lines) > 400:      # minified or a giant data literal
        return False
    if len(text) / len(lines) > 120:            # mean line far too long
        return False
    if sum(c.isalpha() for c in text) / len(text) < 0.25:
        return False
    try:
        ast.parse(text)                         # must be real, parseable Python
    except (SyntaxError, ValueError, RecursionError):
        return False
    return True


class Dedup:
    """Exact-content dedup on a normalised hash. 8 bytes per document."""

    def __init__(self):
        self.seen = set()

    def is_new(self, text: str) -> bool:
        key = hashlib.sha256(" ".join(text.split()).encode()).digest()[:8]
        if key in self.seen:
            return False
        self.seen.add(key)
        return True


class Decontaminator:
    """Reject any document sharing a 13-gram with the eval prompts."""

    def __init__(self):
        self.ngrams = set()

    # Namespaced ids are mandatory from datasets 5.x; the bare names raise
    # HfUriError. Solutions are included alongside prompts -- a training file
    # containing MBPP's answer is contamination however the question is worded.
    EVAL_SOURCES = [
        ("openai/openai_humaneval", "test", ("prompt", "canonical_solution")),
        ("google-research-datasets/mbpp", "test", ("text", "code")),
        ("google-research-datasets/mbpp", "validation", ("text", "code")),
        ("google-research-datasets/mbpp", "prompt", ("text", "code")),
    ]

    def load(self):
        try:
            from datasets import load_dataset, disable_progress_bars
        except ImportError:
            raise SystemExit("decontamination needs `datasets` installed")
        disable_progress_bars()

        texts = []
        for name, split, fields in self.EVAL_SOURCES:
            try:
                ds = load_dataset(name, split=split)
            except Exception as e:
                print(f"  could not load {name}:{split} -- {type(e).__name__}: {e}")
                continue
            before = len(texts)
            for row in ds:
                texts.extend(row[f] for f in fields if row.get(f))
            print(f"  {name}:{split} -> {len(texts) - before} texts")

        for t in texts:
            for g in self._grams(t):
                self.ngrams.add(g)
        print(f"decontamination: {len(texts)} eval texts -> {len(self.ngrams):,} n-grams")
        if not self.ngrams:
            raise SystemExit(
                "decontamination loaded no eval prompts (offline?). Training on a\n"
                "contaminated corpus makes the benchmark meaningless, so this is fatal.\n"
                "Fix the network, or pass --no-decontam to accept that deliberately.")

    @staticmethod
    def _grams(text):
        words = text.split()
        for i in range(len(words) - NGRAM + 1):
            yield hash(" ".join(words[i:i + NGRAM]))

    def is_clean(self, text: str) -> bool:
        if not self.ngrams:
            return True
        return not any(g in self.ngrams for g in self._grams(text))


# ---------------------------------------------------------------- tokeniser

def train_tokenizer(source, cfg, n_docs, out_path):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

    specials = special_tokens(cfg.n_sentinels)
    tok = Tokenizer(models.BPE())
    # Byte-level keeps every byte representable and lets BPE learn the
    # multi-space tokens that make Python indentation cheap.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=cfg.vocab_size,
        special_tokens=specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    print(f"training BPE: vocab {cfg.vocab_size}, {len(specials)} reserved specials, "
          f"up to {n_docs:,} docs")

    def texts():
        for i, t in enumerate(get_iter(source, limit=n_docs)):
            if quality_ok(t):
                yield t

    tok.train_from_iterator(texts(), trainer=trainer)
    tok.save(str(out_path))

    # The sentinels must be one contiguous id range -- corruption.py relies on it.
    ids = [tok.token_to_id(f"<extra_id_{i}>") for i in range(cfg.n_sentinels)]
    assert ids == list(range(ids[0], ids[0] + cfg.n_sentinels)), "sentinels not contiguous"
    print(f"tokenizer -> {out_path}  (sentinels at ids {ids[0]}..{ids[-1]})")
    return tok


# ---------------------------------------------------------------- stage 1

def build_stage1(source, tok, cfg, target_tokens, decon):
    eos_id = tok.token_to_id(EOS)
    dedup = Dedup()
    train_f = open(DATA / "stage1_train.bin", "wb")
    val_f = open(DATA / "stage1_val.bin", "wb")
    n_train = n_val = kept = seen = 0

    try:
        for text in get_iter(source):
            seen += 1
            if not quality_ok(text) or not dedup.is_new(text) or not decon.is_clean(text):
                continue

            ids = tok.encode(text).ids + [eos_id]
            arr = np.asarray(ids, dtype=np.uint16)
            if kept % VAL_EVERY == 0:
                val_f.write(arr.tobytes()); n_val += arr.size
            else:
                train_f.write(arr.tobytes()); n_train += arr.size
            kept += 1

            if kept % 2000 == 0:
                print(f"  kept {kept:,}/{seen:,} docs  {n_train:,} tokens", flush=True)
            if n_train >= target_tokens:
                break
    finally:
        train_f.close(); val_f.close()

    print(f"stage1: {kept:,} docs kept of {seen:,} seen "
          f"({n_train:,} train / {n_val:,} val tokens)")
    return {"train_tokens": n_train, "val_tokens": n_val, "docs": kept, "seen": seen}


# ---------------------------------------------------------------- stage 2

SPHINX = re.compile(r"^\s*(:param|:return|:rtype|:raises|Args:|Returns:|Raises:|>>>)", re.M)


def clean_request(doc: str) -> str | None:
    """Keep the prose summary; drop parameter tables and doctests."""
    head = doc.split("\n\n")[0].strip()
    head = SPHINX.split(head)[0].strip()
    head = " ".join(head.split())
    if not 20 <= len(head) <= 300:
        return None
    if len(head.split()) < 4:
        return None
    if not re.match(r"^[A-Za-z]", head):
        return None
    return head


def strip_docstring(func_src: str, node, func_lineno: int) -> str:
    """Remove the docstring lines from a function's source text."""
    body = node.body
    if not body:
        return func_src
    first = body[0]
    is_doc = (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
              and isinstance(first.value.value, str))
    if not is_doc:
        return func_src
    lines = func_src.splitlines()
    start = first.lineno - func_lineno
    end = first.end_lineno - func_lineno
    return "\n".join(lines[:start] + lines[end + 1:])


def extract_pairs(source: str):
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return
    # Module-level defs only. Walking the whole tree also yields methods, whose
    # bodies call self._helpers that do not exist outside their class, and
    # closures, which read their enclosing scope. Neither can run standalone,
    # and standalone runnable code is exactly what we want the model to write.
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        first_args = [a.arg for a in node.args.args][:1]
        if first_args and first_args[0] in ("self", "cls"):
            continue
        doc = ast.get_docstring(node, clean=True)
        if not doc:
            continue
        request = clean_request(doc)
        if not request:
            continue
        seg = ast.get_source_segment(source, node)
        if not seg:
            continue
        code = strip_docstring(seg, node, node.lineno).rstrip()
        n_lines = len(code.splitlines())
        if not 2 <= n_lines <= 80 or len(code) > 4000:
            continue
        try:                       # must still parse once the docstring is gone
            ast.parse(code)
        except (SyntaxError, ValueError, IndentationError):
            continue
        yield {"request": request, "code": code}


def build_stage2(source, cfg, max_pairs, decon):
    dedup = Dedup()
    train_f = open(DATA / "stage2_train.jsonl", "w")
    val_f = open(DATA / "stage2_val.jsonl", "w")
    n_train = n_val = seen = 0

    try:
        for text in get_iter(source):
            if not quality_ok(text):
                continue
            for pair in extract_pairs(text):
                seen += 1
                blob = pair["request"] + "\n" + pair["code"]
                if not dedup.is_new(blob) or not decon.is_clean(blob):
                    continue
                line = json.dumps(pair) + "\n"
                total = n_train + n_val
                if total % VAL_EVERY == 0:
                    val_f.write(line); n_val += 1
                else:
                    train_f.write(line); n_train += 1
                if (n_train + n_val) % 5000 == 0:
                    print(f"  {n_train + n_val:,} pairs", flush=True)
            if n_train >= max_pairs:
                break
    finally:
        train_f.close(); val_f.close()

    print(f"stage2: {n_train:,} train / {n_val:,} val pairs (from {seen:,} candidates)")
    return {"train_pairs": n_train, "val_pairs": n_val}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tokenizer", "stage1", "stage2", "all"])
    ap.add_argument("--source", choices=["hf", "local"], default="hf")
    ap.add_argument("--target-tokens", type=int, default=600_000_000)
    ap.add_argument("--max-pairs", type=int, default=500_000)
    ap.add_argument("--tokenizer-docs", type=int, default=50_000)
    ap.add_argument("--retrain-tokenizer", action="store_true")
    ap.add_argument("--no-decontam", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    DATA.mkdir(exist_ok=True)
    tok_path = DATA / "tokenizer.json"
    meta_path = DATA / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    from tokenizers import Tokenizer
    if args.stage in ("tokenizer", "all"):
        if tok_path.exists() and not args.retrain_tokenizer:
            print(f"reusing {tok_path}")
            tok = Tokenizer.from_file(str(tok_path))
        else:
            tok = train_tokenizer(args.source, cfg, args.tokenizer_docs, tok_path)
    else:
        tok = Tokenizer.from_file(str(tok_path))

    sample = 'def add(a, b):\n    """Sum two numbers."""\n    return a + b\n'
    ids = tok.encode(sample).ids
    assert tok.decode(ids) == sample, "tokenizer does not round-trip"
    print(f"round-trip ok | {len(sample)/len(ids):.2f} chars/token")

    decon = Decontaminator()
    if not args.no_decontam:
        decon.load()

    if args.stage in ("stage1", "all"):
        meta.update(build_stage1(args.source, tok, cfg, args.target_tokens, decon))
    if args.stage in ("stage2", "all"):
        meta.update(build_stage2(args.source, cfg, args.max_pairs, decon))

    meta.update({"vocab_size": tok.get_vocab_size(),
                 "eos_id": tok.token_to_id(EOS),
                 "sentinel_start": tok.token_to_id("<extra_id_0>"),
                 "n_sentinels": cfg.n_sentinels,
                 "source": args.source})
    meta_path.write_text(json.dumps(meta, indent=2))
    print("\n" + json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
