"""Keep only stage 2 pairs whose code can actually run on its own.

    python filter_pairs.py --out-prefix stage2clean

86% of the mined pairs reference names that were never imported or defined --
`re`, `np`, `os`, project globals -- because extract_pairs() takes the function
source and discards the file's imports. Training on those teaches the model to
emit undefined references, which is exactly what it does.

This keeps the subset that is already self-contained, so the effect of that one
flaw can be measured without re-mining the corpus.
"""
import argparse
import ast
import builtins
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning)

ROOT = Path(__file__).parent
DATA = ROOT / "data"
BUILTINS = set(dir(builtins))


class Scope(ast.NodeVisitor):
    """Approximate scope analysis: which names are read but never bound?"""

    def __init__(self):
        self.bound, self.loaded = set(), set()

    def bind(self, node):
        if isinstance(node, ast.Name):
            self.bound.add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for e in node.elts:
                self.bind(e)
        elif isinstance(node, ast.Starred):
            self.bind(node.value)

    def visit_FunctionDef(self, n):
        self.bound.add(n.name)
        a = n.args
        for arg in list(a.args) + list(a.posonlyargs) + list(a.kwonlyargs):
            self.bound.add(arg.arg)
        if a.vararg:
            self.bound.add(a.vararg.arg)
        if a.kwarg:
            self.bound.add(a.kwarg.arg)
        self.generic_visit(n)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, n):
        self.bound.add(n.name)
        self.generic_visit(n)

    def visit_Assign(self, n):
        for t in n.targets:
            self.bind(t)
        self.generic_visit(n)

    def visit_AnnAssign(self, n):
        self.bind(n.target); self.generic_visit(n)

    def visit_AugAssign(self, n):
        self.bind(n.target); self.generic_visit(n)

    def visit_For(self, n):
        self.bind(n.target); self.generic_visit(n)

    visit_AsyncFor = visit_For

    def visit_comprehension(self, n):
        self.bind(n.target); self.generic_visit(n)

    def visit_withitem(self, n):
        if n.optional_vars:
            self.bind(n.optional_vars)
        self.generic_visit(n)

    def visit_ExceptHandler(self, n):
        if n.name:
            self.bound.add(n.name)
        self.generic_visit(n)

    def visit_Import(self, n):
        for a in n.names:
            self.bound.add((a.asname or a.name).split(".")[0])

    def visit_ImportFrom(self, n):
        for a in n.names:
            self.bound.add(a.asname or a.name)

    def visit_Global(self, n):
        self.bound.update(n.names)

    def visit_Nonlocal(self, n):
        self.bound.update(n.names)

    def visit_Name(self, n):
        if isinstance(n.ctx, ast.Load):
            self.loaded.add(n.id)


def undefined_names(code):
    """Names the snippet reads but never binds. None if it will not parse."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return None
    s = Scope()
    s.visit(tree)
    return s.loaded - s.bound - BUILTINS


def is_test(pair) -> bool:
    try:
        fn = ast.parse(pair["code"]).body[0]
    except Exception:
        return True
    return (fn.name.lower().startswith("test")
            or pair["request"].lower().startswith(("test", "tests")))


def filter_file(src: Path, dst: Path, max_lines: int):
    kept = dropped_undef = dropped_test = dropped_long = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            pair = json.loads(line)
            if is_test(pair):
                dropped_test += 1
                continue
            if len(pair["code"].splitlines()) > max_lines:
                dropped_long += 1
                continue
            u = undefined_names(pair["code"])
            if u is None or u:
                dropped_undef += 1
                continue
            fout.write(json.dumps(pair) + "\n")
            kept += 1
    total = kept + dropped_undef + dropped_test + dropped_long
    print(f"  {src.name} -> {dst.name}")
    print(f"      kept                {kept:7,}  ({kept/total:5.1%})")
    print(f"      undefined names     {dropped_undef:7,}")
    print(f"      test functions      {dropped_test:7,}")
    print(f"      over {max_lines} lines       {dropped_long:7,}")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default="stage2clean")
    ap.add_argument("--max-lines", type=int, default=25)
    args = ap.parse_args()

    for split in ("train", "val"):
        filter_file(DATA / f"stage2_{split}.jsonl",
                    DATA / f"{args.out_prefix}_{split}.jsonl",
                    args.max_lines)


if __name__ == "__main__":
    main()
