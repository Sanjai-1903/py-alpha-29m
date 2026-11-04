---
license: mit
language:
  - en
tags:
  - code
  - python
  - code-generation
  - instruction-tuning
task_categories:
  - text2text-generation
size_categories:
  - 100K<n<1M
---

# py-alpha-29m training pairs

`(request, code)` pairs mined from Python docstrings, used to fine-tune
[py-alpha-29m](https://huggingface.co/sanjaimurugan/py-alpha-29m). Three
variants, from raw to progressively filtered — the differences between them are
the useful part, because each was measured.

| file | rows | size | what it is |
|---|---|---|---|
| `stage2_train.jsonl` | 500,000 | 354 MB | raw mined pairs |
| `stage2clean_train.jsonl` | 54,860 | 19 MB | self-contained, ≤25 lines, no test functions |
| `stage2aug_train.jsonl` | 51,201 | 24 MB | clean, split into (request + signature) → body |

Validation splits ship alongside each, 1 row in 200.

## Format

```json
{"request": "Read file and return its contents.",
 "code": "def read_file(file_name):\n    with open(file_name, 'r') as f:\n        return f.read()"}
```

The augmented variant moves the signature into the input and keeps only the
body as the target, plus `signature` and `plain_request` so pairs can be
reassembled:

```json
{"request": "Read file and return its contents.\ndef read_file(file_name):",
 "code": "    with open(file_name, 'r') as f:\n        return f.read()"}
```

## How it was built

Every module-level function with a docstring in
[`codeparrot/codeparrot-clean`](https://huggingface.co/datasets/codeparrot/codeparrot-clean)
was extracted: the docstring's first prose paragraph becomes the request, the
function with its docstring removed becomes the code. Methods and closures are
excluded — a body calling `self._helper` cannot run on its own.

Source files were filtered (parseable, ≤400-char lines, ≥25% alphabetic),
deduplicated by content hash, and **decontaminated against HumanEval and MBPP**
— both prompts and reference solutions — by 13-gram overlap.

## Known flaws, measured

An audit of 40,000 raw pairs found:

| | share |
|---|---|
| reference names never imported or defined (`os`, `np`, `re`, project globals) | **86.6%** |
| test functions | 10.3% |
| median body length | 10 lines |

`stage2clean` removes all three. In a controlled comparison this raised the
model's parse rate from 83% to 90%; it did not change pass@1.

**The deeper limitation is register.** These are library internals —
`cpython_tags()`, `_IsTestMethod()` — because nobody writes a docstring on
`def reverse(s): return s[::-1]`. The corpus contains almost no tutorial-style
code, and no filtering creates it. If you need simple instructional examples,
this dataset will not supply them.

## Provenance

Derived from `codeparrot/codeparrot-clean`, itself deduplicated public Python
from GitHub. The extraction code is MIT licensed and lives at
[github.com/Sanjai-1903/py-alpha-29m](https://github.com/Sanjai-1903/py-alpha-29m)
(`prepare.py`, `filter_pairs.py`, `augment_pairs.py`). Licensing of the
underlying code is governed by the upstream dataset and the original
repositories, not by this card.

The stage 1 pretraining corpus (600M tokens) is not published here: it is a flat
`uint16` array meaningful only with this project's exact tokeniser, and
`prepare.py` regenerates it in about 20 minutes.
