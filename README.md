# py-alpha-29m — a 29.5M encoder-decoder trained from scratch on Python

An encoder-decoder transformer built and trained end to end on one laptop: own
tokeniser, own corpus, own span-corruption pretraining, own evaluation harness.

It writes syntactically valid, idiomatic-looking Python. It almost never writes
*correct* Python. Both halves of that sentence are the point — the numbers below
are reported as measured, including the ones that didn't move.

```
parameters        29,521,920
architecture      4 encoder + 12 decoder blocks, cross-attention every layer
pretraining       600,001,082 tokens of deduplicated, decontaminated Python
fine-tuning       500,000 (request, code) pairs mined from docstrings
hardware          one Apple M4 Pro (MPS), ~30,000 tokens/sec
training time     5h 00m (stage 1) + 1h 20m (stage 2)
```

## Results

| | val loss | perplexity | token accuracy |
|---|---|---|---|
| Stage 1 — span corruption | 1.2153 | 3.31 | 74.6% exact, 89.0% top-5 |
| Stage 2 — request → code | 1.8659 | 6.5 | — |

| benchmark | pass@1 | parse rate |
|---|---|---|
| MBPP (100) | **0.0%** | 90.0% |
| MBPP (100), signature given | **0.0%** | 97.0% |
| HumanEval (60) | **1.7%** | 88.3% |

The single HumanEval pass is `strlen` — `return len(string)`. For scale,
Qwen2.5-Coder-0.5B scores 28.0 on HumanEval and 52.9 on MBPP with 16.7× the
parameters and 9,167× the training tokens.

## What the experiments established

Four interventions were tried. Every one that addressed **format** worked; none
moved **correctness**.

| intervention | effect on format | effect on pass@1 |
|---|---|---|
| Filter training pairs for length, drop test functions | parse 83% → 90% | none |
| Give the model the function signature | parse 90% → 97% | none |
| Richer prompts (HumanEval vs MBPP) | — | 0 → 1/60 |
| Best-of-10 sampling, temperatures 0.2–0.8 | degrades sharply | none |

Two findings worth stating directly:

**Conditioning is not the bottleneck.** Handed the function name and parameter
list — two of its three jobs removed — the model still scored 0%. It writes
`def sort_matrix(a0):` then a body with no relationship to sorting. The limit is
composing logic, not understanding the request.

**There is nothing hiding in the distribution.** Best-of-n never beat greedy at
any temperature, and parse rate collapses from 88% to 2% as temperature rises
from 0 to 0.8. The probability mass behind the argmax is noise, not plausible
alternatives. Greedy already extracts everything the model has.

## Architecture

Pre-norm LayerNorm, SwiGLU, sinusoidal positions, MHA, output projection tied to
the decoder embedding, separate encoder and decoder embeddings.

```
d_model 320 · n_head 10 (head_dim 32) · d_ffn 832
enc_layers 4 · dec_layers 12 · enc_seq_len 512 · dec_seq_len 1024
vocab 8,192 byte-level BPE + 100 sentinels
```

A note for anyone copying this: **42% of the parameters exist to read the
request** — encoder embedding (2.62M), encoder layers (4.83M) and cross-attention
(4.92M) — while stage 2 requests average 18 tokens. The encoder-decoder split was
chosen to learn cross-attention, not for efficiency. For short-request-to-long-code,
decoder-only would spend those parameters on generation instead.

## Loading the released weights

The model is a custom architecture, so `AutoModel` will not load it. Take
`model.py` and `config.py` from this repo alongside the checkpoint:

```python
import torch
from config import Config
from model import SLM

ck = torch.load("stage2aug.pt", map_location="cpu")
model = SLM(Config(**ck["config"]))
model.load_state_dict(ck["model"])
model.eval()
```

`stage1.pt` is the span-corruption model — judge it with
`sample.py --mode denoise`, since it has never seen a natural-language request.
`stage2aug.pt` writes a function body given a request and a signature.

## Pipeline

```bash
python prepare.py all --source hf --target-tokens 600000000   # tokeniser + corpus + pairs
python train.py --stage 1 --steps 16000 --batch-size 32 --grad-accum 2
python train.py --stage 2 --init checkpoints/stage1_best.pt --batch-size 16 --dec-seq-len 384
python eval.py --benchmark mbpp --limit 100
```

| file | role |
|---|---|
| `config.py` | single source of truth; prints the parameter count |
| `prepare.py` | BPE tokeniser, corpus packing, pair mining, decontamination |
| `corruption.py` | T5-style span corruption, with an invertibility self-test |
| `model.py` | the model; self-test checks init loss, causality, padding isolation |
| `train.py` | both stages, overfit mode, resume, length bucketing |
| `eval.py` | MBPP + HumanEval, sandboxed execution, best-of-n |
| `sample.py` | generation, in `code` and `denoise` modes |
| `accuracy.py` | token-level accuracy, which loss alone cannot give you |
| `filter_pairs.py` | scope analysis to find undefined references |
| `augment_pairs.py` | splits pairs into (request + signature) → body |

Every component has a self-test. `python model.py` verifies the initial loss sits
at `ln(vocab_size)`, that the decoder cannot see future tokens, and that encoder
padding is invisible to cross-attention. `python corruption.py` verifies masking
is exactly invertible.

## Data quality notes

An audit of 40,000 training pairs found **86.6% reference names that are never
imported or defined** — `os`, `np`, `re`, project globals — because the miner
takes the function body and discards the file's imports. A controlled experiment
showed this costs less than it appears: the model emits undefined references only
3.3% of the time on MBPP, since those prompts need no imports.

The deeper problem is register. Docstring-mined pairs are library plumbing —
`cpython_tags()`, `_IsTestMethod()` — because nobody documents
`def reverse(s): return s[::-1]`. The corpus contains no tutorial-style code, and
no amount of filtering creates it.

## Limitations

Not useful for writing code. Produces plausible-looking Python with incorrect
logic. Trained only on Python. Benchmarked on 100/500 MBPP and 60/164 HumanEval
problems — enough to distinguish 0% from 50%, not 0% from 1%.

## Data provenance and licence

Pretrained on [`codeparrot/codeparrot-clean`](https://huggingface.co/datasets/codeparrot/codeparrot-clean),
deduplicated public Python from GitHub. Training data was decontaminated against
HumanEval and MBPP (prompts *and* reference solutions) by 13-gram overlap before
any training, so the benchmark numbers above are not inflated by memorisation.

The code in this repository is MIT licensed. The provenance and licensing of the
training corpus is a separate matter, governed by the upstream dataset.
