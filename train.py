"""Train the encoder-decoder SLM.

    python train.py --stage 1                 # span corruption on raw Python
    python train.py --stage 2 --init checkpoints/stage1_best.pt
    python train.py --stage 1 --overfit-batches 8   # sanity check: loss -> ~0

Stage 1 teaches the model Python by reconstructing masked spans. Stage 2 then
fine-tunes request -> code on the mined pairs. Same model, same loop; only
where the batches come from changes.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import Config, EOS, PAD
from corruption import make_batch, raw_window_for
from model import SLM

ROOT = Path(__file__).parent
DATA, CKPT, OUT = ROOT / "data", ROOT / "checkpoints", ROOT / "out"

sys.stdout.reconfigure(line_buffering=True)


def pick_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Stage1Batches:
    """Random windows of raw Python, masked fresh every time they are drawn."""

    def __init__(self, path, cfg, batch_size, sentinel_start, pad_id, device, eos_id=None):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.cfg, self.bs, self.device = cfg, batch_size, device
        self.sentinel_start, self.pad_id, self.eos_id = sentinel_start, pad_id, eos_id
        self.window = raw_window_for(cfg.enc_seq_len, cfg.corruption_rate, cfg.mean_span_len)
        if len(self.tokens) < self.window + 1:
            raise SystemExit(f"{path} has only {len(self.tokens)} tokens; run prepare.py")

    def __call__(self, rng):
        hi = len(self.tokens) - self.window - 1
        starts = rng.integers(0, hi, size=self.bs)
        windows = [self.tokens[s:s + self.window].astype(np.int64).tolist() for s in starts]
        arrays = make_batch(windows, self.cfg, self.sentinel_start, self.pad_id,
                            rng, self.eos_id)
        return tuple(torch.from_numpy(a).to(self.device) for a in arrays)


class Stage2Batches:
    """Mined (request, code) pairs. Encoder reads the request, decoder writes
    the code; loss falls on decoder tokens only."""

    # A batch pads to its longest member, so mixing a 20-token answer with a
    # 1200-token one wastes most of the decoder's compute on padding. Draw a
    # larger pool, sort it by length, and take one contiguous slice -- members
    # end up similar in length, which cuts padding without biasing the sample.
    POOL = 8

    def __init__(self, path, tok, cfg, batch_size, pad_id, eos_id, device):
        self.rows = [json.loads(l) for l in open(path)]
        self.tok, self.cfg, self.bs = tok, cfg, batch_size
        self.pad_id, self.eos_id, self.device = pad_id, eos_id, device
        if not self.rows:
            raise SystemExit(f"{path} is empty; run prepare.py stage2")
        # Character length is a good enough proxy for token length to sort by,
        # and avoids tokenising the pool twice.
        self.lengths = np.array([len(r["code"]) for r in self.rows])

    def __call__(self, rng):
        pool = rng.integers(0, len(self.rows), size=self.bs * self.POOL)
        pool = pool[np.argsort(self.lengths[pool])]
        offset = rng.integers(0, len(pool) - self.bs + 1)
        picks = pool[offset:offset + self.bs]
        reqs = [self.rows[i]["request"] for i in picks]
        codes = [self.rows[i]["code"] for i in picks]

        enc = [e.ids[: self.cfg.enc_seq_len] for e in self.tok.encode_batch(reqs)]
        dec = [e.ids[: self.cfg.dec_seq_len - 1] + [self.eos_id]
               for e in self.tok.encode_batch(codes)]

        be, bd = max(len(x) for x in enc), max(len(x) for x in dec)
        n = len(enc)
        enc_input = np.full((n, be), self.pad_id, dtype=np.int64)
        enc_mask = np.zeros((n, be), dtype=bool)
        dec_input = np.full((n, bd), self.pad_id, dtype=np.int64)
        dec_target = np.full((n, bd), -100, dtype=np.int64)

        for i, (e, d) in enumerate(zip(enc, dec)):
            enc_input[i, :len(e)] = e
            enc_mask[i, :len(e)] = True
            dec_input[i, 1:len(d)] = d[:-1]      # shift right, start from pad
            dec_target[i, :len(d)] = d

        return tuple(torch.from_numpy(a).to(self.device)
                     for a in (enc_input, enc_mask, dec_input, dec_target))


def lr_at(step, total, peak, warmup_frac, floor_frac=0.1):
    warmup = max(1, int(total * warmup_frac))
    if step < warmup:
        return peak * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))


@torch.no_grad()
def evaluate(model, batches, iters, ctx):
    model.eval()
    rng = np.random.default_rng(1234)          # same batches every eval
    total = 0.0
    for _ in range(iters):
        ei, em, di, dt = batches(rng)
        with ctx:
            _, loss = model(ei, di, em, dt)
        total += loss.item()
    model.train()
    return total / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2], default=1)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--eval-every", type=int, default=500)
    # Stage 2 steps are several times slower than stage 1's, so the default of
    # 20 leaves ~90s of silence that reads as a hang. Default by stage below.
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--eval-iters", type=int, default=40)
    # Decoder attention costs batch x heads x len^2 per layer, so one long
    # sample in a batch of 32 can allocate tens of GB across 12 layers. Capping
    # length bounds that; p95 of the mined code is ~495 tokens, so a cap of 384
    # truncates a small tail rather than losing examples.
    ap.add_argument("--dec-seq-len", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--overfit-batches", type=int, default=0)
    ap.add_argument("--init", default=None, help="checkpoint to warm-start from")
    ap.add_argument("--data-prefix", default="stage2",
                    help="reads data/<prefix>_{train,val}.jsonl (stage 2 only)")
    ap.add_argument("--tag", default=None,
                    help="checkpoint/log name; defaults to stage<N>")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    CKPT.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    cfg = Config()
    if args.dec_seq_len:
        cfg.dec_seq_len = args.dec_seq_len
    meta = json.loads((DATA / "meta.json").read_text())

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    pad_id, eos_id = tok.token_to_id(PAD), tok.token_to_id(EOS)
    sentinel_start = tok.token_to_id("<extra_id_0>")

    device = pick_device(args.device)
    dtype = args.dtype or ("bf16" if device == "cuda" else "fp32")
    ctx = (torch.autocast(device_type=device, dtype=torch.bfloat16)
           if dtype == "bf16" else torch.autocast(device_type=device, enabled=False))

    defaults = {1: (16_000, cfg.lr, 20), 2: (4_000, cfg.lr / 3, 5)}
    steps = args.steps or defaults[args.stage][0]
    lr = args.lr or defaults[args.stage][1]
    log_every = args.log_every or defaults[args.stage][2]

    model = SLM(cfg).to(device)
    if args.init:
        ck = torch.load(args.init, map_location=device)
        model.load_state_dict(ck["model"])
        print(f"warm-started from {args.init} (step {ck['step']})")

    opt = model.configure_optimizer(lr, cfg.weight_decay, cfg.betas, device)

    start_step = 0
    tag = args.tag or f"stage{args.stage}"
    if args.resume:
        ck = torch.load(CKPT / f"{tag}_latest.pt", map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        start_step = ck["step"] + 1
        print(f"resumed at step {start_step}")

    if args.stage == 1:
        train_b = Stage1Batches(DATA / "stage1_train.bin", cfg, args.batch_size,
                                sentinel_start, pad_id, device, eos_id)
        val_b = Stage1Batches(DATA / "stage1_val.bin", cfg, args.batch_size,
                              sentinel_start, pad_id, device, eos_id)
    else:
        train_b = Stage2Batches(DATA / f"{args.data_prefix}_train.jsonl", tok, cfg,
                                args.batch_size, pad_id, eos_id, device)
        val_b = Stage2Batches(DATA / f"{args.data_prefix}_val.jsonl", tok, cfg,
                              args.batch_size, pad_id, eos_id, device)

    print(f"device {device} | dtype {dtype} | params {model.num_params():,}")
    print(f"stage {args.stage} | {steps:,} steps x {args.grad_accum} accum "
          f"x batch {args.batch_size} | peak lr {lr:.2e}")

    if args.compile:
        model = torch.compile(model)

    rng = np.random.default_rng(0)
    fixed = None
    if args.overfit_batches:
        g = np.random.default_rng(0)
        fixed = [train_b(g) for _ in range(args.overfit_batches)]
        steps, args.eval_every = min(steps, 400), 50
        print(f"OVERFIT MODE: {args.overfit_batches} fixed batches, {steps} steps")

    log = open(OUT / f"train_{tag}.jsonl", "a")
    best, t0, seen_tokens = float("inf"), time.time(), 0
    model.train()

    for step in range(start_step, steps):
        cur_lr = lr_at(step, steps, lr, cfg.warmup_frac)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        opt.zero_grad(set_to_none=True)
        running = 0.0
        for micro in range(args.grad_accum):
            ei, em, di, dt = fixed[micro % len(fixed)] if fixed else train_b(rng)
            with ctx:
                _, loss = model(ei, di, em, dt)
            (loss / args.grad_accum).backward()
            running += loss.item() / args.grad_accum
            # Real tokens actually processed: encoder positions that are not
            # padding, plus decoder positions the loss is computed on.
            seen_tokens += int(em.sum().item() + (dt != -100).sum().item())

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % log_every == 0:
            el = time.time() - t0
            print(f"step {step:6d} | loss {running:.4f} | lr {cur_lr:.2e} | "
                  f"gnorm {norm:.2f} | {seen_tokens/max(el,1e-9):,.0f} tok/s | {el:.0f}s")

        if (step > 0 and step % args.eval_every == 0) or step == steps - 1:
            if fixed:
                print(f"  [overfit] loss {running:.4f} -- must trend to ~0")
                continue
            vl = evaluate(model, val_b, args.eval_iters, ctx)
            print(f"  EVAL step {step} | val_loss {vl:.4f} | ppl {math.exp(min(vl,20)):.1f}")
            log.write(json.dumps({"step": step, "train_loss": running, "val_loss": vl}) + "\n")
            log.flush()
            ck = {"model": model.state_dict(), "optimizer": opt.state_dict(),
                  "step": step, "config": cfg.__dict__, "val_loss": vl, "stage": args.stage}
            torch.save(ck, CKPT / f"{tag}_latest.pt")
            if vl < best:
                best = vl
                torch.save(ck, CKPT / f"{tag}_best.pt")
                print(f"  new best {vl:.4f} -> checkpoints/{tag}_best.pt")

    log.close()
    print(f"done in {(time.time()-t0)/60:.1f} min | best val_loss {best:.4f}")


if __name__ == "__main__":
    main()
