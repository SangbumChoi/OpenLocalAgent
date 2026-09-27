#!/usr/bin/env python
"""Continuous-learning daemon — one bounded training chunk per invocation (run nightly by cron).

Each run: resume model+optimizer+step from the tracker (or init on the first run), append fresh
synthetic data to a persisted growing pool, train `--steps` steps, eval, log metrics, and save the
new state + dataset as content-addressed artifacts (deduped). Designed to be safe to run daily on
CPU and to pick up exactly where it left off.

  python scripts/cron_train.py --steps 80
  # crontab (every day at 02:00):
  #   0 2 * * *  cd /path/to/LocalAgent && /usr/bin/python scripts/cron_train.py --steps 80 >> runs/cron.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import os
import random

import torch

from localagent.data.agent_synth import Generator
from localagent.data.render import IGNORE, assistant_body, prompt_text
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.track import Tracker
from localagent.train.loop import cosine_lr, pad_batch, set_lr


def _row(rec, tok):
    p, b = tok.encode(rec["p"]), tok.encode(rec["b"]) + [tok.eos_id]
    return p + b, [IGNORE] * len(p) + b


@torch.no_grad()
def _eval_top1(model, recs, tok, device):
    model.eval()
    cor = ntok = 0
    for rec in recs:
        ids, labels = _row(rec, tok)
        x = torch.tensor([ids[:-1]], device=device)
        y = torch.tensor(labels[1:], device=device)
        m = y != IGNORE
        if m.sum() == 0:
            continue
        logits, _ = model(x)
        cor += (logits[0][m].argmax(-1) == y[m]).sum().item()
        ntok += int(m.sum())
    return cor / max(1, ntok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--root", default="runs/track")
    ap.add_argument("--per-run", type=int, default=400, help="new samples added each run")
    args = ap.parse_args()
    device = "cpu"
    tok = load_tokenizer("byte")
    tr = Tracker(args.root)
    tmp = os.path.join(args.root, "tmp"); os.makedirs(tmp, exist_ok=True)

    # 1) resume model + optimizer + step (or init)
    state_path = tr.latest_artifact("state")
    if state_path:
        ck = torch.load(state_path, map_location=device, weights_only=False)
        cfg = ModelConfig(**ck["cfg"]); model = LocalAgentLM(cfg).to(device)
        model.load_state_dict(ck["model"]); step0 = ck["step"]; run_no = ck["run_no"] + 1
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
        opt.load_state_dict(ck["opt"])
        print(f"resumed from step {step0} (run {run_no})", flush=True)
    else:
        cfg = ModelConfig.from_yaml("configs/model/ultra-tiny-1m.yaml")
        model = LocalAgentLM(cfg).to(device); step0 = 0; run_no = 1
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
        print("cold start (no prior state)", flush=True)

    # 2) grow the persisted dataset pool
    pool = []
    ds_path = tr.latest_artifact("dataset")
    if ds_path:
        pool = [json.loads(line) for line in open(ds_path)]
    fresh = Generator(level=min(5, run_no), seed=10_000 + run_no).generate(args.per_run)
    seen = {r["p"] for r in pool}
    for s in fresh:
        p = prompt_text(s)
        if p not in seen:
            pool.append({"p": p, "b": assistant_body(s)}); seen.add(p)
    held = [{"p": prompt_text(s), "b": assistant_body(s)}
            for s in Generator(level=5, seed=999, split="eval").generate_balanced(20)]

    # 3) train one bounded chunk (resumed optimizer)
    run_id = tr.start_run(f"cron-{run_no}", {"steps": args.steps, "pool": len(pool), "step0": step0})
    rows = [_row(r, tok) for r in pool]
    rng = random.Random(run_no)
    model.train()
    for i in range(args.steps):
        set_lr(opt, cosine_lr(i, args.steps, 1e-3, 10, 0.3))
        x, y = pad_batch([rng.choice(rows) for _ in range(32)], tok.pad_id, device)
        _, loss = model(x, targets=y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if i % max(1, args.steps // 4) == 0:
            tr.log_metric(run_id, step0 + i, "loss", loss.item())
    step1 = step0 + args.steps
    acc = _eval_top1(model, held, tok, device)
    tr.log_metric(run_id, step1, "held_out_top1", acc)

    # 4) persist new state + dataset as content-addressed artifacts (deduped)
    sp = os.path.join(tmp, "state.pt")
    torch.save({"cfg": cfg.__dict__, "model": model.state_dict(), "opt": opt.state_dict(),
                "step": step1, "run_no": run_no}, sp)
    tr.log_artifact(run_id, sp, "state")
    dp = os.path.join(tmp, "dataset.jsonl")
    with open(dp, "w") as f:
        for r in pool:
            f.write(json.dumps(r) + "\n")
    tr.log_artifact(run_id, dp, "dataset")
    tr.end_run(run_id)

    print(f"run {run_no}: steps {step0}->{step1}  pool={len(pool)}  held_out_top1={acc*100:.1f}%",
          flush=True)
    print("tracker:", tr.summary(), flush=True)


if __name__ == "__main__":
    main()
