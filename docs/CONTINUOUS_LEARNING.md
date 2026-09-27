# Continuous learning + experiment/data logging

A minor, CPU-friendly "train a little every night" loop, plus the tracking that makes it sane.

## Logging: state of the art, and what we use (and why not MLflow)

| tool | what it's good at | why not here |
|---|---|---|
| **MLflow** | runs/metrics/params + model registry, self-host | heavy; **duplicates artifacts** (each run copies the model/data); SQL+filestore overhead |
| **Weights & Biases** | best-in-class UI, sweeps | SaaS + account; overkill for a 1M model on CPU |
| **Aim** | fast local, many-run UI, open source | great option if you want a UI; still a daemon/extra dep |
| **ClearML / Neptune** | full MLOps / SaaS | heavy for a hobby cron job |
| **DVC / Git-LFS / HF Hub** | **data + model versioning by content hash** | this is the key idea — we borrow it |

The duplication you flagged is the real problem: nightly runs that each copy the weights/dataset
blow up storage. The **state-of-the-art trick to avoid it is content-addressed storage (CAS)**:
hash the bytes (sha256), store each unique blob *once*, and have runs reference the hash. That's
exactly what Git, DVC, Git-LFS, and the HF Hub do under the hood.

So `localagent/track.py` is deliberately tiny:
- **SQLite** (`runs.db`) for run / metric / param / artifact **metadata** — queryable, no server.
- **CAS** (`cas/<sha256>`) for **model + dataset artifacts** — deduped automatically.

Two identical checkpoints across 30 runs cost **one** copy, not 30 (`tracker.summary()` reports
`dedup_saved_rows`). Swap in MLflow/W&B/Aim later for a UI — the interface is 6 methods.

## The nightly loop (`scripts/cron_train.py`)

Each invocation does exactly one bounded chunk and exits (cron-friendly):

1. **Resume** model + optimizer + step from the tracker's latest `state` artifact (or cold-start).
2. **Grow the dataset**: load the persisted `dataset` pool, append fresh synthetic samples, save.
3. **Train** `--steps` steps with the *resumed* optimizer state (true continuation, not restart).
4. **Eval** held-out next-byte top-1; `log_metric`.
5. **Persist** new state + dataset as **content-addressed** artifacts; `end_run`.

Demonstrated over 3 runs (resume + growth visible):
```
run 1: steps   0->60   pool=300  held_out_top1=12.1%   (cold start)
run 2: steps  60->120  pool=551  held_out_top1=24.1%   (resumed)
run 3: steps 120->180  pool=780  held_out_top1=53.3%   (resumed)
```

## Run it

```bash
python scripts/cron_train.py --steps 80              # one nightly chunk
# crontab -e  (every day at 02:00, logs appended):
0 2 * * *  cd /path/to/LocalAgent && /usr/bin/python scripts/cron_train.py --steps 80 >> runs/cron.log 2>&1
```

Inspect runs/metrics with plain SQL:
```bash
sqlite3 runs/track/runs.db "SELECT run_id, step, key, value FROM metrics ORDER BY ts;"
```

## Notes
- This is the "minor" continuous-learning function — it trains the LM on the growing pool and
  resumes cleanly. The full flywheel (tool/pointer heads, grounded eval, GRPO) lives in
  `scripts/flywheel.py` / `analyze_loop.py`; point the cron at those once you have a GPU.
- Artifacts (`runs/`) are git-ignored — the CAS is your local model/data store, not the repo.
