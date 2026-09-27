# The three training stages

Training is three stages on one spine. Each answers a different question, and each is only worth
running if the one before it succeeded.

```
pretrain  ──▶  midtrain  ──▶  posttrain  ──▶  eval
   │              │               │              │
 open text    agentic         supervised     ten public
 corpora      trajectories    fine-tuning    benchmarks
```

`openlocalagent.stages.Stage` is the vocabulary. Every surface — the CLI's `train` choices, the
`stage:` key in a config, the run directory name, the `stage` field in each curve record — uses
these names.

| Stage | Question it answers | Data | Metric that tells you it worked |
|---|---|---|---|
| `pretrain` | Can it model language at all? | open text corpora | loss / bits-per-byte falling |
| `midtrain` | Does it know what an agent does? | open agentic data the harness does not score | loss on held-out general text |
| `posttrain` | Does it emit the exact call asked for? | the benchmarks' own train splits and related synthetic data | step-success on the ten suites |

Midtrain is about **capability**, posttrain about **exactness**. Keeping them separate is what lets
midtrain read broad, messy, real agentic data without that messiness reaching the output format.

### What separates midtrain from posttrain

One question: **does this corpus correspond to a suite the ten-benchmark harness scores?** Yes puts
it in posttrain, no puts it in midtrain. The two are disjoint by construction, so midtrain has no
access to the benchmarks and whatever it buys has to show up as transfer.

| | Rows | Tokens |
|---|---|---|
| midtrain — agentic, unscored | 62,891 | 62.9 M |
| posttrain — benchmark splits + related synthetic | 199,383 | 210.0 M |

Compute follows **pretrain > midtrain > posttrain**, with posttrain making roughly a single pass
over the benchmark data rather than drilling it. Sizes and the per-tier budget table are in
[DATASETS.md](DATASETS.md).

### Why midtrain holds out general text only

Midtrain's corpora are either in its own mixture or in posttrain's, so an agent holdout here would
measure contamination rather than skill. Its held-out signal is LM loss on general text; the
agent-side signal comes from posttrain's held-out split and from the ten-benchmark suite.

`distill`, `rl` and `eval` hang off that spine rather than extending it.

## Running a stage

```bash
openlocalagent train pretrain  configs/pretrain/la-93m.yaml
openlocalagent train midtrain  configs/midtrain/la-93m.yaml
openlocalagent train posttrain configs/posttrain/la-93m.yaml
```

On the training box, one GPU at a time, detached:

```bash
scripts/train_tier.sh 0 pretrain 93m               # one job
scripts/train_queue.sh 0 pretrain:93m midtrain:93m posttrain:93m   # the whole spine
```

The queue runs jobs back to back and stops at the first failure, so a stage never trains from a
checkpoint the stage before it failed to finish.

## Resuming

Stage configs set `runtime.resume: true`, which continues an existing `latest.pt` and starts clean
when there is none. Re-running a killed job therefore picks up where it stopped. The CLI's
`--resume` flag is the strict form: it *requires* the checkpoint and fails when it is absent, which
is what you want when verifying a resume and not what you want in a launcher.

## Loss curves

Every stage appends one record per step to `<out_dir>/curve.jsonl` while it trains — not at the
end, so a live run and a crashed one both have a curve. Read it with
`openlocalagent.train.curve.read_curve`, which returns typed `CurvePoint` records.

```python
from openlocalagent.train.curve import read_curve
points = read_curve("results/runs/pretrain-la-93m")
print(points[-1].step, points[-1].loss, points[-1].learning_rate)
```

The file is append-only, so a resumed run continues it and re-reading gives the whole history
across the resume.

## The stage a checkpoint records

`Stage.POSTTRAIN.checkpoint_stage` is `"sft"`. Checkpoints and lineage receipts written before the
rename carry that value and the resume checks compare against it, so the artifact format is
deliberately left behind the vocabulary rather than invalidating existing runs. `Stage.parse`
accepts `"sft"` wherever a stage name is read.
