# Matched 34M pretraining pilot: 5 tokens/parameter

Status on 2026-07-29: **data gate verified; training not started**. Raw paper-v5 acquisition and the
four-suite external prompt-only denylist gate are complete. Paper-all preparation retained 504,010
documents and packed 523,358,082 training plus 5,311,528 validation tokens (528,669,610 total) with
a train-only 16K tokenizer. Independent freeze verification passed and reproduced canonical SHA-256
`b005ef46b2fd3c8db91a725ff5dca894d97448f2a8e174f7d7f36811dcdd2aa9`. The bounded audit of
19,334 supplied normalized denylist prompts made 8,633,077 candidate checks and removed 15
documents; its limitations are explicitly non-exhaustive. Every 5-TPP run remains unstarted. The
earlier 858,832-byte Mind2Web full-DOM v1 overflow remains historical; the v2 ranker-bound export
and v3 freeze pass with 9,378 prompts and a 1,771-byte maximum. These are decontamination artifacts,
not benchmark scores. WebLINX still requires residual privacy/legal review and external receipt
archival. The existing 25-update matched smoke and hybrid-only Colab smoke are plumbing checks;
neither is valid architecture-selection evidence.

## Estimand and fixed treatment

The pilot estimates a compound parameter-matched backbone-configuration treatment. Relative to
the control, the hybrid replaces eight of twelve attention mixers with gated causal short
convolutions **and** reduces FFN width from 1,328 to 1,152 to keep total backbone parameters
similar. Training tokens, data order, optimizer, schedule, tokenizer, and seed remain controlled.
Without an additional fixed-FFN or fixed-mixer ablation, this design does not identify the isolated
causal effect of replacing attention with convolution.

| arm | backbone parameters | scheduled loss-token opportunities | tokens/parameter |
|---|---:|---:|---:|
| `webgpu-35m-hybrid` | 34,199,488 | 171,409,408 | 5.012 |
| `webgpu-35m-attn` | 34,275,776 | 171,409,408 | 5.001 |

Both arms use 5,231 optimizer updates × 2 rows/microbatch × 8 accumulated microbatches × 2,048
tokens. Realized loss-token counts in `metrics.json` and the checkpoint are authoritative because
a sampled final packed row can contain padding. The arms deliberately use the same update count,
not separately rounded token-per-parameter counts, so they consume identical deterministic sample
draws.

Three paired seeds are frozen in:

- `configs/train/pretrain-paper-5tpp-{hybrid,attn}-seed2026.yaml`
- `configs/train/pretrain-paper-5tpp-{hybrid,attn}-seed2027.yaml`
- `configs/train/pretrain-paper-5tpp-{hybrid,attn}-seed2028.yaml`

Within a seed, the configs differ only in `model_config` and `log.out_dir`. Across replicates of
one architecture, only `runtime.seed` and `log.out_dir` differ. Tests enforce both claims.

## Data gate

Do not start an arm until all of these conditions pass:

1. `data/shards/paper-all/manifest.json` proves at least **171,409,408 retained training tokens**.
   The runner enforces `data.min_train_tokens` before constructing or training the model.
2. The 16K tokenizer was trained on the frozen training split only and its file identity matches
   the packed manifest and both configs.
3. Train/validation documents and normalized content hashes are disjoint.
4. The corpus preparation manifest binds the configured BFCL, Mind2Web, WebLINX, BrowserGym,
   local action, local DOM, and local agent-eval prompt-only exclusions. Missing external suites
   fail closed. These pretraining exclusions establish revision-bound local holdout; they are not
   the later labeled source used to claim chronological freshness.
5. Dataset revisions, licenses, accepted/rejected counts, source token counts, and every shard hash
   are archived.
6. Acquisition starts only after a self-hashed dry-run plan verifies 30.8 GB of free storage and
   exact local copies of all three revision-pinned license artifacts. Every source must fill its
   character allocation; a short stream cannot become a paper corpus.

All four external components now satisfy the prompt-only exclusion subgate. Their strictly replayed
suite outputs share the same 10,113-byte benchmark-plan identity and are composed in
`data/private/paper-external-denylists-v3.json` (file SHA-256
`3466e9242ccc3aadf487fd2c7fa1dc7bdc9ed14a37007955f75cfece0c040ad1`; self-hash
`826f53f5699f3c4b8f311a9fe70561f5b7d9aa99ce42ce250548e46aa644010b`). The independently
verified paper-all manifest binds these four private inputs together with the three local
exclusions, satisfying item 4 for this frozen corpus. Its bounded scan does not replace WebLINX
review, an exhaustive contamination guarantee, or native benchmark evaluation.

As of 2026-07-29, acquisition is complete and independently `--resume` verified. The private
paper-v5 mixture is 2,450,305,820 bytes with SHA-256
`46650a9bc0ebbdacc6dbd6c87ca3191aee47152f0d179bb2f2e0475f1017094a`; its 229,501-byte download
manifest has file SHA-256 `df92d634fb372ee143225fdf8eb4acc4d7f710cc325d0768f57312856386d5d8`
and self-hash `d0cae6e931738261a9481b3ec02c628bed189f27f24c8dfcae02e75bf7238d94`.
The train-only tokenizer SHA-256 is
`a6de45b9f5e5d7b570c3b12191cc9299fe728e857844b56486760c32f5d45436`. The packed manifest has
file SHA-256 `17f98426499b0539eb24f37a384f552c2699b60ae19ebb17b82574b0f661a86a` and canonical SHA-256
`a6b39482592d8479fe38f4bcd54f9341d3510fe1e7b27a569c1fed717ac4d928`; the freeze has file
SHA-256 `bb04b1b93d95a2542863e231ac484ae7ca9560c97f8d53c6d46bfaae143b489b` and canonical SHA-256
`b005ef46b2fd3c8db91a725ff5dca894d97448f2a8e174f7d7f36811dcdd2aa9`. Independent freeze
verification reproduced the latter identity.

Follow the download, denylist, and corpus-preparation commands in
[`docs/TRAINING_SYSTEM.md`](../TRAINING_SYSTEM.md). Do not satisfy the minimum by repeating the
5,000-row synthetic agent corpus; the pilot is a broad text/code pretraining comparison.

The executable gate is content-addressed by
`configs/data/pretrain-paper-freeze.yaml`. It must create and then re-verify the same freeze before
the first optimizer update:

```bash
PYTHONPATH=src python scripts/freeze_corpus.py configs/data/pretrain-paper-freeze.yaml \
  --out data/shards/paper-all/freeze.json
PYTHONPATH=src python scripts/freeze_corpus.py configs/data/pretrain-paper-freeze.yaml \
  --verify data/shards/paper-all/freeze.json
```

The freeze contains no timestamp or absolute path. Its self-hash binds the corpus manifest and
every shard/length artifact, the tokenizer and its training split, the content-bound split map,
the mixture config/download/raw/filtered/staging lineage, every required evaluation exclusion,
and the six matched consumer/model configs. A missing external suite, cross-split content hash,
tokenizer mismatch, or sub-171,409,408-token corpus fails without writing a freeze.
Each of the six training configs names this exact freeze and specification. The pretraining entry
point rebuilds and compares the audit before constructing the model, then stores the freeze
self-hash in checkpoint/data lineage. Skipping the standalone verification command therefore does
not bypass the gate.

Reverify the completed acquisition in the pinned CPython 3.12 preparation environment:

```bash
PYTHONPATH=src python scripts/download_pretrain_mixture.py configs/data/pretrain-paper.yaml \
  --out data/raw/paper-v5 \
  --license-evidence smollm-card=data/provenance/paper/smollm-card.md \
  --license-evidence codeparrot-card=data/provenance/paper/codeparrot-card.md \
  --license-evidence websight-card=data/provenance/paper/websight-card.md \
  --resume
```

This completed-manifest path rechecks the plan, storage admission, committed source artifacts, and
the mixture byte/hash identity. Any plan/config drift or spool corruption fails closed.

## Execution order

Run one arm at a time on stable power and record device, OS, PyTorch version, power mode, and wall
time. Counterbalance order to reduce thermal/order bias:

| seed | first | second |
|---:|---|---|
| 2026 | hybrid | attention |
| 2027 | attention | hybrid |
| 2028 | hybrid | attention |

```bash
uv run --frozen localagent train pretrain \
  configs/train/pretrain-paper-5tpp-hybrid-seed2026.yaml
uv run --frozen localagent train pretrain \
  configs/train/pretrain-paper-5tpp-attn-seed2026.yaml

uv run --frozen localagent train pretrain \
  configs/train/pretrain-paper-5tpp-attn-seed2027.yaml
uv run --frozen localagent train pretrain \
  configs/train/pretrain-paper-5tpp-hybrid-seed2027.yaml

uv run --frozen localagent train pretrain \
  configs/train/pretrain-paper-5tpp-hybrid-seed2028.yaml
uv run --frozen localagent train pretrain \
  configs/train/pretrain-paper-5tpp-attn-seed2028.yaml
```

Every config is resumable and evaluates the same deterministic validation draws at step 0, every
250 updates, and the final update. Never compare the two arms from their last checkpoint files
alone; compare their complete validation histories, realized token counts, lineage, and failure
records. The project PyTorch-2.13 environment captures and restores MPS RNG state, synchronizes
queued MPS writes before checkpoint publication, and passed a live M5 interrupted/resumed hybrid
regression with bit-identical losses and parameters. Resume remains fail-closed on execution
identity or backend-state drift and is not claimed to reproduce across different devices or
PyTorch builds.

## Downstream post-training data binding

The downstream paper configs preserve the historical frozen 5,000-row eval-v1 artifact but use a
new 50,000-row train-v2 artifact. The production train JSONL is 526,494,339 bytes with SHA-256
`233f4f2d796568097897c73d4547a0129e73a8509981a308600779e3cb4cc060`;
its 6,341-byte sidecar has file SHA-256
`9d415aef41a1557d4dd16339fcde94d6dff5fcf6fec121372e5cfe3f1875f383` and self-hash
`e5b9d66c7761fb6d9f731e4fab2aa5b316d4714911ab7d00d89a9cbe1bd36243`.
The 1,600-byte train-v2 generator config SHA-256 is
`2f03b929507e49f7f73a50e125c144fdd09efa6989306ad0e3c0d03beabc6dbe`.
Strict loads found zero semantic-row overlap and zero rendered-prefix overlap across 93,504 train
and 7,963 eval prefixes. BPE accounting found 3,633,959 input / 2,191,728 loss tokens in train-v2
and 280,949 / 179,607 in eval-v1; maximum rendered lengths were 244 and 214 with no truncation.
Production RL gold maxima are 62 train and 56 eval tokens against a 256-token cap.

Both deterministic 10,000-step SFT plans schedule the same 160,000 draws, 11,629,065 input tokens,
and 7,012,269 loss tokens, within the 5–20M target. The hybrid plan at
`data/provenance/paper/stage-budgets/sft-paper-hybrid.json` has file SHA-256/self-hash
`605c418c338c35c02e0947ae9d063dacf81ce80d0b5ce26c9b3979d5c88681e2` and
`f72ac3ff9e1e8866c5437e831f0b3b9340ee68e669bd3278917b21bc4960b286`;
the attention plan at `data/provenance/paper/stage-budgets/sft-paper-attn.json` has identities
`5b6594e70aad40400affcf9cdace07dc3e77863564a267060af1a34768a7aab3` and
`f8e27c4576ec1ed395804fc08cfa2fd6a5835203140710c5f8b58e38f76ff64c`.
Both artifacts were independently replay-verified exactly with `plan_stage_budget.py --verify`.
The production fanout in `docs/TRAINING_SYSTEM.md` is also complete: its general/code/structured
split streams form an exact, disjoint 504,010-document union of the parent. The first planner
replay rejected the former 2,500-update ceiling at only 7,429,270 supervised tokens. The corrected
25,000-update hybrid and attention plans each replay-verify 85,536,552 input and 74,836,551 loss
tokens, inside the 70–100M target; their self-hashes are
`39e7dc0c17adc9cf46cb541f6eb5531195fede79a5bca47dce42fa9b55ca4a78` and
`f52d8446ee70c144b4c406ddbd0656df99deab7d82f112f18376c1057db79cb4`.
No post-training run or performance result exists, and this readiness does not bypass the
architecture-promotion rule below.

## Promotion rule

This 5-tpp matrix is a pilot, not the final quality claim. Report every seed and the paired
hybrid-minus-attention validation-loss difference with uncertainty. Do not select the hybrid from
training loss alone, and do not pool intermediate checkpoints as independent samples.

The architecture can advance to the 20-tpp and downstream midtrain/SFT comparison only if:

- all six runs have valid, matching data/tokenizer/code lineage;
- neither arm has an instability, NaN, missing checkpoint interval, or materially different
  realized token budget;
- the hybrid is not consistently worse on frozen broad-text and code validation; and
- its measured cached WebGPU TTFT/TPOT advantage survives the separate browser benchmark. The
  current [34M cached-decode result](results/m5-webgpu-cached-decode-20260728.summary.json) passes
  only this random-weight latency gate; it is not quality or promotion evidence by itself.

Advancement does not make the 34M pair an autoregressive deployment candidate. Its measured hybrid
rate is only 74.05/64.54/57.53/47.15 tokens/s at 128/512/1,024/1,536 tokens and misses the
100-tokens/s engineering floor everywhere. It may proceed as a scientific backbone screen or as a
one-forward structured-action candidate only if the trained full policy meets the absolute TTFA
and `Success@B` gates. The autoregressive deployment track should promote the 10M-class shape—or a
new optimized size—unless a real quantized/kernel implementation moves a larger trained model
past the absolute throughput and tail gates.

Numerical non-inferiority margins and task-level promotion thresholds must be frozen before
unblinding the six final results. Until then the repo must describe the hybrid as the latency
treatment, not as the selected architecture.

## Local compute planning

A short MPS planning probe on the current 10-core M5 Air estimated roughly 47 hours of ideal
training compute for one sequential pair at sequence length 2,048 (about 30 hours attention plus
18 hours hybrid), before evaluation, checkpoint I/O, and thermal throttling. This is a scheduling
estimate, not a paper result. Three paired seeds are therefore a multi-day unattended run and
should start only when the machine is plugged in, charging, out of low-power mode, prevented from
sleeping, and logging thermal/power state. External paid compute requires explicit authorization.

The fixed configs imply the following exact scheduled-token arithmetic and linearized local-time
planning bounds:

| Work item | Scheduled token opportunities | Ideal M5 planning time |
|---|---:|---:|
| One 5-TPP arm | 171,409,408 | hybrid ~18 h; attention ~30 h |
| Three paired 5-TPP seeds | 1,028,456,448 | ~144 h / 6.0 days |
| One 20-TPP arm from scratch | 685,539,328 | hybrid ~73 h; attention ~121 h |
| Both 20-TPP arms from scratch | 1,371,078,656 | ~194 h / 8.1 days |
| Six-arm screen plus both full arms | 2,399,535,104 | ~338 h / 14.1 days |

These are optimistic linear extrapolations, not measured end-to-end durations. They exclude
acquisition, tokenizer/packing, validation, export, browser runs, midtraining, 10,000-step SFT,
generation-heavy RL, failures, and analysis. The complete two-arm local plan therefore does not
fit the paper calendar with a defensible contingency margin.

The default no-paid-compute lane is:

1. complete the three-seed 5-TPP architecture screen;
2. promote only one architecture to downstream training;
3. either train that winner from scratch with its 20-TPP config or author and freeze a separate
   5-to-20-TPP continuation protocol before looking at the screen outcomes; and
4. describe the 20-TPP model as a finalist capability run, not a second matched architecture
   comparison.

A continuation would save scheduled compute, but the existing 5-TPP and 20-TPP configs have
different WSD horizons and warmups. Treating a fresh-optimizer continuation as equivalent to a
from-scratch 20-TPP arm would change the estimand. Do not improvise it after model selection.
Running both full arms, adding more devices, or moving to paid accelerators requires an explicit
compute decision before outcomes are unblinded.
