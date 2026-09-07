# GPU training pipeline (HF Jobs): pretrain → SFT → GRPO

Moves the heavy training off the CPU dev box onto a rented GPU via **Hugging Face Jobs**, running the
classic from-scratch stack on **real public datasets**.

## The pipeline (`scripts/train_job.py`)
1. **Pretrain** — next-byte LM on **FineWeb-edu** (raw English text). Gives the byte model general
   language features. (`--real`; else the in-repo synthetic stream.)
2. **SFT** — instruction-tune on **Salesforce/xlam-function-calling-60k**: each row is a query + the
   available tools (rendered in-context) → the gold `<tool_call>{json}</tool_call>`. Teaches
   generative, in-context function calling over arbitrary tool sets.
3. **GRPO** — RL with a **verifiable reward**: sample G rollouts per prompt, reward = 1 iff the
   decoded call AST-matches the gold (no learned reward model needed), group-relative advantage.

Each stage checkpoints to `runs/job/`; `--push <repo>` uploads the final model (`model.pt` +
`model.safetensors` + `config.json`) to the Hub. Model: `configs/model/tiny-30m-byte.yaml` (28M,
byte-level — no tokenizer training needed; FineWeb pretrains on raw UTF-8 bytes).

## Run it
```bash
hf auth login                       # token needs Jobs access + write to the push repo
bash scripts/launch_hf_job.sh       # defaults: flavor l4x1, push danelcsb/localagent-30m-v2
# override: FLAVOR=a10g-large PUSH_REPO=you/model bash scripts/launch_hf_job.sh
hf jobs ps            # running jobs
hf jobs logs <ID>     # stream logs
```
**Cost**: an L4 (`l4x1`, 24 GB) is ample for 28M; the full run (6k pretrain + 3k SFT + 300 GRPO
steps) is ~1–2 h ≈ a couple dollars. Bump `--flavor`/steps for bigger runs.

**Local smoke** (CPU, synthetic, ~2 min — validates the whole chain):
```bash
python scripts/train_job.py --quick
```

## On RLHF
GRPO here is **RL with a verifiable reward** (the deterministic tool-call metric) — the right,
reward-model-free RL for this domain. *Preference-based RLHF* (a learned reward model from human/AI
preference pairs, then PPO/DPO) is a separate next phase: it needs preference data and a reward head,
and pays off most for open-ended text quality rather than verifiable tool calls. The hook for it is
`train/rl.py`; a DPO path would slot in alongside `grpo`.

## Honest caveat
The project's finding is that **28M is capacity-bound** on hard OOD selection. A real GPU
pretrain→SFT→GRPO on real data is the right way to test how far the method goes at this size; to
break the ceiling, raise `PARAM_BUDGET` and add a larger config (a follow-up, more GPU time).

## Alternative GPU backend: Google Colab CLI (free T4)
The same `train_job.py` runs on a Colab GPU via the [Colab CLI](https://github.com/googlecolab/google-colab-cli)
— a **free T4** is enough for this 28M model (A100 on Pro). Cheaper than HF Jobs for the loop habit.

```bash
uv tool install google-colab-cli
colab auth -s la                                   # Google account
HF_TOKEN=hf_xxx bash scripts/launch_colab.sh       # T4; pushes to danelcsb/localagent-30m-v2
#   GPU=A100 PUSH_REPO=you/model bash scripts/launch_colab.sh
```
`launch_colab.sh` provisions the runtime, uploads an HF token, runs `scripts/colab_bootstrap.py`
remotely (clone → install → `train_job.py`), downloads the final checkpoint, and stops the runtime.
(`colab exec -f` runs a *Python* file remotely, so the bootstrap shells out to git/pip itself.)

**Colab-hardened bootstrap:**
- **Keeps Colab's torch** — installs with `pip install -e . --no-deps` + the extras, so a fresh torch
  wheel can't clobber the CUDA-matched one and drop you to CPU.
- **Auto-resumes** — if `sft.pt`/`pretrain.pt` already landed on the push repo (Colab disconnects on a
  ~2-3h run), it `--init-hub`s from the latest stage and skips what's done. Combined with the
  per-stage pushes, a dropped session costs minutes, not hours.
- **Bigger batch** (256) to use the GPU on a 28M model.
- *Deferred:* mixed precision (`autocast`) — a real ~2-4× speedup, but needs a GradScaler for T4 fp16
  and on-GPU testing, so it's not wired yet.

The Colab CLI also ships a Claude Code **skill**, so in an env where it's installed an agent can drive
these steps directly.
