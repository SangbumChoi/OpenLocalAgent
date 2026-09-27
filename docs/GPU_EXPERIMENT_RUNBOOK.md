# Portable GPU experiment runbook

The campaign is designed to run on a separate CUDA machine without changing the research
protocol. It compares the attention/hybrid/vision and 10M–96M architecture arms, records the
public evaluation matrix, and can bind warm-versus-random checkpoint transfer reports.

## Install and preflight

```bash
git clone https://github.com/SangbumChoi/LocalAgent.git
cd LocalAgent
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,data,demo,export]"
python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

Install the optional Hub and tracking clients when using acquisition/upload/logging:

```bash
pip install -e ".[data,hub,tracking]"
```

## Acquire approved Hugging Face sources

The campaign does not silently pull evaluation tasks. Its approved source manifest contains only
the six training sources and the existing public model reference, with file patterns and a
resolved revision recorded in `acquisition-manifest.json`. Five rows are anonymous-public;
Salesforce xLAM is Hub-gated and requires your own `hf auth login` or `HF_TOKEN` when selected:

```bash
PYTHONPATH=src python scripts/run_gpu_campaign.py \
  --output runs/gpu-campaign-cuda \
  --skip-benchmark \
  --acquire-hf --hf-out data/hf-campaign
```

Use `--hf-dry-run` first on a new machine. Select a subset with repeated `--hf-source` flags. The
downloaded manifest is hash-bound and is the input for later normalization/training; benchmark
evaluation rows are intentionally rejected.

## Run the matched architecture matrix

This uses fresh random weights and measures prefill, cached decode, uncached decode, memory
estimates, dtype, and CUDA peak allocation. It is a systems comparison, not a quality score.

```bash
PYTHONPATH=src python scripts/run_gpu_campaign.py \
  --output runs/gpu-campaign-cuda \
  --device cuda --dtype auto \
  --prompt-len 512 --decode 128 --repeats 5
```

To log the same run to Weights & Biases, install the `tracking` extra and authenticate first with
`wandb login` (or `WANDB_API_KEY`), then add:

```bash
  --wandb --wandb-project localagent --wandb-mode online
```

Use `--wandb-mode offline` when the GPU machine has no network; the run can be synced later with
the W&B CLI. Tokens are never written into the campaign receipt.

For a faster smoke test, use `--prompt-len 64 --decode 32 --repeats 2`. To run a subset, repeat
`--model webgpu-10m-attn` / `--model webgpu-10m-hybrid` (and so on). The complete architecture
matrix and staged train configs are in
[`webgpu-realistic-campaign.v1.yaml`](../configs/experiments/webgpu-realistic-campaign.v1.yaml).

## Analyze pretrained-weight reuse

After producing a base and target checkpoint with the same tokenizer/config, bind the pair in the
campaign receipt:

```bash
PYTHONPATH=src python scripts/run_gpu_campaign.py \
  --output runs/gpu-campaign-transfer \
  --device cuda --skip-benchmark \
  --transfer runs/pretrain/latest.pt:runs/sft/latest.pt \
  --checkpoint runs/sft/latest.pt
```

The transfer report separates embeddings, attention/mixer, FFN, normalization, output, and action
heads. Promotion requires a matched no-transfer control and held-out task improvement; movement
alone is not evidence that transfer is useful.

## Run the staged training arms

The CLI is config-driven; do not replace these with ad-hoc flags. Run the attention and hybrid arms
with the same seed, corpus freeze, token budget, and output naming convention, then repeat the same
mid-training, SFT, and RL configs from the campaign manifest:

```bash
localagent train pretrain configs/train/pretrain-paper-tier-10m-attn.yaml
localagent train pretrain configs/train/pretrain-paper-tier-10m-hybrid.yaml
localagent train midtrain configs/train/midtrain-paper-tier-10m-attn.yaml
localagent train midtrain configs/train/midtrain-paper-tier-10m-hybrid.yaml
localagent train sft configs/train/sft-paper-tier-10m-attn.yaml
localagent train sft configs/train/sft-paper-tier-10m-hybrid.yaml
localagent train rl configs/train/rl-paper-tier-10m-attn.yaml
localagent train rl configs/train/rl-paper-tier-10m-hybrid.yaml
```

The 96M pair uses the corresponding `*-tier-96m-*` files. These stages may require acquired,
hash-bound corpus manifests and public train-only adapters; a missing artifact should stop with an
error rather than substitute evaluation rows. Keep a separate output directory per architecture,
seed, and stage so the transfer script can compare exact pairs.

## Evaluation order

1. Run the campaign and inspect the architecture/cache receipt.
2. Train only the six matrix rows marked `train` (AndroidControl, AITW, Mind2Web, AgentNet,
   xLAM Function Calling, ToolACE).
3. Keep BrowserGym, AndroidWorld, OSWorld, ToolSandbox, MCPMark, AppWorld, and other benchmark
   tasks evaluation-only. Run their pinned native adapters separately and attach their receipts.
4. Add visual/mobile scores only when an emulator/VM and independent verifier are present.
5. Export the selected checkpoint to ONNX/WebGPU and run browser parity/latency checks before any
   deployment or Hugging Face publication claim.

The campaign never downloads benchmark payloads, starts an emulator, calls external accounts, or
silently trains. Missing runtimes are recorded in the preflight section rather than converted into
zero or success scores.

## Publish after verification

Uploads are separate from acquisition and require an authenticated identity. The publisher checks
that the model/dataset repository owner matches the logged-in Hub user and refuses unsealed or
evaluation-only datasets:

```bash
hf auth login
PYTHONPATH=src python scripts/publish_hf_campaign.py \
  --checkpoint runs/sft/latest.pt \
  --model-repo YOUR_ID/localagent-webgpu \
  --model-out runs/hf_campaign_model \
  --dataset-dir data/hf-campaign \
  --dataset-repo YOUR_ID/localagent-training-data \
  --public --push
```

Replace `YOUR_ID` with the username returned by `hf auth whoami`. A local bundle can be built
without `--push`; no public URL is claimed until the upload and anonymous re-fetch audit pass.
