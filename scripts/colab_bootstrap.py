"""Colab bootstrap — runs ON the Colab runtime via `colab exec -f scripts/colab_bootstrap.py`.

`colab exec` transmits this file to the remote kernel and runs it (Python only, no shell), so we
clone the repo, install it, and run the same GPU pipeline as the HF Job. Config via the constants
below (the launcher `scripts/launch_colab.sh` sed-overrides them).

Hardened for Colab: (1) installs WITHOUT touching Colab's CUDA-matched torch; (2) auto-resumes from
the latest stage checkpoint already on the Hub (Colab disconnects on ~2-3h runs); (3) bigger batch to
use the GPU. HF push works once the runtime is HF-authenticated (the launcher uploads a token).
"""
import os
import subprocess

# --- config (launch_colab.sh overrides these) ---
REPO = "https://github.com/sangbumchoi/localagent"
BRANCH = "claude/tiny-llm-agents-5C3c0"
PUSH_REPO = "danelcsb/localagent-30m-v2"
WORKDIR = "/content/localagent"
BATCH = 256        # 28M model on a 16GB T4 — fp32 still leaves room; bigger batch => faster


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


# 1. code: clone (or update) the repo at the branch
if os.path.isdir(WORKDIR):
    sh(f"cd {WORKDIR} && git fetch --depth 1 origin {BRANCH} && git checkout {BRANCH} && git pull")
else:
    sh(f"git clone --depth 1 -b {BRANCH} {REPO} {WORKDIR}")
os.chdir(WORKDIR)

# 2. deps — install the package + extras but DO NOT reinstall torch (Colab's is CUDA-matched;
#    a fresh torch wheel can drop you to CPU or mismatch CUDA). --no-deps skips torch/numpy.
sh("pip install -q -e . --no-deps")
sh("pip install -q 'datasets>=2.19' 'huggingface_hub>=0.24' safetensors pyyaml tokenizers tqdm")

import torch
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}", flush=True)

# 3. HF auth for the push (token uploaded to /content/hf_token by the launcher, optional)
if os.path.exists("/content/hf_token"):
    from huggingface_hub import login
    login(open("/content/hf_token").read().strip())

# 4. RESUME: if a later stage already landed on the Hub (after a disconnect), start from it.
resume, stages = "", "pretrain,sft,grpo"
if PUSH_REPO:
    try:
        from huggingface_hub import HfApi
        have = set(HfApi().list_repo_files(PUSH_REPO))
        if "sft.pt" in have:
            resume, stages = f"--init-hub {PUSH_REPO}:sft.pt", "grpo"
            print("found sft.pt on the Hub -> resuming at GRPO", flush=True)
        elif "pretrain.pt" in have:
            resume, stages = f"--init-hub {PUSH_REPO}:pretrain.pt", "sft,grpo"
            print("found pretrain.pt on the Hub -> resuming at SFT", flush=True)
    except Exception as e:  # noqa: BLE001 - repo may not exist yet; just start fresh
        print(f"(no resume checkpoint: {e})", flush=True)

# 5. run the pipeline (CUDA auto-detected; per-stage pushes persist progress)
push = f"--push {PUSH_REPO}" if PUSH_REPO else ""
sh(f"python scripts/train_job.py --real --stages {stages} {resume} "
   f"--pretrain-steps 6000 --sft-steps 3000 --grpo-steps 300 --batch {BATCH} --seq-len 512 {push}")
print("COLAB_BOOTSTRAP_DONE", flush=True)
