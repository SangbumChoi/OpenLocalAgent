#!/usr/bin/env bash
# Launch the full pretrain -> midtrain -> posttrain run on a Hugging Face Job (GPU).
#
# Prereqs: `hf auth login` with a token that has (a) Jobs access (Pro/Team or PAYG credits) and
# (b) write access to the --push repo. Run `hf jobs hardware` to see flavors/pricing.
#
# Usage:  bash scripts/train_on_hf.sh
# Override via env:  FLAVOR, BRANCH, PUSH_REPO, IMAGE, TIMEOUT, REAL, STAGES
set -euo pipefail

FLAVOR="${FLAVOR:-l4x1}"                 # 24GB L4 — ample for a 28M byte model; cheap
IMAGE="${IMAGE:-pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
REPO_URL="${REPO_URL:-https://github.com/SangbumChoi/OpenLocalAgent}"
BRANCH="${BRANCH:-main}"
PUSH_REPO="${PUSH_REPO:-danelcsb/openlocalagent-30m-v2}"
TIMEOUT="${TIMEOUT:-6h}"
REAL="${REAL:---real}"                   # set REAL="" to run on the in-repo synthetic data instead
STAGES="${STAGES:-pretrain,sft,grpo}"
INIT_HUB="${INIT_HUB:-}"                  # resume: 'repo:file.pt' (e.g. to skip a finished pretrain)

# The job: install deps, clone the repo @ branch, install the package, run the full pipeline, push.
read -r -d '' CMD <<EOF || true
set -e
apt-get update -q && apt-get install -y -q git
pip install -q -U "datasets>=2.19" "huggingface_hub>=0.24" safetensors pyyaml
git clone --depth 1 -b ${BRANCH} ${REPO_URL} /work && cd /work
pip install -q -e .
python scripts/train_job.py ${REAL} --stages ${STAGES} ${INIT_HUB:+--init-hub ${INIT_HUB}} \
  --pretrain-steps 6000 --sft-steps 3000 --grpo-steps 300 \
  --batch 64 --seq-len 512 --push ${PUSH_REPO}
EOF

echo "Launching HF Job:  flavor=${FLAVOR}  branch=${BRANCH}  push=${PUSH_REPO}  real=${REAL:-no}"
hf jobs run \
  --flavor "${FLAVOR}" \
  --secrets HF_TOKEN \
  --timeout "${TIMEOUT}" \
  --detach \
  "${IMAGE}" \
  bash -c "${CMD}"

echo
echo "Track it:  hf jobs ps        (running jobs)"
echo "Logs:      hf jobs logs <JOB_ID>"
echo "Result will be pushed to:  https://huggingface.co/${PUSH_REPO}"
