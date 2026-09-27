#!/usr/bin/env bash
# Run the full pretrain -> midtrain -> posttrain pipeline on a Google Colab GPU (free T4, or A100 on Pro) — an
# alternative to HF Jobs. Uses the Google Colab CLI: https://github.com/googlecolab/google-colab-cli
#
# Prereqs:  uv tool install google-colab-cli   &&   colab auth -s la   (Google account)
# A free Colab T4 is enough for this 28M byte model; A100 is faster on Pro.
#
# Usage:    bash scripts/train_on_colab.sh
# Override: GPU, BRANCH, PUSH_REPO, HF_TOKEN, SESSION
set -euo pipefail

SESSION="${SESSION:-la}"
GPU="${GPU:-T4}"                       # T4 (free) | A100 | L4 ...
BRANCH="${BRANCH:-main}"
PUSH_REPO="${PUSH_REPO:-danelcsb/openlocalagent-30m-v2}"
HF_TOKEN="${HF_TOKEN:-}"               # for the final push to the Hub (optional)

# template the bootstrap with this run's branch/push target
tmp=$(mktemp /tmp/colab_bootstrap.XXXX.py)
sed -e "s#^BRANCH = .*#BRANCH = \"${BRANCH}\"#" \
    -e "s#^PUSH_REPO = .*#PUSH_REPO = \"${PUSH_REPO}\"#" \
    scripts/colab_bootstrap.py > "$tmp"

echo "Provisioning Colab ${GPU} runtime '${SESSION}'..."
colab new -s "${SESSION}" --gpu "${GPU}"

if [ -n "${HF_TOKEN}" ]; then
  echo "${HF_TOKEN}" > /tmp/hf_token
  colab upload -s "${SESSION}" /tmp/hf_token /content/hf_token
  rm -f /tmp/hf_token
fi

echo "Running the pipeline on Colab (pretrain -> midtrain -> posttrain)..."
colab exec -s "${SESSION}" -f "$tmp"

echo "Downloading the final checkpoint..."
colab download -s "${SESSION}" /content/openlocalagent/runs/job/tiny-30m-byte-final.pt \
  results/runs/colab-final.pt || echo "(download skipped — also pushed to ${PUSH_REPO} if HF_TOKEN was set)"

colab stop -s "${SESSION}"
rm -f "$tmp"
echo "Done. Result: results/runs/colab-final.pt  and/or  https://huggingface.co/${PUSH_REPO}"
