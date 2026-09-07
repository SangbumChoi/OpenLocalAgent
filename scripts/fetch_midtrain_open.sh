#!/usr/bin/env bash
# Acquire open agentic corpora for midtrain that are NOT the training split of any scored suite.
#
# Licence gate, applied before anything is fetched: permissive only. Salesforce/APIGen-MT-5k
# (CC-BY-NC-4.0) and osunlp/UGround (CC-BY-NC-SA) are deliberately absent for the same reason
# WebLINX is - non-commercial terms. internlm/Agent-FLAN is absent for a different reason: it is
# derived from ToolBench, which the harness scores, so it belongs to posttrain's side of the line.
#
# Whole-file GETs only. Range reads through this mirror run at ~61 KB/s against ~302 MB/s for a
# whole file.
set -euo pipefail
BASE=http://nexus.tossbank.bz/repository/huggingface-proxy
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="${MIDTRAIN_RAW:-$root/data/raw/midtrain-open}"
mkdir -p "$RAW"

get() {   # get <repo> <revision> <path-in-repo> <local-name>
  local repo="$1" rev="$2" path="$3" out="$RAW/$4"
  if [ -s "$out" ]; then echo "[$(date +%H:%M:%S)] have $4"; return 0; fi
  echo "[$(date +%H:%M:%S)] GET $repo/$path"
  curl -sS -o "$out" -m 3600 -w '  http=%{http_code} %{size_download} bytes %{speed_download} B/s\n' \
    "$BASE/datasets/$repo/resolve/$rev/$path"
}

# microsoft/orca-agentinstruct-1M-v1  (CDLA-Permissive-2.0)
# Only the agentic-leaning splits. open_domain_qa / rc / mcq / text_* / creative_content are
# general instruction data - that is pretraining's job, not this stage's.
ORCA=86d60918
orca() { get microsoft/orca-agentinstruct-1M-v1 "$ORCA" "data/$1" "orca-$2"; }
orca tool_use-00000-of-00001.parquet            tool_use.parquet             # tool control
orca webagent_flow-00000-of-00001.parquet       webagent_flow.parquet        # web control
orca code_-00000-of-00002.parquet               code-0.parquet               # python control
orca code_-00001-of-00002.parquet               code-1.parquet
orca follow_up-00000-of-00002.parquet           follow_up-0.parquet          # multi-turn
orca follow_up-00001-of-00002.parquet           follow_up-1.parquet
orca fs_cot_flow-00000-of-00001.parquet         fs_cot_flow.parquet          # task comprehension
orca analytical_reasoning-00000-of-00001.parquet analytical_reasoning.parquet
orca struct2text_flow-00000-of-00001.parquet    struct2text_flow.parquet
orca rag-00000-of-00001.parquet                 rag.parquet

# xingyaoww/code-act (Apache-2.0) - python control
get xingyaoww/code-act afba3436 data/codeact-00000-of-00001-41e65239aebc832c.parquet codeact.parquet
get xingyaoww/code-act afba3436 data/general-00000-of-00001-d4bf3f50dd590916.parquet codeact-general.parquet

# nvidia/When2Call (CC-BY-4.0) - when to call a tool at all, and when not to
get nvidia/When2Call 0582f774 train/when2call_train_sft.jsonl when2call-sft.jsonl

# interstellarninja/hermes_reasoning_tool_use (Apache-2.0) - tool use with reasoning traces
get interstellarninja/hermes_reasoning_tool_use c27e4549 data/train-00000-of-00001.parquet hermes-tool-use.parquet

echo "[$(date +%H:%M:%S)] MIDTRAIN_OPEN_FETCH_DONE"
ls -la "$RAW"
