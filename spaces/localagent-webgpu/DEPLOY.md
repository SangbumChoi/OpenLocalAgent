# Deploy the latest model + WebGPU demo to Hugging Face

Everything here is **prepared but not pushed** — publishing to your HF account is yours to run.
Requires `huggingface_hub` (installed) and a write token. Set `HF_USER` to your own namespace;
the commands below deliberately do not assume a public account or URL. The current release target
is the 10,524,544-parameter BPE checkpoint at
`runs/sft-mind2web-public-continuation-20260805/latest.pt` (SHA-256
`6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45`).

The current publisher binds the model and WebGPU bundle to one 63-tool pool (50 standard tools,
11 mobile actions, and 2 full-field email/Notion schemas). The pool is inferred from the
checkpoint's 51-class fixed head using `scripts/publish_hf_release.py`; the current local staging
and parity audit is [`m502-local-hf-space-stage-current-v1.json`](../../docs/paper/results/raw/m502-local-hf-space-stage-current-v1.json).
The app (`app.js`) uses the structured route head → dense selector → pointer-copy dispatch over the
63-tool release surface plus named, bounded URL/search/planner safety adapters. It also applies the
versioned `side_effect_confirmation_v1` boundary: read-only actions may proceed, state-changing
email/Notion/message/file/shell/browser-interaction actions are marked for confirmation, and
prompt-injection or secret-exfiltration signals are blocked. The static demo has no external
credentials and never executes those side effects. A public Hub URL is not claimed until the
upload completes and is independently verified.

The older m57/m58 stateful-productivity child remains a historical local artifact. The current
10,524,544-parameter BPE checkpoint and 63-tool bundle are the release target; generated ONNX and
safetensors files remain outside Git and must be regenerated or uploaded by an authenticated
maintainer.

## 0. Authenticate (once)
```bash
hf auth login          # paste a token with write access  (or: export HF_TOKEN=hf_xxx)
```

## 0b. One-command verified release

The repository also provides a fail-closed publisher that builds the model and Space bundles,
verifies the exact checkpoint/parity hashes, uploads both repositories only with explicit
`--publish`, and writes the anonymous current-checkpoint audit.  The command refuses to publish
without a token or `--audit-output`:

```bash
PYTHONPATH=src python scripts/publish_hf_release.py \
  --checkpoint runs/sft-mind2web-public-continuation-20260805/latest.pt \
  --model-repo "$HF_USER/localagent-webgpu-10m" \
  --space-repo "$HF_USER/localagent-webgpu" \
  --model-out build/hf-current \
  --web-out build/web-current \
  --space-out build/space-current \
  --audit-output docs/paper/results/raw/public-model-demo-manifest-current-v2.json \
  --public --publish
```

Omit `--publish` for a local-only preparation run.  A successful upload is still not considered a
current release unless the final anonymous audit records `current_checkpoint_match=true`.

## 1. Export the inference bundle from the current checkpoint
```bash
CURRENT_CHECKPOINT=runs/sft-mind2web-public-continuation-20260805/latest.pt
python -c "from localagent.inference.export.to_onnx import export_web; \
           export_web('${CURRENT_CHECKPOINT}', 'build/web', action_only=True)"
# writes the full logits graph, hidden-only action graph, heads/meta, and bundle-manifest.json
# bundle-manifest.json is published only after all fp32/fp16 graphs pass hard PyTorch parity

python -c "import json, pathlib; p=pathlib.Path('build/web'); \
           m=json.loads((p/'meta.json').read_text()); \
           b=json.loads((p/'bundle-manifest.json').read_text()); \
           assert m.get('action_model_file') == 'action_model.fp16.onnx'; \
           assert b['schema_version'] >= 3; \
           assert b['parity_gate']['hard_gate'] and b['parity_gate']['passed']; \
           assert b['parity_gate']['results'][m['action_model_file']]['passed']"
```

## 2. Model repo — host the checkpoint + ONNX
```bash
HF_USER=your-huggingface-user
MODEL_REPO="$HF_USER/localagent-webgpu-10m"
SPACE_REPO="$HF_USER/localagent-webgpu"
HF_OUT=build/hf-current
python scripts/push_to_hf.py --checkpoint "$CURRENT_CHECKPOINT" --out "$HF_OUT" \
  --repo "$MODEL_REPO" --public --push
hf repo create "$MODEL_REPO" --repo-type model -y || true
hf upload "$MODEL_REPO" build/web/ . --repo-type model
# (load in PyTorch via this repo's LocalAgentLM/ModelConfig — pure PyTorch, no transformers dep)
```

## 3. Space — the WebGPU demo
The Space is `sdk: static` (see the frontmatter in `README.md`). Copy the complete checked bundle
next to the app, then push the whole folder.
```bash
cp build/web/{model.fp16.onnx,action_model.fp16.onnx,heads.json,meta.json,dispatch_heads.json,bundle-manifest.json} spaces/localagent-webgpu/
hf repo create "$SPACE_REPO" --repo-type space --space_sdk static -y || true
hf upload "$SPACE_REPO" spaces/localagent-webgpu/ . --repo-type space \
  --exclude "DEPLOY.md"        # DEPLOY.md is for maintainers, not the live Space
```
`hf upload` puts the ONNX graphs on LFS automatically. The demo is then live at
`https://huggingface.co/spaces/$HF_USER/localagent-webgpu`.

## 3b. Verify the public release against the exact checkpoint

After both uploads, run the anonymous auditor. It fetches the public API metadata and resolver
bytes, then emits a manifest whose `current_checkpoint_sha256` is populated only when the remote
model `config.json` contains the exact checkpoint hash. A legacy model or stale Space therefore
fails the current-release gate instead of being mistaken for this checkpoint.

```bash
PYTHONPATH=src python scripts/audit_public_hf_release.py \
  --model-repo "$MODEL_REPO" \
  --space-repo "$SPACE_REPO" \
  --checkpoint "$CURRENT_CHECKPOINT" \
  --dataset-url "https://huggingface.co/datasets/$HF_USER/localagent-dispatch-data" \
  --output docs/paper/results/raw/public-model-demo-manifest-current-v2.json
```

Then pass the manifest and checkpoint to the workshop gate:

```bash
PYTHONPATH=src python scripts/workshop_gate.py \
  --public-artifact-manifest docs/paper/results/raw/public-model-demo-manifest-current-v2.json \
  --current-checkpoint "$CURRENT_CHECKPOINT" \
  --strict
```

## 3a. Verify before uploading (fail closed)

The source tree deliberately omits generated ONNX/model artifacts. Verify the export and the
static app before copying anything into the Space:

```bash
PYTHONPATH=src python scripts/verify_demo_deploy.py \
  --demo-dir spaces/localagent-webgpu \
  --bundle-dir build/web \
  --checkpoint runs/sft-mind2web-public-continuation-20260805/latest.pt \
  --expected-tool-count 63
```

The command checks the exporter manifest, every byte count and SHA-256, the hard PyTorch parity
gate, the action-graph contract, and the app's manifest loader. It returns non-zero when the clean
Space is missing the generated bundle (the expected result until the sync step). To stage a local
deployable copy, use the explicit sync operation and then verify the target itself:

```bash
PYTHONPATH=src python scripts/verify_demo_deploy.py \
  --demo-dir spaces/localagent-webgpu \
  --bundle-dir build/web --sync \
  --checkpoint runs/sft-mind2web-public-continuation-20260805/latest.pt \
  --expected-tool-count 63
PYTHONPATH=src python scripts/verify_demo_deploy.py \
  --demo-dir spaces/localagent-webgpu \
  --checkpoint runs/sft-mind2web-public-continuation-20260805/latest.pt \
  --expected-tool-count 63
```

The current local paired Space staging was regenerated and verified by the [`m502 receipt`](../../docs/paper/results/raw/m502-local-hf-space-stage-current-v1.json), and the tracked static bundle is independently rechecked by [`m515`](../../docs/paper/results/raw/m515-current-webgpu-deploy-verify-v1.json): the exact `6a6520…` checkpoint, ONNX parity, and WebGPU bundle all match. This remains a local pre-upload receipt until the same files are uploaded to an authenticated Hub Space.

Do not hand-create `bundle-manifest.json` or upload a partial bundle. A verified local export is
still not native WebGPU capability evidence; the workshop gate separately requires a hardware
adapter receipt, native benchmark receipts, transfer/no-transfer ablations, and public URLs.

## Notes
- The bundle files are git-ignored deploy artifacts — they are NOT in this source tree; step 1
  regenerates them deterministically from the checkpoint.
- To verify locally before pushing: `cd spaces/localagent-webgpu && cp ../../build/web/*.{onnx,json}
  . && python -m http.server 8000` then open http://localhost:8000. The interactive demo may retry
  on WASM. Benchmark pages require an explicit provider, the manifest, and the distinct hidden-only
  action graph; they fail rather than falling back or relabeling the full logits graph.
- Suggested model-card blurb for the model repo README:
  > Sub-100M, BPE, pretrained-from-scratch tool-calling agent. The exact parameter count
  > comes from the exported `meta.json`/`bundle-manifest.json`; the checked seed-2027 bundle has
  > 10,524,544 parameters. Generable dispatch:
  > 5-way route head → dense two-tower selector (scores any tool by its description embedding) →
  > pointer-copy arguments. 63-tool release surface (standard, mobile, and productivity schemas).
  > Runs in-browser and requests ONNX Runtime Web's WebGPU provider — see the linked
  > Space. Per-node placement is not exposed by ORT Web.

## Matched cached-decode latency page

The cache-bearing benchmark is a separate untrained-random-weight latency artifact. It is not the
trained action bundle above. Export produces separate prefill and fixed-`T=1` decode graphs and
publishes `matched-decode.json` only after multi-length, multi-step token/cache parity passes.
Generate the 34.2M pair from the repository root:

```bash
python scripts/export_matched_webgpu_decode.py \
  --out runs/webgpu/random-cached-decode-latency-seed-20260728-v2
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/spaces/localagent-webgpu/decode-benchmark.html?backend=webgpu&manifest=../../runs/webgpu/random-cached-decode-latency-seed-20260728-v2/matched-decode.json
```

The two smaller matched pairs use the same exporter and page:

```bash
python scripts/export_matched_webgpu_decode.py \
  --hybrid-config configs/model/webgpu-16m-hybrid.yaml \
  --attention-config configs/model/webgpu-16m-attn.yaml \
  --out runs/webgpu/random-cached-decode-16m-seed-20260728
python scripts/export_matched_webgpu_decode.py \
  --hybrid-config configs/model/webgpu-10m-hybrid.yaml \
  --attention-config configs/model/webgpu-10m-attn.yaml \
  --out runs/webgpu/random-cached-decode-10m-seed-20260728
```

Use `backend=wasm` for the separately labeled WASM condition. The page is run-once: reload before
collecting another repetition set. If publishing this benchmark in a static Space, copy the entire
export directory (pair manifest, both provenance/config files, and graph files) under the Space
directory and pass its deployed `matched-decode.json` through the `manifest` query parameter. Do
not publish a graph without its exporter-produced manifest and provenance; the page fails closed
on their labels, trajectory-parity evidence, byte counts, and SHA-256 identities.

For WebGPU, the page requests `gpu-buffer` present-cache outputs and rebinds the returned tensors
as the next call's past inputs without reading cache contents into JavaScript. That is not evidence
of physical residency or per-node placement, which ONNX Runtime Web does not expose. The graph
returns fresh presents each step: attention uses append/concat and short-conv replaces a fresh
fixed-width tail. It is not an in-place or paged-cache implementation.

The tracked three-run summaries and every raw/config link are in the
[paper result index](../../docs/paper/results/README.md). Only the 10.5M hybrid clears the
100 tok/s engineering reference at all four tested contexts; the 34.2M and 15.6M pairs do not.
This latency page remains a separately labeled systems experiment. The trained complete-action
runner uses the same strict `next_token`, `[B,V]` logits, and cache ABI, but requires a final-RL
checkpoint lineage plus the exact `openai_full_catalog_v1` tokenizer/catalog contract.

## Production cached autoregressive bundle

Export the autoregressive control from the same final RL checkpoint and verified BPE tokenizer
with `export_cached_decode(...)`, including the non-empty `training_artifact_sha256` set. Copy its
complete output directory under `spaces/localagent-webgpu/cached/`; do not rename individual
graphs or sidecars. The browser defaults to `cached/meta.json` and loads this bundle lazily only
when an autoregressive policy is selected, so the structured action graph remains independent.

The deployed primary `bundle-manifest.json` must also contain an artifact identity whose
`file` is exactly `cached/provenance.json`, with the fetched file's byte count and lowercase
SHA-256. Benchmark-grade pages reject an unpinned cached provenance file. That provenance must in
turn pin `meta.json`, `training-lineage.json`, the copied tokenizer/config, and both selected
prefill/decode graphs. The runtime validates the complete chain before constructing either cached
ORT session; stale output names, dtypes, cache geometry, tokenizer/catalog metadata, checkpoint
ancestry, or graph identities stop the autoregressive policy instead of falling back to
full-context recomputation.
