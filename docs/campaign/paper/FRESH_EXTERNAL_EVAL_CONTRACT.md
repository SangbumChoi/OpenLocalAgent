# Fresh external action-evaluation contract

Status: the legacy v1 and hardened v2 derived exact-action lanes are implemented, but **no
external source export or paper slice is currently frozen**.
Acquiring a real revision-bound external source, constructing the contract with exact training
artifact identities, and externally timestamping the resulting identities are prerequisites.
This document does not claim those steps have happened.

Implementation:

- `src/localagent/eval/external_action_contract.py`
- `src/localagent/eval/realtime.py`
- `scripts/fresh_action_eval.py`
- `tests/test_external_action_contract.py`

## Purpose and boundary

The 20-case local action suite is diagnostic. It cannot resolve the paper's two-percentage-point
non-inferiority margin. The final structured-versus-autoregressive comparison requires at least
200 new unique tasks plus a real external benchmark slice. This contract makes that future slice
deterministic and auditable without fabricating or downloading benchmark data.

The mechanism has two deliberately different versions:

- **v1 is historical compatibility.** It freezes pretrain/midtrain/SFT inputs and compares
  adapter-supplied normalized calls. Valid existing v1 artifacts remain byte-reproducible and keep
  the historical `localagent_fresh_external_action_comparison` kind. They are explicitly not raw
  whole-output model evaluations.
- **v2 is the fail-closed paper lane.** It requires pretrain, midtrain, SFT, and RL text plus a
  bound lineage export for every stage. Conversation inputs must declare
  `openai_full_catalog_v1`; the audit scans each exact
  `assistant_training_example.prompt + body + terminal EOS` materialization, including the full
  catalog and prior history. Paired results retain `raw_output` and `finish_reason`, are parsed by
  `parse_tool_output`, validate predicted calls against the frozen recursive schemas, and verify
  the actual checkpoint and bundle files by bytes and SHA-256.

Both versions select a bounded slice by seeded hash rank, fail on prompt/shingle or derived action
template overlap, average repetitions within a case, and bootstrap whole task clusters.

It cannot prove the absence of semantic paraphrases, unreported training data, human benchmark
exposure, or contamination in a training artifact not named by the contract. The derived template
check replaces scalar gold arguments in a normalized prompt and retains the sorted multiset of
tool names, matching the evaluator's call-order-insensitive semantics. It is a conservative
skeleton-equality check, not a semantic template classifier. Unlabeled
pretraining text can be prompt/shingle screened but cannot receive a labeled-action template
proof.

Training artifacts are streamed and screened in bounded row/character chunks; the runner stops on
the first overlap rather than retaining an unbounded match list. The version-1 external source is
a single JSON object and is parsed in memory, with hard ceilings of 256 MiB and 50,000 cases.
Per-record training input has a hard 16 MiB ceiling, and total declared training artifacts have an
8 GiB per-file ceiling. These are explicit resource bounds, not a streaming external-source
parser. v2 currently uses the same bounded single-object source representation.

All v2 JSON envelopes reject duplicate keys and unknown keys. Function catalogs use the shared
recursive prompt-contract validator, including supported-keyword checks, nested
`additionalProperties`, unique `required` entries, finite JSON values, and reserved framing-marker
rejection. Gold calls must name a catalog tool and satisfy that exact recursive schema.

## Isolation lifecycle

There are two different clocks and two different prompt-only artifacts:

- **Pretraining decontamination.** Before tokenizer fitting or any training, freeze prompt-only
  exports from the already chosen public benchmark revisions. Those files may be used only as
  exclusion inputs. They contain no expected actions or outcomes. Because the revisions were
  chosen before training and may already be public, this proves locally held-out,
  revision-bound evaluation—not chronological freshness.
- **Fresh post-training capability evaluation.** After every compared checkpoint and training
  text artifact is frozen, acquire a new immutable benchmark revision, hidden steward export, or
  procedurally generated seed set that was not selected while inspecting model outcomes. Freeze
  and externally timestamp its labeled slice before running either system. Its prompt-only
  denylist protects only future training.

Do not reuse one artifact to make both claims. Hash-ranking an old public suite after training
does not make that suite chronologically fresh.

For the post-training lane, follow this order:

1. Finish training and freeze the exact pretrain, midtrain, SFT, and RL text artifacts used by the
   candidate checkpoints. Export and retain the four stage-lineage sidecars described below.
2. Export a real external benchmark revision through a benchmark-specific adapter into the source
   schema below. Do not inspect model outputs.
3. Author the contract with exact source and training identities and the already-decided
   selection/analysis parameters.
4. Run `freeze`. It selects first and then audits the selected cases. A contaminated selected case
   causes failure; the runner does not substitute another case post hoc.
5. Externally timestamp the contract, frozen slice, prompt-only denylist, and freeze-manifest
   hashes **before** any candidate or baseline outcomes are inspected.
6. Run both systems on identical frozen case/repetition opportunities. For v2, preserve the
   unmodified whole model string and terminal reason in every record. Do not repair, normalize, or
   pre-parse it in the producer.
7. Run `compare`. v2 strictly parses the whole output, rejects content outside tool envelopes,
   duplicate JSON keys, unknown call-object keys, malformed markers, non-finite values, unknown
   tools, and recursive schema violations. It rejects disagreement with any supplied `success`
   boolean.

The frozen slice and its expected calls are evaluation-only. They must not enter training. For any
future training after the freeze, only the prompt-only denylist may be supplied to corpus
preparation, and all new training artifacts require a new audit and a newly timestamped contract.

Neither the v1 nor v2 derived exact-action schema is a native scoring wrapper for every named
benchmark.
BFCL accepts some alternative normalized calls; Mind2Web and WebLINX have multi-gold,
element/intent/text metrics; BrowserGym scores whole trajectories by final environment reward.
Any conversion must therefore be labeled a derived exact-action slice rather than an official
benchmark score. Faithful multi-gold and closed-loop trajectory evaluation require separate
versioned contracts. Hardening the evidence chain does not turn this result into an official
benchmark score.

## External source export

No adapter is implicit. BFCL, BrowserGym, Mind2Web, WebLINX, or another source must be converted
from one named immutable revision and split to:

```json
{
  "kind": "localagent_external_action_export",
  "schema_version": 1,
  "benchmark": "real-benchmark-name",
  "revision": "immutable-upstream-revision",
  "split": "heldout-split",
  "cases": [
    {
      "source_case_id": "stable-upstream-case-id",
      "task_cluster_id": "upstream-task-or-template-family-cluster",
      "template_id": "upstream-template-id",
      "family": "function_calling",
      "prompt": "The exact user instruction.",
      "tools": [
        {
          "name": "tool_name",
          "description": "Tool description.",
          "parameters": {
            "type": "object",
            "properties": {
              "argument": {"type": "string"}
            },
            "required": ["argument"]
          }
        }
      ],
      "expected_calls": [
        {
          "name": "tool_name",
          "arguments": {"argument": "grounded value"}
        }
      ],
      "metadata": {}
    }
  ]
}
```

The hardened lane uses the same case fields with `"schema_version": 2`; the contract and source
versions must match. Every object envelope has an exact key set. The tool schema is restricted to
the recursive subset shared with `localagent.data.prompt_contract`: `type`, `description`, `enum`,
`format`, `properties`, `required`, `additionalProperties`, and `items`. Unsupported schema
keywords fail instead of being silently ignored.

Every source case needs a stable case, task-cluster, and template identifier. Prompts must be
unique after Unicode NFKC, case folding, and token/whitespace normalization. Every gold call must
name a declared tool and satisfy its declared object schema. Both current versions are for
one-or-more tool-call action tasks; text-only abstention and trajectory adapters require a separate
explicit contract rather than an ambiguous null action.

The public frozen `case_id` is bound to the complete canonical case record. `task_cluster_id` is a
revision-namespaced hash of the upstream cluster ID; `template_id` additionally binds the upstream
template ID and derived skeleton. These are stable grouping IDs, not Merkle roots over every group
member: changing another member does not change the group ID. Raw upstream identifiers are
retained only as hashes plus the source array index in the frozen bundle.

## Freeze contract

The contract itself must be externally timestampable JSON. Paths may be absolute or relative to
the contract:

```json
{
  "kind": "localagent_fresh_external_action_eval_contract",
  "schema_version": 1,
  "source": {
    "path": "private/external-source.json",
    "bytes": 123456,
    "sha256": "<lowercase SHA-256>",
    "benchmark": "real-benchmark-name",
    "revision": "immutable-upstream-revision",
    "split": "heldout-split"
  },
  "limits": {
    "max_artifact_bytes": 8589934592,
    "max_source_bytes": 268435456,
    "max_record_bytes": 8388608,
    "max_source_cases": 10000
  },
  "selection": {
    "seed": "slmw2026-final-v1",
    "min_cases": 200,
    "max_cases": 256,
    "min_task_clusters": 100,
    "max_cases_per_task_cluster": 4,
    "max_cases_per_template": 2
  },
  "decontamination": {
    "shingle_size": 5,
    "min_shingles": 8,
    "min_coverage": 0.9,
    "anchors_per_entry": 8,
    "max_denylist_shingles": 2048
  },
  "training_artifacts": [
    {
      "stage": "pretrain",
      "name": "paper-pretrain-filtered",
      "format": "corpus_jsonl",
      "path": "data/shards/paper-all/filtered.jsonl",
      "records": 1000000,
      "bytes": 1234567890,
      "sha256": "<lowercase SHA-256>"
    },
    {
      "stage": "midtrain",
      "name": "paper-midtrain-agent",
      "format": "conversation_jsonl",
      "path": "data/synth/agent_midtrain.jsonl",
      "records": 5000,
      "bytes": 12345678,
      "sha256": "<lowercase SHA-256>"
    },
    {
      "stage": "sft",
      "name": "paper-sft-agent",
      "format": "conversation_jsonl",
      "path": "data/synth/agent_sft.jsonl",
      "records": 5000,
      "bytes": 12345678,
      "sha256": "<lowercase SHA-256>"
    }
  ],
  "analysis": {
    "bootstrap_resamples": 10000,
    "bootstrap_seed": 2026,
    "exact_action_noninferiority_margin": -0.02
  }
}
```

For v1, `training_artifacts` must cover pretrain, midtrain, and SFT. For v2 it must cover pretrain,
midtrain, SFT, and RL, and at least one `conversation_jsonl` declaration is mandatory for each of
midtrain, SFT, and RL. Every v2 conversation declaration adds:

```json
{"conversation_prompt_contract": "openai_full_catalog_v1"}
```

Legacy conversation rendering is intentionally unsupported by the v2 auditor: its exact training
materialization is not interchangeable with the full-catalog contract. Migrating a legacy-trained
checkpoint needs a separate, explicitly versioned renderer rather than a permissive fallback.

Every distinct text or canonical conversation input that affected the compared checkpoints must
be listed. Supported formats are:

- `corpus_jsonl`: one object per line with exactly one string-valued `text`, `content`, or `code`
  field; ambiguous multi-text-field rows fail closed;
- `conversation_jsonl`: one canonical `localagent.data.schema.Conversation` per line; and
- `text`: one bounded UTF-8 text file.

Packed token arrays are not a substitute for auditable source text.

V2 also requires `lineage_artifacts`, with one or more entries per stage:

```json
{
  "lineage_artifacts": [
    {
      "stage": "rl",
      "name": "candidate-rl-lineage",
      "path": "runs/candidate/rl-lineage.json",
      "bytes": 1234,
      "sha256": "<SHA-256>"
    }
  ]
}
```

Each bound sidecar has this exact shape:

```json
{
  "kind": "localagent_training_lineage_export",
  "schema_version": 1,
  "stage": "rl",
  "checkpoint_sha256": "<SHA-256 of the actual stage checkpoint>",
  "lineage": {
    "version": 1,
    "stage": "rl",
    "config_sha256": "<SHA-256>",
    "model_config_sha256": "<SHA-256>",
    "data_sha256": "<SHA-256>",
    "tokenizer_sha256": "<SHA-256>",
    "git": {
      "commit": "<40-hex Git commit>",
      "repository_sha256": "<SHA-256>",
      "dirty": false,
      "worktree_sha256": "<SHA-256>"
    },
    "parent_checkpoint_sha256": "<frozen SFT checkpoint SHA-256>"
  },
  "training_artifact_sha256": ["<every declared RL input SHA-256>"],
  "conversation_prompt_contract": "openai_full_catalog_v1"
}
```

The pretrain sidecar uses `null` for `conversation_prompt_contract` and has no parent. Midtrain,
SFT, and RL must carry `parent_checkpoint_sha256` inside `lineage`, resolving respectively to a
frozen pretrain, midtrain, and SFT sidecar. The sidecar's training hash set must exactly equal the
declarations for its stage. Multiple lineage sidecars per stage are allowed when baseline and
candidate use different chains; during v2 comparison, each actual checkpoint SHA-256 must occur in
a frozen RL sidecar.

This repository validates the sidecar but does not yet export it directly from a live checkpoint.
Production collection therefore needs a small checkpoint-to-sidecar migration step that copies
the already embedded LocalAgent lineage and hashes the same checkpoint descriptor. Do not hand-edit
or weaken the sidecar schema. Directory bundles likewise need a deterministic archive or
self-hashed bundle manifest represented as one regular file; v2 deliberately does not assign a
portable hash to a mutable directory tree.

## Commands

```bash
python scripts/fresh_action_eval.py freeze private/final-action-contract.json \
  --slice private/final-action-slice.json \
  --denylist private/final-action-prompts.json \
  --manifest private/final-action-freeze-manifest.json

# Later verification must reproduce all three outputs byte-for-byte.
python scripts/fresh_action_eval.py verify private/final-action-contract.json \
  --slice private/final-action-slice.json \
  --denylist private/final-action-prompts.json \
  --manifest private/final-action-freeze-manifest.json

# Only after external timestamping and paired model collection:
python scripts/fresh_action_eval.py compare \
  --manifest private/final-action-freeze-manifest.json \
  --slice private/final-action-slice.json \
  --baseline runs/final/baseline-actions.json \
  --candidate runs/final/structured-actions.json \
  --out runs/final/exact-action-comparison.json
```

Freeze outputs are canonical single-line JSON with trailing newline. Existing output files are
accepted only if they reproduce exactly; drift is never overwritten. The manifest hashes itself
by canonicalizing all fields except `manifest_self_sha256`. Hashed identities store contract
spelling for declared inputs and basenames for the contract/outputs, not absolute resolved paths,
so moving an otherwise identical bundle does not change manifest bytes.

## Paired result contracts

### Historical v1 normalized calls

The original unversioned result object is retained only for byte-replay compatibility:

```json
{
  "system": {
    "name": "structured",
    "checkpoint_sha256": "<SHA-256>",
    "bundle_sha256": "<SHA-256>"
  },
  "records": [
    {
      "case_id": "extcase-...",
      "task_cluster_id": "extcluster-...",
      "repetition": 0,
      "predicted_calls": [
        {
          "name": "tool_name",
          "arguments": {"argument": "grounded value"}
        }
      ],
      "success": true
    }
  ]
}
```

This is an adapter-normalized call record, not a raw result. It does not prove that the model
emitted a valid LocalAgent envelope, produced no outside text, avoided duplicate keys, or reached a
complete stop. Its claimed checkpoint/bundle hashes are not opened. A new non-raw collection must
at least use the explicit kind `localagent_external_normalized_call_result`, schema version 1; its
comparison kind is `localagent_fresh_external_normalized_call_comparison` and retains the same
limitation. Never present either lane as whole-output model accuracy.

### Hardened raw whole-output result

The v2 paper lane accepts only the separately versioned raw result kind:

```json
{
  "kind": "localagent_external_raw_model_output_result",
  "schema_version": 1,
  "system": {
    "name": "structured",
    "checkpoint": {
      "path": "runs/final/latest.pt",
      "bytes": 123456789,
      "sha256": "<actual file SHA-256>"
    },
    "bundle": {
      "path": "runs/final/webgpu-bundle.tar",
      "bytes": 12345678,
      "sha256": "<actual file SHA-256>"
    }
  },
  "records": [
    {
      "case_id": "extcase-...",
      "task_cluster_id": "extcluster-...",
      "repetition": 0,
      "raw_output": "<tool_call>{\"arguments\":{\"argument\":\"grounded value\"},\"name\":\"tool_name\"}</tool_call>",
      "finish_reason": "eos",
      "success": true
    }
  ]
}
```

Artifact paths are relative to the result file unless absolute. They must identify non-symlink
regular files and must still match the declared bytes and SHA-256 during comparison. The accepted
finish reasons are `eos`, `stop`, `length`, `max_new_tokens`, `runtime_error`, and `cancelled`;
only `eos` and `stop` are complete enough to score correct. A syntactically complete call with
`length` still scores incorrect.

Every frozen case must appear. Both systems must have identical case/repetition opportunities and
cluster assignments, with no duplicate case/repetition record. `success` is optional, but if
present it must agree with independently recomputed scoring. Raw output is never repaired:
`parse_tool_output` must accept the complete string, every call must satisfy the frozen recursive
catalog, and then call name/arguments are compared order-insensitively. The raw comparison kind is
`localagent_fresh_external_raw_output_action_comparison`.

Raw and normalized lanes cannot be mixed in one paired comparison. A v2 freeze manifest rejects
both legacy and explicit normalized-call results. It also requires each actual checkpoint hash to
appear in a frozen RL lineage sidecar.

## Statistical estimand and decision boundary

For each system, repetitions are averaged within a case. The point estimate is:

```text
mean over cases(candidate exact-action rate - baseline exact-action rate)
```

The deterministic percentile bootstrap draws whole `task_cluster_id` clusters with replacement.
All cases in a drawn cluster remain together, and the same cluster draw is applied to both
systems. It reports the candidate-minus-baseline point estimate, 95% interval, case and cluster
counts, cluster-size range, and seed.

The exact-action non-inferiority gate passes only when:

```text
95% interval lower bound > -0.02
```

This command deliberately does not promote a system. The separately frozen systems gate still
requires the structured policy's median-of-run p95 TTFA to be at least 15% lower under matched
hardware/runtime conditions. Both gates, the at-least-200-new-task requirement, and the real
external slice must pass.
