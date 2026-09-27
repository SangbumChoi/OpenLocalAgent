"""Export a trained LocalAgent checkpoint to a Hugging Face Hub model repo (Phase 9).

Builds a self-contained bundle — `config.json` (the ModelConfig), weights (`model.safetensors`,
falling back to `pytorch_model.bin`), the tokenizer when a BPE checkpoint records one, all agent
heads (`agent_heads.bin`), and a `README.md` model card — then optionally pushes it with
`huggingface_hub`.

The model is pure-PyTorch; byte-tier checkpoints need no tokenizer file, while BPE-tier
checkpoints include and hash their tokenizer.  The card documents loading via this repo's
`LocalAgentLM`/`ModelConfig` (no `transformers` dependency).

Push requires auth: an HF token via `--token`, the `HF_TOKEN` env var, or a cached `hf auth login`.
Without a token it only writes the local bundle and prints how to push.
"""

from __future__ import annotations

import json
import hashlib
import os
import shutil
from collections.abc import Mapping
from pathlib import Path

import torch

from localagent.model import ModelConfig

_CARD = """---
license: mit
library_name: pytorch
tags: [tool-calling, agent, tiny-llm, {tokenizer_tag}, on-device, from-scratch]
pipeline_tag: text-generation
---

# {name} — LocalAgent ({params:.2f}M params)

A **from-scratch, {tokenizer_label}** tool-calling agent model from
[LocalAgent](https://github.com/sangbumchoi/localagent). Pure PyTorch, **{params:.2f}M params**,
trained on CPU. It pairs a tiny decoder (GQA + RoPE + SwiGLU{recur}) with a **dual head**
(tool-selection classifier + pointer/copy argument head) and **prompt-grounded constrained
decoding** for reliable tool calls across {ntools} tools (general assistant, the Claude Code /
Codex coding surface, and computer-use / productivity tools), including parallel two-call turns.

## Architecture
- vocab {vocab} ({tokenizer_label}), d_model {d_model}, layers {n_layers}{loops}, heads {n_heads}/{n_kv_heads} (GQA), ffn {ffn}
- factorized embeddings: {factorized}
- tokenizer: `{tokenizer_file}` (SHA-256 `{tokenizer_sha256}`)

## Files
- `config.json` — `ModelConfig`
- `{weight_file}` — decoder weights
- `agent_heads.bin` — trained tool/pointer/route/dispatch heads plus dispatch metadata (optional)

## What it can do (use cases)
One {tokenizer_label} model that turns a natural-language turn into a grounded tool call — across an
assistant, a coding agent, computer-use/productivity apps, and **parallel two-call** turns:

| you say | it calls |
|---|---|
| "What's the weather in Cusco?" | `get_weather(city="Cusco")` |
| "What is 19 * 19 * 5?" | `calculator(expression="19*19*5")` |
| "Open the file bin/run.sh." | `read_file(path="bin/run.sh")` |
| "Grep for 'TODO'." | `grep_search(pattern="TODO")` |
| "Run the tests." | `run_tests()` |
| "Commit with message 'fix bug'." | `git_commit(message="fix bug")` |
| "Send an email to Greta." | `send_email(recipient="Greta")` |
| "Go to figma.com." | `open_url(url="figma.com")` |
| "Send a Slack message saying 'ship it'." | `slack_send(message="ship it")` |
| "Create a Jira ticket titled 'broken link'." | `jira_issue(summary="broken link")` |
| "Compose an email to Judy **and** search for how tall is Everest." | `send_email(recipient="Judy")` + `web_search(query="how tall is Everest")` |

Multi-turn coding (grounds a follow-up arg from a tool response):
`read_file(tests/test_api.py)` → result → `run_tests()` → "FAILED…" → fix.
At catalog scale (100s–1000s of tools) selection is done by **retrieval** (top-k) instead of a
fixed head. See the [LocalAgent repo](https://github.com/sangbumchoi/localagent).

## Load (pure PyTorch, no transformers)
```python
import json, torch
from huggingface_hub import hf_hub_download
from localagent.model import LocalAgentLM, ModelConfig

cfg_d = json.load(open(hf_hub_download("{repo}", "config.json")))
cfg = ModelConfig(**{{k: v for k, v in cfg_d.items() if k in ModelConfig.__dataclass_fields__}})
model = LocalAgentLM(cfg)
{weight_loading}
model.eval()
# BPE checkpoints also ship tokenizer.json; byte-tier checkpoints use ByteTokenizer().
tokenizer_meta = cfg_d.get("tokenizer", {{}})
if tokenizer_meta.get("kind") == "bpe":
    from localagent.model.tokenizer import load_tokenizer
    tokenizer = load_tokenizer("bpe", hf_hub_download("{repo}", tokenizer_meta["filename"]))
```
See the LocalAgent repo for the grounded decoder / agent runtime (tool head, pointer head,
retrieval, parallel-call decode).
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokenizer_bundle(
    checkpoint: Mapping[str, object],
    checkpoint_path: Path,
    out_dir: Path,
) -> dict[str, str | None]:
    """Resolve, verify, and copy a recorded BPE tokenizer into the HF bundle."""

    metadata = checkpoint.get("tokenizer")
    if not isinstance(metadata, Mapping) or metadata.get("kind", "byte") == "byte":
        return {
            "kind": "byte",
            "label": "byte-level",
            "tag": "byte-level",
            "filename": None,
            "sha256": None,
        }
    kind = metadata.get("kind")
    if kind != "bpe":
        raise ValueError(f"unsupported checkpoint tokenizer kind: {kind!r}")
    raw_path = metadata.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("BPE checkpoint tokenizer metadata must contain path")
    recorded = Path(raw_path)
    candidates = [
        recorded,
        checkpoint_path.parent / recorded,
        Path(__file__).resolve().parents[4] / recorded,
    ]
    source = next((candidate for candidate in candidates if candidate.is_file()), None)
    if source is None:
        raise FileNotFoundError(f"recorded BPE tokenizer is missing: {raw_path}")
    digest = _sha256(source)
    declared = metadata.get("sha256")
    if declared is not None and declared != digest:
        raise ValueError(
            f"recorded BPE tokenizer SHA-256 mismatch: declared={declared}, observed={digest}"
        )
    filename = "tokenizer.json"
    shutil.copyfile(source, out_dir / filename)
    return {
        "kind": "bpe",
        "label": "byte-fallback BPE",
        "tag": "bpe",
        "filename": filename,
        "sha256": digest,
    }


def _weight_loading(repo_id: str, weight_file: str) -> str:
    """Return a model-card loader matching the weight file emitted by this export."""

    if weight_file == "model.safetensors":
        return (
            "from safetensors.torch import load_file\n"
            f"model.load_state_dict(load_file(hf_hub_download(\"{repo_id}\", \"{weight_file}\")))"
        )
    return (
        f"weights = torch.load(hf_hub_download(\"{repo_id}\", \"{weight_file}\"), "
        "map_location=\"cpu\", weights_only=True)\n"
        "model.load_state_dict(weights)"
    )


def _dispatch_metadata(checkpoint: Mapping[str, object]) -> dict[str, object]:
    """Return serialized dispatch metadata, inferring the standard pool when safe.

    Some continuation checkpoints predate the optional ``dispatch_tool_pool`` field but still
    contain the complete action heads and a 51-class tool head (50 standard tools plus abstention).
    In that narrowly identifiable case the runtime pool is the same 63-name standard/mobile/
    productivity catalog used by the WebGPU exporter.  Unknown head widths remain metadata-free
    rather than being guessed.
    """

    dispatch_pool = checkpoint.get("dispatch_tool_pool")
    inferred = False
    if not isinstance(dispatch_pool, list):
        tool_head = checkpoint.get("tool_head")
        bias = tool_head.get("fc.bias") if isinstance(tool_head, Mapping) else None
        width = getattr(bias, "shape", ())[0] if getattr(bias, "shape", ()) else None
        if width == 51:
            from localagent.agent.mobile_toolset import mobile_tools, realistic_productivity_tools
            from localagent.agent.toolset import STANDARD_TOOLS

            dispatch_pool = [
                tool.name
                for tool in [*STANDARD_TOOLS, *mobile_tools(), *realistic_productivity_tools()]
            ]
            inferred = True

    if not isinstance(dispatch_pool, list):
        return {}
    metadata: dict[str, object] = {
        "tool_pool": [str(name) for name in dispatch_pool],
        "ptr_args": [str(name) for name in checkpoint.get("ptr_args", [])]
        if isinstance(checkpoint.get("ptr_args", []), list)
        else [],
        "examples": checkpoint.get("examples", {})
        if isinstance(checkpoint.get("examples", {}), Mapping)
        else {},
        "retrieval_examples": checkpoint.get("retrieval_examples", {})
        if isinstance(checkpoint.get("retrieval_examples", {}), Mapping)
        else {},
    }
    if inferred:
        metadata["provenance"] = "inferred_standard_tool_pool_from_51_class_tool_head"
    return metadata


def export_hf(checkpoint: str, out_dir: str, repo_id: str | None = None, token: str | None = None,
              private: bool = True, push: bool = False) -> str:
    checkpoint_path = Path(checkpoint).resolve()
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg_d = ck["cfg"] if isinstance(ck["cfg"], dict) else ck["cfg"].__dict__
    cfg = ModelConfig(**{k: v for k, v in cfg_d.items() if k in ModelConfig.__dataclass_fields__})
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    tokenizer = _tokenizer_bundle(ck, checkpoint_path, out_path)

    dispatch_metadata = _dispatch_metadata(ck)
    config = {
        "model_type": "localagent",
        "architecture": f"LocalAgentLM ({tokenizer['label']} GQA+RoPE+SwiGLU)",
        "parameter_count": cfg.estimate_params(),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "tokenizer": {
            "kind": tokenizer["kind"],
            "filename": tokenizer["filename"],
            "sha256": tokenizer["sha256"],
        },
        **cfg_d,
    }
    if dispatch_metadata:
        config["agent"] = dispatch_metadata
    sd = ck["state_dict"]
    try:
        from safetensors.torch import save_file
        weight_file = "model.safetensors"
        save_file({k: v.contiguous() for k, v in sd.items()}, str(out_path / weight_file))
    except Exception:
        weight_file = "pytorch_model.bin"
        torch.save(sd, out_path / weight_file)

    weight_path = out_path / weight_file
    config["weights"] = {
        "filename": weight_file,
        "bytes": weight_path.stat().st_size,
        "sha256": _sha256(weight_path),
    }
    (out_path / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    heads = {
        key: ck[key]
        for key in ("tool_head", "ptr_head", "route_head", "dense_selector", "selector_proj")
        if key in ck and ck[key] is not None
    }
    if dispatch_metadata:
        heads.update(dispatch_metadata)
    if heads:
        torch.save(heads, out_path / "agent_heads.bin")

    card = _CARD.format(
        name=cfg.name, params=cfg.estimate_params() / 1e6, vocab=cfg.vocab_size,
        d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
        ffn=cfg.ffn_hidden, factorized=cfg.factorized,
        recur=(" + depth-recurrence" if cfg.n_loops > 1 else ""),
        loops=(f" x{cfg.n_loops} loops" if cfg.n_loops > 1 else ""),
        ntools=len(dispatch_metadata.get("tool_pool", [])) or 0,
        repo=repo_id or "<your-repo>", tokenizer_label=tokenizer["label"],
        tokenizer_tag=tokenizer["tag"], tokenizer_file=tokenizer["filename"] or "built-in",
        tokenizer_sha256=tokenizer["sha256"] or "not-applicable",
        weight_file=weight_file,
        weight_loading=_weight_loading(repo_id or "<your-repo>", weight_file),
    )
    (out_path / "README.md").write_text(card, encoding="utf-8")

    if not push:
        return out_dir
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("push=True but no token (pass --token or set HF_TOKEN / hf auth login)")
    from huggingface_hub import HfApi, create_repo
    create_repo(repo_id, token=token, private=private, exist_ok=True, repo_type="model")
    HfApi().upload_folder(folder_path=str(out_path), repo_id=repo_id, token=token, repo_type="model")
    return f"https://huggingface.co/{repo_id}"


def export(checkpoint: str, out_path: str) -> None:  # uniform with other export/* modules
    print(export_hf(checkpoint, out_path, push=False))
