"""Export the model to ONNX (Phase 9) — enables onnxruntime / onnxruntime-web (WASM + WebGPU).

Exports final-token LM logits for ordinary decoding and separate browser/action bundles where full
hidden sequences are required. Cached prefill/decode graphs expose final-token logits plus a
compatibility greedy token and recurrent state. Every cached graph is parity-gated against the
exact PyTorch model before its provenance is published.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from localagent.model import LocalAgentLM, ModelConfig


# Visual-prefix models need a separate image-input/export ABI.  Refuse to silently export only the
# text path until that contract is implemented and parity-tested.
def _assert_text_exportable(cfg: ModelConfig) -> None:
    if cfg.vision_enabled:
        raise ValueError(
            "vision-enabled checkpoints require the screenshot export ABI; text-only ONNX/WebGPU "
            "export is intentionally disabled"
        )


class _LogitsOnly(nn.Module):
    """ONNX-friendly wrapper: token ids -> final-token logits ``[batch, vocab]``."""

    def __init__(self, model: LocalAgentLM):
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.model(idx)[0][:, -1, :]


class _LogitsAndHidden(nn.Module):
    """ONNX wrapper exposing BOTH outputs the browser bundle needs:

    * ``logits``  (B, T, vocab)   — next-byte distribution for decoding
    * ``hidden``  (B, T, d_model) — the post-final-norm features the tool/pointer heads read
      (== ``feats`` from ``LocalAgentLM.forward(..., return_hidden=True)``).
    """

    def __init__(self, model: LocalAgentLM):
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor):
        logits, hidden = self.model(idx, return_hidden=True)
        return logits, hidden


class _HiddenOnly(nn.Module):
    """ONNX wrapper for action inference: token ids -> normalized backbone features only."""

    def __init__(self, model: LocalAgentLM):
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(idx)


def export(checkpoint: str, out_path: str, opset: int = 17, check: bool = True) -> None:
    ck, _checkpoint_identity = _load_weights_only_checkpoint(checkpoint)
    cfg_d = ck["cfg"] if isinstance(ck["cfg"], dict) else ck["cfg"].__dict__
    cfg = ModelConfig(**{k: v for k, v in cfg_d.items() if k in ModelConfig.__dataclass_fields__})
    _assert_text_exportable(cfg)
    model = LocalAgentLM(cfg).eval()
    model.load_state_dict(ck["state_dict"])
    wrap = _LogitsOnly(model).eval()

    dummy = torch.randint(0, cfg.vocab_size, (1, min(16, cfg.max_seq_len)))
    torch.onnx.export(
        wrap,
        (dummy,),
        out_path,
        opset_version=opset,
        dynamo=False,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
    )
    print(f"wrote {out_path}")

    if check:
        import numpy as np
        import onnxruntime as ort

        with torch.no_grad():
            ref = wrap(dummy).numpy()
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        got = sess.run(["logits"], {"input_ids": dummy.numpy()})[0]
        diff = float(np.abs(ref - got).max())
        print(f"parity vs PyTorch: max|Δ|={diff:.2e} ({'OK' if diff < 1e-3 else 'CHECK'})")
        # argmax agreement (what actually matters for decoding)
        agree = (ref.argmax(-1) == got.argmax(-1)).mean()
        print(f"argmax agreement: {agree * 100:.1f}%")


# --------------------------------------------------------------------------------------------
# Browser bundle (onnxruntime-web: WASM + WebGPU)
# --------------------------------------------------------------------------------------------


def _round_nested(x, dp: int = 6):
    """Round a (possibly nested) python list of floats to ``dp`` decimals for JSON size."""
    if isinstance(x, list):
        return [_round_nested(v, dp) for v in x]
    return round(float(x), dp)


def _heads_json(ck: dict) -> dict:
    """Serialize tool_head + pointer_head weights as plain nested arrays (no onnx needed in JS).

    tool_head: a single Linear (d_model -> 22). JS computes  hidden[:, -1] @ W.T + b.
    pointer_head: per-arg copy mechanism. For arg a, with query q = arg_emb[a] (d_model):
        start_logit[t] = hidden[t] . (start_W @ q)   (i.e. hidden @ (start_W @ q))
        end_logit[t]   = hidden[t] . (end_W   @ q)
    We export arg_emb, start_W, end_W raw; JS reproduces the two matvecs + einsum.
    """
    from localagent.agent.pointer_head import PTR_ARGS
    from localagent.agent.tool_head import CLASSES

    th = ck["tool_head"]
    ph = ck["ptr_head"]
    ptr_args = list(ck.get("ptr_args", PTR_ARGS))
    ptr_arg_idx = {arg: index for index, arg in enumerate(ptr_args)}
    expected_rows = len(ptr_args)
    actual_rows = int(ph["arg_emb.weight"].shape[0])
    if actual_rows != expected_rows:
        raise ValueError(
            "checkpoint pointer argument metadata disagrees with embedding rows: "
            f"args={expected_rows}, rows={actual_rows}"
        )
    tool_w = th["fc.weight"].cpu().tolist()  # (22, d_model)
    tool_b = th["fc.bias"].cpu().tolist()  # (22,)
    stop_index = CLASSES.index("text")

    return {
        "tool_head": {
            # weight[c] is the row for class c; logit_c = sum_d hidden[d]*weight[c][d] + bias[c]
            "weight": _round_nested(tool_w),  # shape (22, d_model)
            "bias": _round_nested(tool_b),  # shape (22,)
            "classes": list(CLASSES),  # ordered, len 22
            "stop_index": stop_index,  # index of "text" (no-tool / planner STOP)
        },
        "pointer_head": {
            # arg_emb[i] is the query vector for PTR_ARGS[i]; shape (n_args, d_model)
            "arg_emb": _round_nested(ph["arg_emb.weight"].cpu().tolist()),
            # start_W / end_W are (d_model, d_model). projected query qp = start_W @ arg_emb[i].
            # then start_logit[t] = hidden[t] . qp_start ;  end_logit[t] = hidden[t] . qp_end.
            # (matches PointerHead.logits: einsum('btd,bd->bt', feats, start(q)).)
            "start_W": _round_nested(ph["start.weight"].cpu().tolist()),  # (d_model, d_model)
            "end_W": _round_nested(ph["end.weight"].cpu().tolist()),  # (d_model, d_model)
            "args": ptr_args,
            "arg_idx": ptr_arg_idx,
            # decode rule JS must apply: start = argmax(start_logit);
            #   end_logit[:start] = -inf; end = argmax(end_logit); span is [start, end] inclusive.
        },
    }


def _meta_json(
    cfg: ModelConfig,
    tokenizer,
    *,
    tokenizer_file: str | None = None,
    model_file: str | None = None,
    action_model_file: str | None = None,
    model_parameters: int | None = None,
    tools=None,
) -> dict:
    """Tokenization, model, and toolset contract consumed by the JavaScript runtime."""
    from localagent.agent.tool_head import CLASSES
    from localagent.agent.toolset import STANDARD_TOOLS
    from localagent.model import tokenizer as tk

    is_byte = cfg.vocab_size == 256

    markers = {
        "user": {"text": tk.USER, "ids": tokenizer.encode(tk.USER)},
        "assistant": {"text": tk.ASSISTANT, "ids": tokenizer.encode(tk.ASSISTANT)},
        "tool": {"text": tk.TOOL, "ids": tokenizer.encode(tk.TOOL)},
        "tool_call_open": {"text": tk.TOOL_CALL_OPEN, "ids": tokenizer.encode(tk.TOOL_CALL_OPEN)},
        "tool_call_close": {
            "text": tk.TOOL_CALL_CLOSE,
            "ids": tokenizer.encode(tk.TOOL_CALL_CLOSE),
        },
        "tool_response_open": {
            "text": tk.TOOL_RESPONSE_OPEN,
            "ids": tokenizer.encode(tk.TOOL_RESPONSE_OPEN),
        },
        "tool_response_close": {
            "text": tk.TOOL_RESPONSE_CLOSE,
            "ids": tokenizer.encode(tk.TOOL_RESPONSE_CLOSE),
        },
    }
    tool_specs = STANDARD_TOOLS if tools is None else list(tools)
    tool_rows = []
    for t in tool_specs:
        props = t.parameters.get("properties", {})
        tool_rows.append(
            {
                "name": t.name,
                "description": t.description,
                "args": list(props.keys()),
                "schema": t.parameters,
            }
        )
    meta = {
        "vocab_size": cfg.vocab_size,
        "d_model": cfg.d_model,
        "max_seq_len": cfg.max_seq_len,
        "pad_id": tokenizer.pad_id,
        "eos_id": tokenizer.eos_id,
        "encoding": "utf-8-bytes" if is_byte else "bytelevel-bpe",
        "markers": markers,
        "tools": tool_rows,
        "tool_classes": list(CLASSES),
    }
    if tokenizer_file is not None:
        meta["tokenizer_file"] = tokenizer_file
    if model_file is not None:
        meta["model_file"] = model_file
    if action_model_file is not None:
        meta["action_model_file"] = action_model_file
    if model_parameters is not None:
        meta["model_parameters"] = model_parameters
    return meta


def _load_web_tokenizer(cfg: ModelConfig, tokenizer_path: str | None):
    """Load and validate the tokenizer contract before writing any bundle artifacts."""
    from localagent.model import tokenizer as tk

    if cfg.vocab_size == 256:
        return tk.ByteTokenizer()
    if tokenizer_path is None:
        raise ValueError(
            "export_web requires tokenizer_path for non-byte checkpoints "
            f"(vocab_size={cfg.vocab_size})"
        )
    try:
        tokenizer = tk.BPETokenizer.from_file(tokenizer_path)
    except Exception as exc:
        raise ValueError(f"could not load BPE tokenizer from {tokenizer_path!r}") from exc
    if tokenizer.vocab_size != cfg.vocab_size:
        raise ValueError(
            "BPE tokenizer vocabulary does not match model config: "
            f"tokenizer={tokenizer.vocab_size}, model={cfg.vocab_size}"
        )
    for marker in tk.SPECIAL_MARKERS:
        marker_id = tokenizer._tokenizer.token_to_id(marker)
        marker_ids = tokenizer.encode(marker)
        if marker_id is None or marker_ids != [marker_id]:
            raise ValueError(
                f"BPE tokenizer must encode agent marker {marker!r} as one registered token"
            )
    return tokenizer


def _convert_web_fp16(fp32_path: str, fp16_path: str) -> None:
    """Convert and pre-optimize a web graph while preserving fp32 output tensors."""
    import onnx
    import onnxruntime as ort
    from onnxruntime.transformers.float16 import convert_float_to_float16

    m32 = onnx.load(fp32_path)
    # input_ids stays int64; floating outputs are cast back to fp32 at the graph boundary so JS
    # reads ordinary Float32Array tensors regardless of internal fp16.
    m16 = convert_float_to_float16(m32, keep_io_types=True)
    # The raw fp16 graph collides with ORT's SimplifiedLayerNormFusion (a fp32 norm Constant left
    # dangling after cast insertion) and FAILS to load at ORT_ENABLE_ALL — the web default.
    # Pre-optimizing at EXTENDED bakes the working RMSNorm fusion into the portable graph.
    raw16 = f"{fp16_path}.raw"
    onnx.save(m16, raw16)
    try:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        so.optimized_model_filepath = fp16_path
        ort.InferenceSession(raw16, so, providers=["CPUExecutionProvider"])
    finally:
        if os.path.exists(raw16):
            os.remove(raw16)


_WEB_PARITY_THRESHOLDS = {
    "fp32": {
        "hidden_max_abs_diff": 1e-3,
        "logits_max_abs_diff": 1e-3,
    },
    "fp16": {
        "hidden_max_abs_diff": 5e-2,
        "logits_max_abs_diff": 1e-1,
    },
}


def _web_parity_fixtures(cfg: ModelConfig) -> list[torch.Tensor]:
    """Return deterministic, content-bound fixtures for the deployable browser graphs."""
    lengths = sorted({min(length, cfg.max_seq_len) for length in (1, 8, 16)})
    return [
        (torch.arange(length, dtype=torch.int64) * 131 + 17 + fixture_index * 977)
        .remainder(cfg.vocab_size)
        .unsqueeze(0)
        for fixture_index, length in enumerate(lengths)
    ]


def _stable_regular_file(
    path: str | Path,
    *,
    load_checkpoint: bool,
) -> tuple[Any | None, dict[str, Any]]:
    """Hash one descriptor-bound regular file and optionally safe-load that exact snapshot."""

    source = os.fspath(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except FileNotFoundError:
        raise FileNotFoundError(f"required bundle artifact is missing: {source}") from None
    except OSError as error:
        raise ValueError(
            f"required bundle artifact must be a non-symlink regular file: {source}"
        ) from error

    digest = hashlib.sha256()
    checkpoint = None
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"required bundle artifact is not a regular file: {source}")
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        if load_checkpoint:
            handle.seek(0)
            checkpoint = torch.load(handle, map_location="cpu", weights_only=True)
        after = os.fstat(handle.fileno())

    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise RuntimeError(f"bundle artifact changed while reading: {source}")
    try:
        path_after = os.lstat(source)
    except FileNotFoundError:
        raise RuntimeError(f"bundle artifact path disappeared while reading: {source}") from None
    if stat.S_ISLNK(path_after.st_mode) or any(
        getattr(after, field) != getattr(path_after, field)
        for field in ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    ):
        raise RuntimeError(f"bundle artifact path changed while reading: {source}")
    identity = {
        "bytes": after.st_size,
        "sha256": digest.hexdigest(),
    }
    return checkpoint, identity


def _bundle_artifact_identity(path: str) -> dict[str, Any]:
    """Hash one descriptor-bound stable regular-file snapshot."""

    _checkpoint, identity = _stable_regular_file(path, load_checkpoint=False)
    return identity


def _load_weights_only_checkpoint(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Safe-load and hash the same descriptor-bound checkpoint snapshot."""

    checkpoint, identity = _stable_regular_file(path, load_checkpoint=True)
    return checkpoint, identity


def _web_graph_parity(
    model: LocalAgentLM,
    graph_path: str,
    fixtures: Sequence[torch.Tensor],
    *,
    expected_outputs: Sequence[str],
    precision: str,
) -> dict[str, Any]:
    """Fail closed unless one emitted ONNX graph matches the exact PyTorch reference."""
    import numpy as np
    import onnxruntime as ort

    if precision not in {"fp32", "fp16"}:
        raise ValueError(f"unknown parity precision {precision!r}")
    artifact_before = _bundle_artifact_identity(graph_path)
    thresholds = {
        output_name: _WEB_PARITY_THRESHOLDS[precision][f"{output_name}_max_abs_diff"]
        for output_name in expected_outputs
    }
    session = ort.InferenceSession(graph_path, providers=["CPUExecutionProvider"])
    actual_outputs = list(session.get_outputs())
    actual_output_names = [output.name for output in actual_outputs]
    if actual_output_names != list(expected_outputs):
        raise RuntimeError(
            f"{os.path.basename(graph_path)} output contract mismatch: "
            f"expected {list(expected_outputs)}, got {actual_output_names}"
        )

    per_fixture = []
    maxima = {name: 0.0 for name in expected_outputs}
    argmax_agreements = []
    with torch.no_grad():
        for fixture in fixtures:
            ref_logits, ref_hidden = model(fixture, return_hidden=True)
            references = {
                "logits": ref_logits.numpy(),
                "hidden": ref_hidden.numpy(),
            }
            exported = session.run(list(expected_outputs), {"input_ids": fixture.numpy()})
            fixture_result: dict[str, Any] = {
                "input_ids_sha256": hashlib.sha256(fixture.numpy().tobytes()).hexdigest(),
                "sequence_length": int(fixture.shape[1]),
                "outputs": {},
            }
            for output_name, value in zip(expected_outputs, exported):
                reference = references[output_name]
                if value.shape != reference.shape:
                    raise RuntimeError(
                        f"{os.path.basename(graph_path)} {output_name} shape mismatch: "
                        f"PyTorch={reference.shape}, ONNX={value.shape}"
                    )
                max_abs_diff = float(np.abs(reference - value).max())
                if not np.isfinite(max_abs_diff):
                    raise RuntimeError(
                        f"{os.path.basename(graph_path)} {output_name} parity was non-finite"
                    )
                maxima[output_name] = max(maxima[output_name], max_abs_diff)
                fixture_result["outputs"][output_name] = {
                    "max_abs_diff": max_abs_diff,
                }
                if output_name == "logits":
                    agreement = float((reference.argmax(axis=-1) == value.argmax(axis=-1)).mean())
                    fixture_result["outputs"][output_name]["argmax_agreement"] = agreement
                    argmax_agreements.append(agreement)
            per_fixture.append(fixture_result)

    for output_name, max_abs_diff in maxima.items():
        threshold = thresholds[output_name]
        if max_abs_diff > threshold:
            raise RuntimeError(
                f"{os.path.basename(graph_path)} failed {precision} parity: "
                f"{output_name} max_abs_diff={max_abs_diff:.6g} exceeds {threshold:.6g}"
            )
    artifact_after = _bundle_artifact_identity(graph_path)
    if artifact_after != artifact_before:
        raise RuntimeError(
            f"{os.path.basename(graph_path)} changed while parity was being evaluated"
        )
    return {
        "artifact": artifact_after,
        "expected_outputs": list(expected_outputs),
        "max_abs_diff_by_output": maxima,
        "minimum_logits_argmax_agreement": (min(argmax_agreements) if argmax_agreements else None),
        "passed": True,
        "per_fixture": per_fixture,
        "precision": precision,
        "provider": "CPUExecutionProvider",
        "threshold_max_abs_diff": max(thresholds.values()),
        "threshold_max_abs_diff_by_output": thresholds,
    }


def _web_parity_gate(
    model: LocalAgentLM,
    graphs: dict[str, tuple[str | None, Sequence[str], str]],
    cfg: ModelConfig,
) -> dict[str, Any]:
    """Validate every emitted graph before the browser bundle manifest can be published."""
    fixtures = _web_parity_fixtures(cfg)
    results = {}
    for artifact_name, (path, outputs, precision) in graphs.items():
        if path is None:
            raise FileNotFoundError(f"declared graph artifact {artifact_name} has no export path")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"declared graph artifact {artifact_name} is missing: {path}")
        results[artifact_name] = _web_graph_parity(
            model,
            path,
            fixtures,
            expected_outputs=outputs,
            precision=precision,
        )
    if not results:
        raise RuntimeError("browser export parity gate received no graphs")
    return {
        "fixture_contract": "ids[i]=(131*i+17+977*fixture_index) mod vocab_size",
        "hard_gate": True,
        "passed": True,
        "reference": "the exact LocalAgentLM checkpoint loaded for this export",
        "results": results,
        "schema_version": 1,
        "thresholds": {
            precision: dict(thresholds) for precision, thresholds in _WEB_PARITY_THRESHOLDS.items()
        },
    }


def _write_bundle_manifest(
    out_dir: str,
    *,
    config_name: str,
    model_parameters: int,
    checkpoint_sha256: str,
    checkpoint_stage: str | None,
    checkpoint_step: int | None,
    model_config_sha256: str,
    artifacts: dict[str, str | None],
    parity_gate: dict[str, Any],
) -> str:
    """Write immutable artifact provenance without recursively hashing the manifest itself."""
    if parity_gate.get("hard_gate") is not True or parity_gate.get("passed") is not True:
        raise RuntimeError("refusing to publish a bundle manifest without passed hard parity")
    entries = {}
    for name, path in sorted(artifacts.items()):
        if path is None:
            raise FileNotFoundError(f"declared bundle artifact {name} has no export path")
        identity = _bundle_artifact_identity(path)
        entries[name] = {
            "file": os.path.basename(path),
            **identity,
        }
    parity_results = parity_gate.get("results")
    if not isinstance(parity_results, dict) or not parity_results:
        raise RuntimeError("refusing to publish a manifest without per-graph parity results")
    declared_graphs = {name for name in artifacts if name.endswith(".onnx")}
    if set(parity_results) != declared_graphs:
        raise RuntimeError(
            "parity results must cover every declared ONNX graph exactly: "
            f"declared={sorted(declared_graphs)}, tested={sorted(parity_results)}"
        )
    for graph_name, result in parity_results.items():
        artifact = entries.get(graph_name)
        if artifact is None:
            raise FileNotFoundError(
                f"parity-tested graph {graph_name} is not a declared bundle artifact"
            )
        tested = result.get("artifact")
        observed = {"bytes": artifact["bytes"], "sha256": artifact["sha256"]}
        if tested != observed:
            raise RuntimeError(
                f"bundle graph {graph_name} changed after parity: "
                f"tested={tested}, observed={observed}"
            )
    manifest = {
        "schema_version": 3,
        "config_name": config_name,
        "model_parameters": model_parameters,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_stage": checkpoint_stage,
        "checkpoint_step": checkpoint_step,
        "model_config_sha256": model_config_sha256,
        "artifacts": entries,
        "parity_gate": parity_gate,
    }
    path = os.path.join(out_dir, "bundle-manifest.json")
    temporary_path = f"{path}.tmp"
    try:
        with open(temporary_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")
        # Close the hash-to-publish race as far as a filesystem artifact protocol can: re-hash
        # every declared file immediately before the atomic manifest replacement.
        for name, artifact_path in sorted(artifacts.items()):
            observed = _bundle_artifact_identity(str(artifact_path))
            expected = {
                "bytes": entries[name]["bytes"],
                "sha256": entries[name]["sha256"],
            }
            if observed != expected:
                raise RuntimeError(f"bundle artifact {name} changed before manifest publication")
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    print(f"wrote {path}")
    return path


def export_web(
    checkpoint: str,
    out_dir: str,
    fp16: bool = True,
    opset: int = 17,
    check: bool = True,
    action_only: bool = False,
    tokenizer_path: str | None = None,
    tools=None,
) -> dict:
    """Emit a browser-ready onnxruntime-web bundle (WASM + WebGPU) into ``out_dir``:

      * ``model.onnx``       — fp32, single graph, inputs ``input_ids`` (int64 [batch, seq]),
                               outputs ``logits`` ([batch, seq, vocab]) AND ``hidden``
                               ([batch, seq, d_model], the last hidden state the heads read).
      * ``model.fp16.onnx``  — fp16 variant (the web default), only when ``fp16=True``.
      * ``heads.json``       — tool_head + pointer_head weights as nested arrays (apply in JS).
      * ``meta.json``        — vocab/d_model/pad_id, tokenizer-aware markers, standard toolset.
      * ``tokenizer.json``   — validated ByteLevel BPE tokenizer for vocabularies larger than 256.

    With ``action_only=True``, the bundle additionally contains ``action_model.onnx`` (and an fp16
    variant when requested). That graph exposes only ``hidden`` and never evaluates the LM output
    projection. The original logits+hidden artifacts remain unchanged for decoding compatibility.

    Byte checkpoints preserve the zero-asset UTF-8 tokenizer. Non-byte checkpoints require
    ``tokenizer_path`` to point to a loadable :class:`BPETokenizer` with the exact model vocabulary.

    Every emitted ONNX graph is parity-gated against the exact loaded PyTorch checkpoint before
    ``bundle-manifest.json`` is published. ``check`` is retained as a compatibility flag controlling
    parity logging only; setting it to ``False`` never disables the hard gate.

    Returns a dict of artifact paths + sizes + parity evidence.
    """
    ck, checkpoint_identity = _load_weights_only_checkpoint(checkpoint)
    cfg_d = ck["cfg"] if isinstance(ck["cfg"], dict) else ck["cfg"].__dict__
    config_digest = hashlib.sha256(
        json.dumps(cfg_d, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    cfg = ModelConfig(**{k: v for k, v in cfg_d.items() if k in ModelConfig.__dataclass_fields__})
    _assert_text_exportable(cfg)
    tokenizer = _load_web_tokenizer(cfg, tokenizer_path)
    # A tokenizer with the right vocabulary size can still assign completely different IDs.  The
    # model's recorded tokenizer hash is therefore part of the deployment contract, not merely
    # metadata.  Refuse to emit any bundle when an explicit BPE asset does not match it.
    if cfg.vocab_size != 256:
        _checkpoint_tokenizer_provenance(ck, cfg, tokenizer_path=tokenizer_path)
    os.makedirs(out_dir, exist_ok=True)
    # A failed re-export must not leave a previously published manifest beside new, unchecked bytes.
    stale_manifest = os.path.join(out_dir, "bundle-manifest.json")
    if os.path.exists(stale_manifest):
        os.remove(stale_manifest)
    model = LocalAgentLM(cfg).eval()
    model.load_state_dict(ck["state_dict"])
    wrap = _LogitsAndHidden(model).eval()

    fp32_path = os.path.join(out_dir, "model.onnx")
    dummy = _web_parity_fixtures(cfg)[-1]
    torch.onnx.export(
        wrap,
        (dummy,),
        fp32_path,
        opset_version=opset,
        dynamo=False,
        input_names=["input_ids"],
        output_names=["logits", "hidden"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
            "hidden": {0: "batch", 1: "seq"},
        },
    )
    print(f"wrote {fp32_path}")

    fp16_path = None
    if fp16:
        fp16_path = os.path.join(out_dir, "model.fp16.onnx")
        _convert_web_fp16(fp32_path, fp16_path)
        print(f"wrote {fp16_path}")

    action_path = None
    action_fp16_path = None
    if action_only:
        action_wrap = _HiddenOnly(model).eval()
        action_path = os.path.join(out_dir, "action_model.onnx")
        torch.onnx.export(
            action_wrap,
            (dummy,),
            action_path,
            opset_version=opset,
            dynamo=False,
            input_names=["input_ids"],
            output_names=["hidden"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "hidden": {0: "batch", 1: "seq"},
            },
        )
        print(f"wrote {action_path}")
        if fp16:
            action_fp16_path = os.path.join(out_dir, "action_model.fp16.onnx")
            _convert_web_fp16(action_path, action_fp16_path)
            print(f"wrote {action_fp16_path}")

    bundled_tokenizer_path = None
    bundled_tokenizer_file = None
    if cfg.vocab_size != 256:
        bundled_tokenizer_file = "tokenizer.json"
        bundled_tokenizer_path = os.path.join(out_dir, bundled_tokenizer_file)
        if os.path.realpath(str(tokenizer_path)) != os.path.realpath(bundled_tokenizer_path):
            shutil.copyfile(str(tokenizer_path), bundled_tokenizer_path)
        print(f"wrote {bundled_tokenizer_path}")

    action_model_file = None
    if action_path is not None:
        action_model_file = os.path.basename(action_fp16_path or action_path)
    model_file = os.path.basename(fp16_path or fp32_path)
    heads = _heads_json(ck)
    meta = _meta_json(
        cfg,
        tokenizer,
        tokenizer_file=bundled_tokenizer_file,
        model_file=model_file,
        action_model_file=action_model_file,
        model_parameters=model.num_params(),
        tools=tools,
    )
    heads_path = os.path.join(out_dir, "heads.json")
    meta_path = os.path.join(out_dir, "meta.json")
    with open(heads_path, "w") as f:
        json.dump(heads, f)
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"wrote {heads_path}  {meta_path}")

    stats = {
        "model.onnx": fp32_path,
        "model.fp16.onnx": fp16_path,
        "heads.json": heads_path,
        "meta.json": meta_path,
    }
    if action_path is not None:
        stats["action_model.onnx"] = action_path
        stats["action_model.fp16.onnx"] = action_fp16_path
    if bundled_tokenizer_path is not None:
        stats["tokenizer.json"] = bundled_tokenizer_path

    # Dispatch heads (route_head + dense_selector query tower + precomputed tool matrix), when the
    # checkpoint carries them. Same JS recipe as tool_head: apply to the onnx `hidden[:, -1]`.
    if ck.get("route_head") and ck.get("dense_selector"):
        from localagent.inference.export.to_dispatch import dispatch_heads_json

        dispatch = dispatch_heads_json(ck, tools=tools)
        dispatch_path = os.path.join(out_dir, "dispatch_heads.json")
        with open(dispatch_path, "w") as f:
            json.dump(dispatch, f)
        print(f"wrote {dispatch_path}")
        stats["dispatch_heads.json"] = dispatch_path

    artifact_paths = {name: path for name, path in stats.items() if isinstance(path, str)}
    declared_graphs: dict[str, tuple[str | None, Sequence[str], str]] = {
        "model.onnx": (fp32_path, ("logits", "hidden"), "fp32"),
    }
    if fp16_path is not None:
        declared_graphs["model.fp16.onnx"] = (
            fp16_path,
            ("logits", "hidden"),
            "fp16",
        )
    if action_path is not None:
        declared_graphs["action_model.onnx"] = (action_path, ("hidden",), "fp32")
    if action_fp16_path is not None:
        declared_graphs["action_model.fp16.onnx"] = (
            action_fp16_path,
            ("hidden",),
            "fp16",
        )
    parity_gate = _web_parity_gate(model, declared_graphs, cfg)
    stats["parity_gate"] = parity_gate
    if check:
        for graph_name, result in parity_gate["results"].items():
            deltas = ", ".join(
                f"{name}={value:.2e}" for name, value in result["max_abs_diff_by_output"].items()
            )
            print(f"{graph_name} parity passed ({deltas})")

    manifest_path = _write_bundle_manifest(
        out_dir,
        config_name=cfg.name,
        model_parameters=model.num_params(),
        checkpoint_sha256=checkpoint_identity["sha256"],
        checkpoint_stage=ck.get("stage"),
        checkpoint_step=ck.get("step"),
        model_config_sha256=config_digest,
        artifacts=artifact_paths,
        parity_gate=parity_gate,
    )
    stats["bundle-manifest.json"] = manifest_path
    full_fp32 = parity_gate["results"]["model.onnx"]["max_abs_diff_by_output"]
    stats["fp32_logits_maxdiff"] = full_fp32["logits"]
    stats["fp32_hidden_maxdiff"] = full_fp32["hidden"]
    if action_path is not None:
        stats["action_hidden_maxdiff"] = parity_gate["results"]["action_model.onnx"][
            "max_abs_diff_by_output"
        ]["hidden"]
    if fp16_path is not None:
        stats["fp16_logits_drift"] = parity_gate["results"]["model.fp16.onnx"][
            "max_abs_diff_by_output"
        ]["logits"]

    for name, path in list(stats.items()):
        if isinstance(path, str) and os.path.exists(path):
            stats[f"{name}_MB"] = os.path.getsize(path) / 1e6
    return stats


# --------------------------------------------------------------------------------------------
# Matched random backbones for latency-only WebGPU experiments
# --------------------------------------------------------------------------------------------

_RANDOM_BACKBONE_WARNING = (
    "UNTRAINED RANDOM WEIGHTS — LATENCY ONLY. This is not a capability or quality artifact."
)
_TRAINED_CACHED_WARNING = (
    "TRAINED CHECKPOINT WEIGHTS — EXPORT PARITY PASSED. Model quality is not evaluated by "
    "this exporter and must be reported separately."
)
_PAIR_DIFFERENCE_FIELDS = frozenset({"name", "ffn_hidden", "layer_types"})
_CACHED_CACHE_ATOL_CEILINGS = {
    "fp32": 1e-3,
    "fp16": 1e-1,
}
_SINGLE_TRAINED_CACHED_DECODE_MANIFEST_TYPE = "single_trained_cached_decode_suite"
_EXPORTABLE_LM_STAGES = frozenset({"pretrain", "midtrain", "sft", "rl"})
_STRUCTURED_HEAD_KEYS = (
    "tool_head",
    "ptr_head",
    "route_head",
    "dense_selector",
)
_UNSUPPORTED_VALUE_HEAD_KEYS = (
    "value_head",
    "critic_head",
    "reward_head",
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _state_dict_sha256(model: nn.Module) -> str:
    """Hash parameter names, tensor contracts, and exact CPU bytes in a stable order."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        header = {
            "dtype": str(value.dtype),
            "name": name,
            "shape": list(value.shape),
        }
        encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        raw = value.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _latency_fixture(vocab_size: int, length: int, fixture_index: int) -> torch.Tensor:
    """Create deterministic pre-tokenized IDs without consuming a random-number stream."""

    ids = torch.arange(length, dtype=torch.int64) * 131 + 17 + fixture_index * 977
    return ids.remainder(vocab_size).unsqueeze(0)


def _hidden_parity(
    model: LocalAgentLM,
    graph_path: Path,
    fixtures: Sequence[torch.Tensor],
    *,
    atol: float,
) -> dict[str, Any]:
    """Hard-gate one hidden-only graph against the exact PyTorch model used for export."""

    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(graph_path), providers=["CPUExecutionProvider"])
    results = []
    max_abs_diff = 0.0
    with torch.no_grad():
        for fixture in fixtures:
            reference = model.forward_features(fixture).numpy()
            exported = session.run(["hidden"], {"input_ids": fixture.numpy()})[0]
            if exported.shape != reference.shape:
                raise RuntimeError(
                    f"{graph_path.name} parity shape mismatch: "
                    f"PyTorch={reference.shape}, ONNX={exported.shape}"
                )
            diff = float(np.abs(reference - exported).max())
            if not np.isfinite(diff):
                raise RuntimeError(f"{graph_path.name} parity produced a non-finite delta")
            max_abs_diff = max(max_abs_diff, diff)
            results.append({"sequence_length": fixture.shape[1], "max_abs_diff": diff})
    if max_abs_diff > atol:
        raise RuntimeError(
            f"{graph_path.name} failed hidden-state parity: "
            f"max_abs_diff={max_abs_diff:.6g} exceeds atol={atol:.6g}"
        )
    return {
        "atol": atol,
        "max_abs_diff": max_abs_diff,
        "passed": True,
        "per_fixture": results,
        "provider": "CPUExecutionProvider",
        "reference": "LocalAgentLM.forward_features",
    }


def _config_source_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _artifact_entry(path: Path, *, precision: str) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "file": path.name,
        "precision": precision,
        "sha256": _sha256_path(path),
    }


def export_random_hidden_backbone(
    config_path: str,
    out_dir: str,
    *,
    seed: int = 20260728,
    pair_role: str,
    fp16: bool = True,
    opset: int = 17,
    fixture_lengths: Sequence[int] = (1, 8, 31),
    fp32_atol: float = 1e-3,
    fp16_atol: float = 5e-2,
) -> dict[str, Any]:
    """Export one deterministic, untrained hidden-only ONNX backbone for latency measurement.

    The resulting graph is deliberately incapable of producing text or actions: it exposes only
    normalized backbone hidden states. A provenance file is written only after every emitted graph
    passes parity against the exact in-memory PyTorch reference.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not pair_role:
        raise ValueError("pair_role must be a non-empty label")
    if not fixture_lengths or any(length < 1 for length in fixture_lengths):
        raise ValueError("fixture_lengths must contain positive integers")

    source = Path(config_path)
    cfg = ModelConfig.from_yaml(str(source))
    cfg.assert_within_budget()
    _assert_text_exportable(cfg)
    if any(length > cfg.max_seq_len for length in fixture_lengths):
        raise ValueError("fixture length exceeds model max_seq_len")

    output = Path(out_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to mix artifacts in non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    copied_config = output / "model-config.yaml"
    shutil.copyfile(source, copied_config)

    # Keep construction reproducible without mutating the caller's global PyTorch RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = LocalAgentLM(cfg).eval()
    wrapper = _HiddenOnly(model).eval()
    state_sha256 = _state_dict_sha256(model)

    dummy_length = min(16, cfg.max_seq_len)
    dummy = _latency_fixture(cfg.vocab_size, dummy_length, 0)
    fp32_path = output / "backbone.fp32.onnx"
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(fp32_path),
        opset_version=opset,
        dynamo=False,
        input_names=["input_ids"],
        output_names=["hidden"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "hidden": {0: "batch", 1: "seq"},
        },
    )

    fp16_path = None
    if fp16:
        fp16_path = output / "backbone.fp16.onnx"
        _convert_web_fp16(str(fp32_path), str(fp16_path))

    import onnx
    import onnxruntime as ort

    graph = onnx.load(str(fp32_path))
    onnx.checker.check_model(graph)
    graph_outputs = [value.name for value in graph.graph.output]
    if graph_outputs != ["hidden"]:
        raise RuntimeError(f"hidden-only export has unexpected outputs: {graph_outputs}")
    if any("lm_head" in initializer.name for initializer in graph.graph.initializer):
        raise RuntimeError("hidden-only export unexpectedly contains an lm_head initializer")

    fixtures = [
        _latency_fixture(cfg.vocab_size, length, index)
        for index, length in enumerate(fixture_lengths)
    ]
    fixture_contract = [
        {
            "input_ids_sha256": hashlib.sha256(fixture.numpy().tobytes()).hexdigest(),
            "sequence_length": int(fixture.shape[1]),
        }
        for fixture in fixtures
    ]
    parity = {
        fp32_path.name: _hidden_parity(model, fp32_path, fixtures, atol=fp32_atol),
    }
    if fp16_path is not None:
        parity[fp16_path.name] = _hidden_parity(model, fp16_path, fixtures, atol=fp16_atol)

    config_dict = asdict(cfg)
    artifacts = {
        fp32_path.name: _artifact_entry(fp32_path, precision="fp32"),
        copied_config.name: _artifact_entry(copied_config, precision="text"),
    }
    if fp16_path is not None:
        artifacts[fp16_path.name] = _artifact_entry(
            fp16_path, precision="fp16_internal_fp32_output"
        )

    provenance = {
        "artifact_type": "random_weight_hidden_backbone_onnx",
        "artifacts": artifacts,
        "capability_artifact": False,
        "capability_metrics": None,
        "graph_contract": {
            "dynamic_axes": ["batch", "sequence"],
            "input": {"dtype": "int64", "name": "input_ids", "shape": ["batch", "sequence"]},
            "omits": [
                "language_model_logits",
                "tool_heads",
                "pointer_heads",
                "route_heads",
                "kv_cache",
            ],
            "output": {
                "dtype": "float32",
                "name": "hidden",
                "shape": ["batch", "sequence", cfg.d_model],
            },
            "tokenizer_asset_included": False,
            "tokenizer_note": "Inputs are already-tokenized IDs; no text semantics are benchmarked.",
        },
        "latency_only": True,
        "model": {
            "config": config_dict,
            "config_canonical_sha256": _canonical_sha256(config_dict),
            "config_source": _config_source_label(source),
            "config_source_sha256": _sha256_path(source),
            "full_model_parameters": model.num_params(),
            "name": cfg.name,
            "pair_role": pair_role,
        },
        "parity": {
            "fixture_contract": ("ids[i]=(131*i+17+977*fixture_index) mod vocab_size"),
            "fixtures": fixture_contract,
            "hard_gate": True,
            "results": parity,
        },
        "purpose": "local WebGPU/WASM backbone latency measurement only",
        "quality_claims": [],
        "schema_version": 1,
        "software": {
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "opset": opset,
            "torch": torch.__version__,
        },
        "trained": False,
        "training_steps": 0,
        "warning": _RANDOM_BACKBONE_WARNING,
        "weights": {
            "checkpoint": None,
            "initialization": (
                "LocalAgentLM constructor defaults: normal(0,0.02) Linear/Embedding/Conv1d; "
                "unit norm gains; zero recurrent-loop embeddings"
            ),
            "seed": seed,
            "source": "deterministic_random_initialization",
            "state_dict_sha256": state_sha256,
        },
    }
    provenance_path = output / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "config": cfg,
        "fp16_path": str(fp16_path) if fp16_path is not None else None,
        "fp32_path": str(fp32_path),
        "model_parameters": model.num_params(),
        "parity": parity,
        "provenance": provenance,
        "provenance_path": str(provenance_path),
        "state_dict_sha256": state_sha256,
    }


def _validate_latency_pair(
    hybrid: ModelConfig,
    attention: ModelConfig,
    *,
    parameter_tolerance: float,
) -> dict[str, Any]:
    if not 0 <= parameter_tolerance <= 1:
        raise ValueError("parameter_tolerance must be in [0, 1]")
    hybrid_dict = asdict(hybrid)
    attention_dict = asdict(attention)
    differing_fields = {key for key in hybrid_dict if hybrid_dict[key] != attention_dict[key]}
    if differing_fields != _PAIR_DIFFERENCE_FIELDS:
        raise ValueError(
            "matched pair must differ in exactly name, ffn_hidden, and layer_types; "
            f"observed {sorted(differing_fields)}"
        )
    if "conv" not in hybrid.block_types() or "attn" not in hybrid.block_types():
        raise ValueError("hybrid treatment must contain both conv and attention layers")
    if set(attention.block_types()) != {"attn"}:
        raise ValueError("attention control must contain attention layers only")

    hybrid_parameters = hybrid.estimate_params()
    attention_parameters = attention.estimate_params()
    relative_delta = abs(attention_parameters - hybrid_parameters) / hybrid_parameters
    if relative_delta > parameter_tolerance:
        raise ValueError(
            f"matched pair parameter delta {relative_delta:.4%} exceeds "
            f"tolerance {parameter_tolerance:.4%}"
        )
    return {
        "attention_parameters": attention_parameters,
        "hybrid_parameters": hybrid_parameters,
        "relative_parameter_delta": relative_delta,
    }


def export_matched_random_backbones(
    hybrid_config_path: str,
    attention_config_path: str,
    out_dir: str,
    *,
    seed: int = 20260728,
    fp16: bool = True,
    opset: int = 17,
    fixture_lengths: Sequence[int] = (1, 8, 31),
    parameter_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Export the matched hybrid/all-attention random pair and a pair-level manifest."""

    hybrid_cfg = ModelConfig.from_yaml(hybrid_config_path)
    attention_cfg = ModelConfig.from_yaml(attention_config_path)
    match = _validate_latency_pair(
        hybrid_cfg,
        attention_cfg,
        parameter_tolerance=parameter_tolerance,
    )

    output = Path(out_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to mix artifacts in non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    hybrid_result = export_random_hidden_backbone(
        hybrid_config_path,
        str(output / "hybrid"),
        seed=seed,
        pair_role="hybrid_treatment",
        fp16=fp16,
        opset=opset,
        fixture_lengths=fixture_lengths,
    )
    attention_result = export_random_hidden_backbone(
        attention_config_path,
        str(output / "attention"),
        seed=seed,
        pair_role="all_attention_control",
        fp16=fp16,
        opset=opset,
        fixture_lengths=fixture_lengths,
    )

    pair_artifacts = {}
    for role, result in (
        ("hybrid", hybrid_result),
        ("attention", attention_result),
    ):
        model_dir = output / role
        for artifact in sorted(model_dir.iterdir()):
            if artifact.is_file():
                relative = artifact.relative_to(output).as_posix()
                pair_artifacts[relative] = {
                    "bytes": artifact.stat().st_size,
                    "sha256": _sha256_path(artifact),
                }

    manifest = {
        "artifact_type": "matched_random_backbone_latency_suite",
        "artifacts": pair_artifacts,
        "capability_artifact": False,
        "controlled_fields": sorted(set(asdict(hybrid_cfg)) - _PAIR_DIFFERENCE_FIELDS),
        "intentional_differences": {
            key: {
                "all_attention_control": asdict(attention_cfg)[key],
                "hybrid_treatment": asdict(hybrid_cfg)[key],
            }
            for key in sorted(_PAIR_DIFFERENCE_FIELDS)
        },
        "latency_only": True,
        "match": match,
        "models": {
            "all_attention_control": {
                "directory": "attention",
                "name": attention_cfg.name,
                "provenance": "attention/provenance.json",
            },
            "hybrid_treatment": {
                "directory": "hybrid",
                "name": hybrid_cfg.name,
                "provenance": "hybrid/provenance.json",
            },
        },
        "purpose": "matched local WebGPU/WASM backbone latency comparison only",
        "quality_claims": [],
        "schema_version": 1,
        "shared_random_seed": seed,
        "trained": False,
        "warning": _RANDOM_BACKBONE_WARNING,
    }
    manifest_path = output / "matched-backbones.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "attention": attention_result,
        "hybrid": hybrid_result,
        "manifest": manifest,
        "manifest_path": str(manifest_path),
    }


# --------------------------------------------------------------------------------------------
# Matched random cache-bearing graphs for autoregressive WebGPU decode latency
# --------------------------------------------------------------------------------------------


def _cache_slot_descriptors(cfg: ModelConfig) -> list[dict[str, Any]]:
    """Describe the stable flattened cache ABI for every recurrent loop/layer pass."""

    descriptors: list[dict[str, Any]] = []
    slot = 0
    for loop in range(cfg.n_loops):
        for layer, kind in enumerate(cfg.block_types()):
            if kind == "attn":
                past_inputs = [f"past_{slot}_key", f"past_{slot}_value"]
                present_outputs = [f"present_{slot}_key", f"present_{slot}_value"]
                shape: list[Any] = [
                    "batch",
                    cfg.n_kv_heads,
                    "cache_sequence",
                    cfg.head_dim,
                ]
                update = "append_one_token_along_axis_2"
            else:
                past_inputs = [f"past_{slot}_conv"]
                present_outputs = [f"present_{slot}_conv"]
                shape = ["batch", cfg.d_model, max(0, cfg.conv_kernel - 1)]
                update = "replace_with_latest_fixed_width_tail"
            descriptors.append(
                {
                    "kind": kind,
                    "layer": layer,
                    "loop": loop,
                    "past_inputs": past_inputs,
                    "present_outputs": present_outputs,
                    "shape": shape,
                    "slot": slot,
                    "update": update,
                }
            )
            slot += 1
    return descriptors


def _flatten_caches(
    caches: Sequence[Any],
    descriptors: Sequence[dict[str, Any]],
) -> tuple[torch.Tensor, ...]:
    """Flatten model cache objects according to the explicit browser ABI."""

    if len(caches) != len(descriptors):
        raise ValueError(
            f"cache slot count mismatch: expected {len(descriptors)}, got {len(caches)}"
        )
    flat: list[torch.Tensor] = []
    for cache, descriptor in zip(caches, descriptors):
        if descriptor["kind"] == "attn":
            if not isinstance(cache, tuple) or len(cache) != 2:
                raise TypeError(f"attention cache slot {descriptor['slot']} must be a (K, V) tuple")
            flat.extend(cache)
        else:
            if not isinstance(cache, torch.Tensor):
                raise TypeError(f"conv cache slot {descriptor['slot']} must be a tensor")
            flat.append(cache)
    return tuple(flat)


def _unflatten_caches(
    flat_caches: Sequence[torch.Tensor],
    descriptors: Sequence[dict[str, Any]],
) -> list[Any]:
    """Reconstruct model cache objects from the flattened ONNX input sequence."""

    caches: list[Any] = []
    offset = 0
    for descriptor in descriptors:
        width = 2 if descriptor["kind"] == "attn" else 1
        values = flat_caches[offset : offset + width]
        if len(values) != width:
            raise ValueError("flattened cache input count does not match the cache ABI")
        caches.append((values[0], values[1]) if width == 2 else values[0])
        offset += width
    if offset != len(flat_caches):
        raise ValueError("flattened cache inputs contain trailing tensors")
    return caches


def _last_token_logits_from_features(
    model: LocalAgentLM,
    features: torch.Tensor,
) -> torch.Tensor:
    """Project only the final normalized feature and return logits ``[batch, vocab]``."""

    final = features[:, -1, :]
    hidden = model.out_proj(final) if model.out_proj is not None else final
    return (
        torch.nn.functional.linear(hidden, model.embed.weight)
        if model.lm_head is None
        else model.lm_head(hidden)
    )


def _next_token_from_features(model: LocalAgentLM, features: torch.Tensor) -> torch.Tensor:
    """Return the compatibility greedy token derived from final-token logits."""

    return torch.argmax(_last_token_logits_from_features(model, features), dim=-1)


class _CachedPrefill(nn.Module):
    """Prompt ids -> final logits, compatibility greedy token, and initialized caches."""

    def __init__(self, model: LocalAgentLM, descriptors: Sequence[dict[str, Any]]):
        super().__init__()
        self.model = model
        self.descriptors = list(descriptors)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features, caches = self.model.forward_features(
            input_ids,
            caches=[None] * self.model.n_cache_slots(),
        )
        logits = _last_token_logits_from_features(self.model, features)
        next_token = torch.argmax(logits, dim=-1)
        return (next_token, logits, *_flatten_caches(caches, self.descriptors))


class _CachedDecode(nn.Module):
    """One token plus past state -> final logits, compatibility token, and updated state."""

    def __init__(self, model: LocalAgentLM, descriptors: Sequence[dict[str, Any]]):
        super().__init__()
        self.model = model
        self.descriptors = list(descriptors)
        try:
            first_attention = next(
                descriptor for descriptor in self.descriptors if descriptor["kind"] == "attn"
            )
        except StopIteration as exc:
            raise ValueError("cached decode requires at least one attention cache slot") from exc
        self.position_source_input = first_attention["past_inputs"][0]
        self.position_source_flat_index = next(
            index
            for index, name in enumerate(
                name for descriptor in self.descriptors for name in descriptor["past_inputs"]
            )
            if name == self.position_source_input
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *flat_caches: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        caches = _unflatten_caches(flat_caches, self.descriptors)
        # This symbolic Size value is intentionally the sole RoPE position source. The caller
        # cannot provide a scalar that diverges from the actual K/V cache length.
        position = flat_caches[self.position_source_flat_index].shape[2]
        features, updated = self.model.forward_features(
            input_ids,
            pos=position,
            caches=caches,
        )
        logits = _last_token_logits_from_features(self.model, features)
        next_token = torch.argmax(logits, dim=-1)
        return (next_token, logits, *_flatten_caches(updated, self.descriptors))


def _cache_names(
    descriptors: Sequence[dict[str, Any]],
    field: str,
) -> list[str]:
    return [name for descriptor in descriptors for name in descriptor[field]]


def _cached_dynamic_axes(
    descriptors: Sequence[dict[str, Any]],
    *,
    decode: bool,
) -> dict[str, dict[int, str]]:
    axes: dict[str, dict[int, str]] = {
        "input_ids": {0: "batch"},
        "next_token": {0: "batch"},
        "logits": {0: "batch"},
    }
    if not decode:
        axes["input_ids"][1] = "prompt_sequence"
    for descriptor in descriptors:
        for name in descriptor["past_inputs"] if decode else []:
            axes[name] = {0: "batch"}
            if descriptor["kind"] == "attn":
                axes[name][2] = "past_sequence"
        for name in descriptor["present_outputs"]:
            axes[name] = {0: "batch"}
            if descriptor["kind"] == "attn":
                axes[name][2] = "present_sequence"
    return axes


def _convert_cached_fp16(fp32_path: Path, fp16_path: Path) -> None:
    """Convert internals and every floating cache boundary to genuine fp16."""

    import onnx
    import onnxruntime as ort
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model = onnx.load(str(fp32_path))
    converted = convert_float_to_float16(model, keep_io_types=False)
    raw_path = Path(f"{fp16_path}.raw")
    onnx.save(converted, str(raw_path))
    try:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        options.optimized_model_filepath = str(fp16_path)
        ort.InferenceSession(
            str(raw_path),
            options,
            providers=["CPUExecutionProvider"],
        )
    finally:
        raw_path.unlink(missing_ok=True)


def _onnx_tensor_dtype(value_info: Any) -> int:
    return int(value_info.type.tensor_type.elem_type)


def _validate_cached_graph_contract(
    path: Path,
    *,
    input_names: Sequence[str],
    output_names: Sequence[str],
    cache_dtype: str,
    decode: bool,
) -> None:
    """Fail closed on names, token width, and the exact ONNX boundary dtypes."""

    import onnx
    from onnx import TensorProto

    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    actual_inputs = [value.name for value in graph.graph.input]
    actual_outputs = [value.name for value in graph.graph.output]
    if actual_inputs != list(input_names):
        raise RuntimeError(f"{path.name} inputs differ from cache ABI: {actual_inputs}")
    if actual_outputs != list(output_names):
        raise RuntimeError(f"{path.name} outputs differ from cache ABI: {actual_outputs}")

    expected_cache_type = TensorProto.FLOAT16 if cache_dtype == "float16" else TensorProto.FLOAT
    if _onnx_tensor_dtype(graph.graph.input[0]) != TensorProto.INT64:
        raise RuntimeError(f"{path.name} input_ids must be int64")
    if _onnx_tensor_dtype(graph.graph.output[0]) != TensorProto.INT64:
        raise RuntimeError(f"{path.name} next_token must be int64")
    if _onnx_tensor_dtype(graph.graph.output[1]) != expected_cache_type:
        raise RuntimeError(f"{path.name} logits must be {cache_dtype}")
    logits_dims = graph.graph.output[1].type.tensor_type.shape.dim
    if len(logits_dims) != 2 or logits_dims[1].dim_value <= 0:
        raise RuntimeError(f"{path.name} logits must have static shape [batch, vocab]")
    cache_inputs = graph.graph.input[1:] if decode else []
    for value in [*cache_inputs, *graph.graph.output[2:]]:
        if _onnx_tensor_dtype(value) != expected_cache_type:
            raise RuntimeError(f"{path.name} cache tensor {value.name} is not {cache_dtype}")
    if decode:
        token_dims = graph.graph.input[0].type.tensor_type.shape.dim
        if len(token_dims) != 2 or token_dims[1].dim_value != 1:
            raise RuntimeError(f"{path.name} decode token axis must be statically one")


def _cached_reference_prefill(
    model: LocalAgentLM,
    input_ids: torch.Tensor,
    descriptors: Sequence[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    features, caches = model.forward_features(
        input_ids,
        caches=[None] * model.n_cache_slots(),
    )
    logits = _last_token_logits_from_features(model, features)
    return torch.argmax(logits, dim=-1), logits, _flatten_caches(caches, descriptors)


def _cached_reference_decode(
    model: LocalAgentLM,
    input_ids: torch.Tensor,
    flat_caches: Sequence[torch.Tensor],
    descriptors: Sequence[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    first_attention_input = next(
        descriptor["past_inputs"][0] for descriptor in descriptors if descriptor["kind"] == "attn"
    )
    flattened_names = _cache_names(descriptors, "past_inputs")
    source_index = flattened_names.index(first_attention_input)
    position = flat_caches[source_index].shape[2]
    features, caches = model.forward_features(
        input_ids,
        pos=position,
        caches=_unflatten_caches(flat_caches, descriptors),
    )
    logits = _last_token_logits_from_features(model, features)
    return torch.argmax(logits, dim=-1), logits, _flatten_caches(caches, descriptors)


def _full_context_last_token(
    model: LocalAgentLM,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fresh, stateless reference used to audit the model's own cached implementation."""

    features = model.forward_features(input_ids)
    logits = _last_token_logits_from_features(model, features)
    return torch.argmax(logits, dim=-1), logits


def _cached_trajectory_parity(
    model: LocalAgentLM,
    prefill_path: Path,
    decode_path: Path,
    descriptors: Sequence[dict[str, Any]],
    fixtures: Sequence[torch.Tensor],
    *,
    cache_dtype: str,
    cache_atol: float,
    decode_steps: int,
    reference: str,
) -> dict[str, Any]:
    """Compare logits and caches over complete trajectories and bind tested graph bytes."""

    import numpy as np
    import onnxruntime as ort

    if decode_steps < 3:
        raise ValueError("cached trajectory parity requires at least three decode steps")
    prefill_artifact_before = _bundle_artifact_identity(str(prefill_path))
    decode_artifact_before = _bundle_artifact_identity(str(decode_path))
    prefill_session = ort.InferenceSession(
        str(prefill_path),
        providers=["CPUExecutionProvider"],
    )
    decode_session = ort.InferenceSession(
        str(decode_path),
        providers=["CPUExecutionProvider"],
    )
    past_names = _cache_names(descriptors, "past_inputs")
    present_names = _cache_names(descriptors, "present_outputs")
    expected_prefill_outputs = ["next_token", "logits", *present_names]
    expected_decode_inputs = ["input_ids", *past_names]
    expected_decode_outputs = ["next_token", "logits", *present_names]
    if [value.name for value in prefill_session.get_outputs()] != expected_prefill_outputs:
        raise RuntimeError("prefill runtime outputs do not match the cache ABI")
    if [value.name for value in decode_session.get_inputs()] != expected_decode_inputs:
        raise RuntimeError("decode runtime inputs do not match the cache ABI")
    if [value.name for value in decode_session.get_outputs()] != expected_decode_outputs:
        raise RuntimeError("decode runtime outputs do not match the cache ABI")

    maximum_cache_delta = 0.0
    maximum_logits_delta = 0.0
    maximum_cached_vs_full_logits_delta = 0.0
    per_fixture: list[dict[str, Any]] = []
    with torch.no_grad():
        for fixture in fixtures:
            reference_token, reference_logits, reference_caches = _cached_reference_prefill(
                model,
                fixture,
                descriptors,
            )
            full_prefix = fixture
            full_context_token, full_context_logits = _full_context_last_token(
                model,
                full_prefix,
            )
            if not torch.equal(reference_token, full_context_token):
                raise RuntimeError(
                    "PyTorch cached prefill disagrees with a fresh full-context forward "
                    f"at prompt length {fixture.shape[1]}"
                )
            exported = prefill_session.run(None, {"input_ids": fixture.numpy()})
            exported_token = exported[0]
            exported_logits = exported[1]
            exported_caches = list(exported[2:])
            if not np.array_equal(exported_token, reference_token.numpy()):
                raise RuntimeError(
                    f"{prefill_path.name} greedy next-token parity failed "
                    f"at prompt length {fixture.shape[1]}"
                )
            steps: list[dict[str, Any]] = []

            def compare_caches(
                stage: str,
                expected_caches: Sequence[torch.Tensor],
                actual_caches: Sequence[Any],
            ) -> float:
                nonlocal maximum_cache_delta
                step_max = 0.0
                for name, expected_cache, actual_cache in zip(
                    present_names,
                    expected_caches,
                    actual_caches,
                ):
                    if tuple(actual_cache.shape) != tuple(expected_cache.shape):
                        raise RuntimeError(
                            f"{stage} cache {name} shape mismatch: "
                            f"PyTorch={tuple(expected_cache.shape)}, ONNX={actual_cache.shape}"
                        )
                    delta = float(
                        np.abs(expected_cache.numpy() - actual_cache.astype(np.float32)).max()
                    )
                    if not np.isfinite(delta):
                        raise RuntimeError(f"{stage} cache {name} parity was non-finite")
                    step_max = max(step_max, delta)
                maximum_cache_delta = max(maximum_cache_delta, step_max)
                if step_max > cache_atol:
                    raise RuntimeError(
                        f"{stage} cache parity delta {step_max:.6g} exceeds {cache_atol:.6g}"
                    )
                return step_max

            def compare_logits(
                stage: str,
                fixture_value: torch.Tensor,
                expected_logits: torch.Tensor,
                actual_logits: Any,
                fresh_logits: torch.Tensor,
            ) -> tuple[float, float]:
                nonlocal maximum_cached_vs_full_logits_delta, maximum_logits_delta
                expected_shape = (fixture_value.shape[0], model.cfg.vocab_size)
                if tuple(expected_logits.shape) != expected_shape:
                    raise RuntimeError(
                        f"{stage} PyTorch logits shape mismatch: "
                        f"expected {expected_shape}, got {tuple(expected_logits.shape)}"
                    )
                if tuple(actual_logits.shape) != expected_shape:
                    raise RuntimeError(
                        f"{stage} ONNX logits shape mismatch: "
                        f"expected {expected_shape}, got {actual_logits.shape}"
                    )
                exported_delta = float(
                    np.abs(expected_logits.numpy() - actual_logits.astype(np.float32)).max()
                )
                cached_full_delta = float(
                    torch.max(torch.abs(expected_logits - fresh_logits)).item()
                )
                if not np.isfinite(exported_delta) or not math.isfinite(cached_full_delta):
                    raise RuntimeError(f"{stage} logits parity was non-finite")
                maximum_logits_delta = max(maximum_logits_delta, exported_delta)
                maximum_cached_vs_full_logits_delta = max(
                    maximum_cached_vs_full_logits_delta,
                    cached_full_delta,
                )
                if exported_delta > cache_atol:
                    raise RuntimeError(
                        f"{stage} logits parity delta {exported_delta:.6g} exceeds {cache_atol:.6g}"
                    )
                if cached_full_delta > _CACHED_CACHE_ATOL_CEILINGS["fp32"]:
                    raise RuntimeError(
                        f"{stage} PyTorch cached-vs-full logits delta "
                        f"{cached_full_delta:.6g} exceeds "
                        f"{_CACHED_CACHE_ATOL_CEILINGS['fp32']:.6g}"
                    )
                return exported_delta, cached_full_delta

            prefill_delta = compare_caches(
                "prefill",
                reference_caches,
                exported_caches,
            )
            prefill_logits_delta, prefill_cached_full_logits_delta = compare_logits(
                "prefill",
                fixture,
                reference_logits,
                exported_logits,
                full_context_logits,
            )
            current_token = reference_token.unsqueeze(1)
            for step in range(decode_steps):
                full_prefix = torch.cat((full_prefix, current_token), dim=1)
                reference_token, reference_logits, reference_caches = _cached_reference_decode(
                    model,
                    current_token,
                    reference_caches,
                    descriptors,
                )
                full_context_token, full_context_logits = _full_context_last_token(
                    model,
                    full_prefix,
                )
                if not torch.equal(reference_token, full_context_token):
                    raise RuntimeError(
                        "PyTorch cached decode disagrees with a fresh full-context forward "
                        f"at prompt length {fixture.shape[1]}, decode step {step + 1}"
                    )
                feeds: dict[str, Any] = {"input_ids": current_token.numpy()}
                feeds.update(zip(past_names, exported_caches))
                exported = decode_session.run(None, feeds)
                exported_token = exported[0]
                exported_logits = exported[1]
                exported_caches = list(exported[2:])
                if not np.array_equal(exported_token, reference_token.numpy()):
                    raise RuntimeError(
                        f"{decode_path.name} greedy next-token parity failed at "
                        f"prompt length {fixture.shape[1]}, decode step {step + 1}"
                    )
                step_delta = compare_caches(
                    f"decode step {step + 1}",
                    reference_caches,
                    exported_caches,
                )
                logits_delta, cached_full_logits_delta = compare_logits(
                    f"decode step {step + 1}",
                    fixture,
                    reference_logits,
                    exported_logits,
                    full_context_logits,
                )
                steps.append(
                    {
                        "cache_max_abs_diff": step_delta,
                        "cached_vs_full_context_logits_max_abs_diff": (cached_full_logits_delta),
                        "cached_vs_full_context_next_token_exact": True,
                        "decode_step": step + 1,
                        "logits_max_abs_diff": logits_delta,
                        "next_token_exact": True,
                    }
                )
                current_token = reference_token.unsqueeze(1)
            per_fixture.append(
                {
                    "decode": steps,
                    "input_ids_sha256": hashlib.sha256(fixture.numpy().tobytes()).hexdigest(),
                    "prefill_cached_vs_full_context_next_token_exact": True,
                    "prefill_cache_max_abs_diff": prefill_delta,
                    "prefill_cached_vs_full_context_logits_max_abs_diff": (
                        prefill_cached_full_logits_delta
                    ),
                    "prefill_logits_max_abs_diff": prefill_logits_delta,
                    "prefill_next_token_exact": True,
                    "prompt_length": int(fixture.shape[1]),
                }
            )

    artifacts_after = {
        "decode": _bundle_artifact_identity(str(decode_path)),
        "prefill": _bundle_artifact_identity(str(prefill_path)),
    }
    if artifacts_after["prefill"] != prefill_artifact_before:
        raise RuntimeError("prefill graph changed during trajectory parity")
    if artifacts_after["decode"] != decode_artifact_before:
        raise RuntimeError("decode graph changed during trajectory parity")
    return {
        "artifacts": artifacts_after,
        "cache_atol": cache_atol,
        "cache_dtype": cache_dtype,
        "decode_steps": decode_steps,
        "final_token_logits_shape": ["batch", model.cfg.vocab_size],
        "greedy_next_token_exact": True,
        "hard_gate": True,
        "logits_atol": cache_atol,
        "max_cache_abs_diff": maximum_cache_delta,
        "max_cached_vs_full_context_logits_abs_diff": (maximum_cached_vs_full_logits_delta),
        "max_logits_abs_diff": maximum_logits_delta,
        "passed": True,
        "per_fixture": per_fixture,
        "provider": "CPUExecutionProvider",
        "reference": reference,
        "reference_independence": {
            "onnx_logits_vs_pytorch_cached_path": True,
            "onnx_vs_pytorch_cached_path": True,
            "pytorch_cached_vs_fresh_full_context_logits": True,
            "pytorch_cached_vs_fresh_full_context_greedy_token": True,
        },
    }


def _cached_graph_io_contract(
    descriptors: Sequence[dict[str, Any]],
    *,
    cache_dtype: str,
    decode: bool,
) -> dict[str, Any]:
    input_names = ["input_ids"]
    if decode:
        input_names.extend(_cache_names(descriptors, "past_inputs"))
    output_names = ["next_token", "logits", *_cache_names(descriptors, "present_outputs")]
    inputs = [
        {
            "dtype": "int64",
            "name": "input_ids",
            "shape": ["batch", 1 if decode else "prompt_sequence"],
        }
    ]
    if decode:
        for descriptor in descriptors:
            for name in descriptor["past_inputs"]:
                inputs.append(
                    {
                        "dtype": cache_dtype,
                        "name": name,
                        "shape": descriptor["shape"],
                    }
                )
    outputs = [
        {"dtype": "int64", "name": "next_token", "shape": ["batch"]},
        {
            "dtype": cache_dtype,
            "name": "logits",
            "shape": ["batch", "vocab_size"],
        },
    ]
    for descriptor in descriptors:
        for name in descriptor["present_outputs"]:
            outputs.append(
                {
                    "dtype": cache_dtype,
                    "name": name,
                    "shape": descriptor["shape"],
                }
            )
    return {
        "input_names": input_names,
        "inputs": inputs,
        "output_names": output_names,
        "outputs": outputs,
    }


def _validate_cached_fixture_lengths(
    fixture_lengths: Sequence[int],
) -> tuple[int, ...]:
    """Require genuinely different traces so a fixed-position decode graph cannot pass."""

    lengths = tuple(fixture_lengths)
    if any(length < 1 for length in lengths):
        raise ValueError("fixture_lengths must contain positive integers")
    if len(lengths) < 2:
        raise ValueError("cached trajectory parity requires at least two distinct fixture lengths")
    if len(set(lengths)) != len(lengths):
        raise ValueError("cached trajectory parity fixture lengths must be distinct")
    return lengths


def _validate_cached_cache_atol(
    value: float,
    *,
    precision: str,
) -> float:
    """Keep the hard parity gate within its fixed benchmark tolerance ceiling."""

    ceiling = _CACHED_CACHE_ATOL_CEILINGS[precision]
    if not math.isfinite(value) or not 0 <= value <= ceiling:
        raise ValueError(f"{precision}_cache_atol must be finite and in [0, {ceiling}]")
    return value


def _checkpoint_model_config(
    checkpoint: Mapping[str, Any],
    requested: ModelConfig,
) -> tuple[ModelConfig, dict[str, Any]]:
    """Validate the checkpoint's recorded architecture and model-config lineage."""

    raw = checkpoint.get("cfg")
    if raw is None:
        raise ValueError("cached-decode checkpoint has no model cfg")
    if isinstance(raw, Mapping):
        raw_dict = dict(raw)
    elif hasattr(raw, "__dict__"):
        raw_dict = dict(vars(raw))
    else:
        raise ValueError("cached-decode checkpoint cfg must be a mapping or config object")
    known_fields = set(ModelConfig.__dataclass_fields__)
    missing_fields = sorted(known_fields - set(raw_dict))
    unknown_fields = sorted(set(raw_dict) - known_fields)
    if missing_fields or unknown_fields:
        raise ValueError(
            "cached-decode checkpoint model cfg is not the current complete ModelConfig "
            f"contract: missing={missing_fields}, unknown={unknown_fields}"
        )
    checkpoint_cfg = ModelConfig(
        **{key: value for key, value in raw_dict.items() if key in ModelConfig.__dataclass_fields__}
    )
    checkpoint_cfg.assert_within_budget()

    requested_dict = asdict(requested)
    checkpoint_dict = asdict(checkpoint_cfg)
    architecture_fields = sorted(set(requested_dict) - {"layer_types"})
    mismatches = [
        field for field in architecture_fields if checkpoint_dict[field] != requested_dict[field]
    ]
    if checkpoint_cfg.block_types() != requested.block_types():
        mismatches.append("layer_types")
    if mismatches:
        details = ", ".join(
            (f"{field}={checkpoint_cfg.block_types()!r} -> {requested.block_types()!r}")
            if field == "layer_types"
            else (f"{field}={checkpoint_dict[field]!r} -> {requested_dict[field]!r}")
            for field in mismatches
        )
        raise ValueError(
            "cached-decode checkpoint architecture/vocabulary does not match "
            f"the selected model config: {details}"
        )

    lineage = checkpoint.get("lineage")
    if lineage is not None and not isinstance(lineage, Mapping):
        raise ValueError("cached-decode checkpoint lineage metadata must be a mapping")
    if isinstance(lineage, Mapping) and lineage.get("model_config_sha256") is not None:
        from localagent.train.stage_data import canonical_sha256

        recorded = lineage["model_config_sha256"]
        observed = canonical_sha256(raw_dict)
        if recorded != observed:
            raise ValueError(
                "cached-decode checkpoint model-config lineage hash does not match cfg"
            )
    if (
        isinstance(lineage, Mapping)
        and lineage.get("stage") is not None
        and lineage["stage"] != checkpoint.get("stage")
    ):
        raise ValueError("cached-decode checkpoint stage and lineage stage disagree")
    return checkpoint_cfg, raw_dict


def _valid_sha256(value: Any, *, label: str) -> str:
    """Return one lowercase SHA-256 digest or reject malformed provenance."""

    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _checkpoint_lineage(
    checkpoint: Mapping[str, Any],
    raw_model_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the content-bound lineage emitted by every current training stage."""

    from localagent.train.stage_data import LINEAGE_VERSION, canonical_sha256

    stage = checkpoint.get("stage")
    if stage not in _EXPORTABLE_LM_STAGES:
        supported = ", ".join(sorted(_EXPORTABLE_LM_STAGES))
        raise ValueError(
            f"cached-decode checkpoint stage must be one of {supported}; observed {stage!r}"
        )
    lineage = checkpoint.get("lineage")
    if not isinstance(lineage, Mapping):
        raise TypeError("cached-decode checkpoint has no lineage metadata")
    version = lineage.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version != LINEAGE_VERSION:
        raise ValueError(
            "cached-decode checkpoint lineage version is unsupported: "
            f"expected {LINEAGE_VERSION}, got {version!r}"
        )
    if lineage.get("stage") != stage:
        raise ValueError("cached-decode checkpoint stage and lineage stage disagree")

    required_hashes = (
        "config_sha256",
        "data_sha256",
        "model_config_sha256",
        "tokenizer_sha256",
    )
    validated_hashes = {
        key: _valid_sha256(
            lineage.get(key),
            label=f"checkpoint lineage.{key}",
        )
        for key in required_hashes
    }
    noncanonical_hashes = [key for key, value in validated_hashes.items() if lineage[key] != value]
    if noncanonical_hashes:
        raise ValueError(
            "cached-decode checkpoint lineage hashes must be lowercase: "
            + ", ".join(noncanonical_hashes)
        )
    if "git" not in lineage:
        raise ValueError("cached-decode checkpoint lineage.git metadata is missing")
    observed_model_config_sha256 = canonical_sha256(dict(raw_model_config))
    if validated_hashes["model_config_sha256"] != observed_model_config_sha256:
        raise ValueError("cached-decode checkpoint model-config lineage hash does not match cfg")
    parent_sha256 = lineage.get("parent_checkpoint_sha256")
    if stage != "pretrain" or parent_sha256 is not None:
        parent_sha256 = _valid_sha256(
            parent_sha256,
            label="checkpoint lineage.parent_checkpoint_sha256",
        )
    if parent_sha256 is not None and lineage["parent_checkpoint_sha256"] != parent_sha256:
        raise ValueError(
            "cached-decode checkpoint lineage.parent_checkpoint_sha256 must be lowercase"
        )

    normalized = dict(lineage)
    normalized.update(validated_hashes)
    if parent_sha256 is not None:
        normalized["parent_checkpoint_sha256"] = parent_sha256
    return normalized


def _strict_load_auxiliary_head(
    name: str,
    module: nn.Module,
    state: Any,
) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise TypeError(f"cached-decode checkpoint {name} state must be a mapping")
    try:
        module.load_state_dict(dict(state), strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(
            f"cached-decode checkpoint {name} is incompatible with the selected LM"
        ) from exc
    return {
        "parameters": sum(parameter.numel() for parameter in module.parameters()),
        "state_dict_keys": sorted(state),
        "validated": True,
    }


def _checkpoint_auxiliary_heads(
    checkpoint: Mapping[str, Any],
    cfg: ModelConfig,
) -> dict[str, Any]:
    """Validate usable SFT probes and reject stale RL or unsupported value heads."""

    stage = checkpoint["stage"]
    state_dict = checkpoint.get("state_dict")
    if isinstance(state_dict, Mapping):
        auxiliary_prefixes = (*_STRUCTURED_HEAD_KEYS, *_UNSUPPORTED_VALUE_HEAD_KEYS)
        embedded = sorted(
            name
            for name in state_dict
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in auxiliary_prefixes)
        )
        if embedded:
            raise ValueError(
                "cached-decode LM state_dict contains auxiliary head parameters: "
                + ", ".join(embedded)
            )

    value_heads = [
        name for name in _UNSUPPORTED_VALUE_HEAD_KEYS if checkpoint.get(name) is not None
    ]
    if value_heads:
        raise ValueError(
            "cached-decode export does not support checkpoint value/reward heads: "
            + ", ".join(value_heads)
        )

    present = [name for name in _STRUCTURED_HEAD_KEYS if checkpoint.get(name) is not None]
    availability = checkpoint.get("structured_heads_available")
    if availability is not None and not isinstance(availability, bool):
        raise ValueError("cached-decode checkpoint structured_heads_available must be boolean")

    invalidated = checkpoint.get("invalidated_structured_heads")
    if stage == "rl":
        if availability is not False:
            raise ValueError(
                "cached-decode RL checkpoint must explicitly declare "
                "structured_heads_available=false"
            )
        if not isinstance(invalidated, list):
            raise ValueError("cached-decode RL checkpoint must list invalidated_structured_heads")
        if len(set(invalidated)) != len(invalidated) or any(
            name not in _STRUCTURED_HEAD_KEYS for name in invalidated
        ):
            raise ValueError(
                "cached-decode RL invalidated_structured_heads must be a unique known-head list"
            )
        if present:
            raise ValueError(
                "cached-decode RL checkpoint carries stale structured heads invalidated by "
                "LM policy optimization: " + ", ".join(present)
            )
        return {
            "available": False,
            "exported": False,
            "invalidated": list(invalidated),
            "validated": True,
        }

    if invalidated is not None:
        raise ValueError(
            "cached-decode checkpoint records invalidated_structured_heads outside the RL stage"
        )
    if stage != "sft" and present:
        raise ValueError(
            f"cached-decode {stage} checkpoint unexpectedly carries structured heads: "
            + ", ".join(present)
        )
    if availability is False and present:
        raise ValueError(
            "cached-decode checkpoint marks structured heads unavailable but carries head state"
        )
    if availability is True and not present:
        raise ValueError(
            "cached-decode checkpoint marks structured heads available without head state"
        )
    if stage != "sft":
        return {
            "available": False,
            "exported": False,
            "invalidated": [],
            "validated": True,
        }

    summaries: dict[str, Any] = {}
    if "tool_head" in present:
        from localagent.agent.tool_head import ToolHead

        summaries["tool_head"] = _strict_load_auxiliary_head(
            "tool_head",
            ToolHead(cfg.d_model),
            checkpoint["tool_head"],
        )
    if "ptr_head" in present:
        from localagent.agent.pointer_head import PointerHead

        summaries["ptr_head"] = _strict_load_auxiliary_head(
            "ptr_head",
            PointerHead(cfg.d_model),
            checkpoint["ptr_head"],
        )
    if "route_head" in present:
        from localagent.agent.routes import RouteHead

        summaries["route_head"] = _strict_load_auxiliary_head(
            "route_head",
            RouteHead(cfg.d_model),
            checkpoint["route_head"],
        )
    if "dense_selector" in present:
        from localagent.agent.dense_selector import DenseToolSelector

        selector = checkpoint["dense_selector"]
        if not isinstance(selector, Mapping):
            raise ValueError("cached-decode checkpoint dense_selector state must be a mapping")
        q_weight = selector.get("q_proj.weight")
        t_weight = selector.get("t_proj.weight")
        if (
            not isinstance(q_weight, torch.Tensor)
            or not isinstance(t_weight, torch.Tensor)
            or q_weight.ndim != 2
            or t_weight.ndim != 2
            or q_weight.shape[1] != cfg.d_model
            or q_weight.shape[0] != t_weight.shape[0]
        ):
            raise ValueError(
                "cached-decode checkpoint dense_selector dimensions are incompatible "
                "with the selected LM"
            )
        projection = int(q_weight.shape[0])
        recorded_projection = checkpoint.get("selector_proj")
        if recorded_projection is not None and recorded_projection != projection:
            raise ValueError(
                "cached-decode checkpoint selector_proj disagrees with dense_selector state"
            )
        summary = _strict_load_auxiliary_head(
            "dense_selector",
            DenseToolSelector(
                cfg.d_model,
                emb_dim=int(t_weight.shape[1]),
                proj=projection,
            ),
            selector,
        )
        summary.update(
            {
                "embedding_dim": int(t_weight.shape[1]),
                "projection_dim": projection,
            }
        )
        summaries["dense_selector"] = summary
    return {
        "available": bool(present),
        "exported": False,
        "heads": summaries,
        "invalidated": [],
        "validated": True,
    }


def _checkpoint_tokenizer_provenance(
    checkpoint: Mapping[str, Any],
    cfg: ModelConfig,
    *,
    tokenizer_path: str | None,
) -> dict[str, Any]:
    """Fail closed on recorded tokenizer identity whenever the checkpoint carries it."""

    raw_tokenizer = checkpoint.get("tokenizer")
    if raw_tokenizer is not None and not isinstance(raw_tokenizer, Mapping):
        raise ValueError("cached-decode checkpoint tokenizer metadata must be a mapping")
    tokenizer = dict(raw_tokenizer) if isinstance(raw_tokenizer, Mapping) else {}
    lineage = checkpoint.get("lineage")
    recorded_hashes: list[str] = []
    if tokenizer.get("sha256") is not None:
        recorded_hashes.append(
            _valid_sha256(
                tokenizer["sha256"],
                label="checkpoint tokenizer.sha256",
            )
        )
    if isinstance(lineage, Mapping) and lineage.get("tokenizer_sha256") is not None:
        recorded_hashes.append(
            _valid_sha256(
                lineage["tokenizer_sha256"],
                label="checkpoint lineage.tokenizer_sha256",
            )
        )
    if len(set(recorded_hashes)) > 1:
        raise ValueError("cached-decode checkpoint tokenizer metadata and lineage hashes disagree")
    recorded_sha256 = recorded_hashes[0] if recorded_hashes else None

    recorded_vocab = tokenizer.get("vocab_size")
    if recorded_vocab is not None and recorded_vocab != cfg.vocab_size:
        raise ValueError(
            "cached-decode checkpoint tokenizer vocabulary does not match model config: "
            f"tokenizer={recorded_vocab}, model={cfg.vocab_size}"
        )
    expected_kind = "byte" if cfg.vocab_size == 256 else "bpe"
    recorded_kind = tokenizer.get("kind", expected_kind)
    if recorded_kind is not None and recorded_kind != expected_kind:
        raise ValueError(
            "cached-decode checkpoint tokenizer kind does not match model vocabulary: "
            f"tokenizer={recorded_kind!r}, expected={expected_kind!r}"
        )

    selected_path = tokenizer_path
    if selected_path is None and tokenizer.get("path") is not None:
        selected_path = str(tokenizer["path"])
    artifact_identity = None
    artifact_label = None
    if selected_path is not None:
        artifact = Path(selected_path)
        if not artifact.is_file():
            raise FileNotFoundError(
                "cached-decode tokenizer artifact is missing; pass an existing "
                f"--tokenizer path: {artifact}"
            )
        artifact_identity = _bundle_artifact_identity(str(artifact))
        artifact_label = _config_source_label(artifact)
        if recorded_sha256 is not None and artifact_identity["sha256"] != recorded_sha256:
            raise ValueError(
                "cached-decode tokenizer artifact SHA-256 does not match checkpoint provenance"
            )
        if cfg.vocab_size != 256:
            _load_web_tokenizer(cfg, str(artifact))
    elif cfg.vocab_size != 256 and (tokenizer or recorded_sha256 is not None):
        raise ValueError(
            "cached-decode checkpoint records a BPE tokenizer but no verifiable artifact path; "
            "pass --tokenizer"
        )

    if cfg.vocab_size == 256 and recorded_sha256 is not None:
        builtin_sha256 = _canonical_sha256(
            {
                "implementation": "localagent.model.tokenizer",
                "kind": "byte",
                "vocab_size": 256,
            }
        )
        if recorded_sha256 != builtin_sha256:
            raise ValueError(
                "cached-decode checkpoint byte-tokenizer SHA-256 does not match "
                "the built-in tokenizer contract"
            )

    return {
        "artifact": artifact_label,
        "artifact_identity": artifact_identity,
        "checkpoint_metadata_present": raw_tokenizer is not None,
        "kind": recorded_kind,
        "encoding": "utf-8-bytes" if cfg.vocab_size == 256 else "bytelevel-bpe",
        "eos_id": 0,
        "pad_id": 0,
        "sha256": recorded_sha256,
        "_source_path": selected_path,
        "verified": bool(
            recorded_sha256 is not None and (artifact_identity is not None or cfg.vocab_size == 256)
        ),
        "vocab_size": cfg.vocab_size,
    }


def _checkpoint_token_count(
    checkpoint: Mapping[str, Any],
    *,
    accounting_field: str,
    legacy_field: str,
) -> int | None:
    accounting = checkpoint.get("token_accounting")
    recorded = None
    if isinstance(accounting, Mapping) and accounting.get(accounting_field) is not None:
        recorded = accounting[accounting_field]
    legacy = checkpoint.get(legacy_field)
    if recorded is not None and legacy is not None and recorded != legacy:
        raise ValueError(
            f"cached-decode checkpoint {accounting_field} token accounting disagrees "
            f"with {legacy_field}"
        )
    value = recorded if recorded is not None else legacy
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"cached-decode checkpoint {accounting_field} token count must be non-negative"
        )
    return value


def _load_cached_decode_checkpoint(
    cfg: ModelConfig,
    checkpoint_path: str,
    *,
    tokenizer_path: str | None,
) -> tuple[LocalAgentLM, dict[str, Any]]:
    """Strict-load one lineage-validated training-stage LM and bind its provenance."""

    source = Path(checkpoint_path)
    checkpoint, identity_before = _load_weights_only_checkpoint(source)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("cached-decode checkpoint payload must be a mapping")

    _, raw_model_config = _checkpoint_model_config(checkpoint, cfg)
    lineage = _checkpoint_lineage(checkpoint, raw_model_config)
    stage = str(checkpoint["stage"])
    tokenizer = _checkpoint_tokenizer_provenance(
        checkpoint,
        cfg,
        tokenizer_path=tokenizer_path,
    )
    tokenizer_source_path = tokenizer.pop("_source_path")
    auxiliary_heads = _checkpoint_auxiliary_heads(checkpoint, cfg)

    step = checkpoint.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("cached-decode checkpoint step must be a non-negative integer")
    history_field = "reward_history" if stage == "rl" else "loss_history"
    history = checkpoint.get(history_field)
    if not isinstance(history, list) or len(history) != step + 1:
        raise ValueError(
            f"cached-decode checkpoint {history_field} must contain exactly step+1 entries"
        )
    loss_tokens = _checkpoint_token_count(
        checkpoint,
        accounting_field="loss_tokens",
        legacy_field="tokens_seen",
    )
    input_tokens = _checkpoint_token_count(
        checkpoint,
        accounting_field="input_tokens",
        legacy_field="input_tokens_seen",
    )
    if stage != "rl" and (loss_tokens is None or loss_tokens <= 0):
        raise ValueError("cached-decode checkpoint must record a positive supervised-token count")
    if input_tokens is not None and loss_tokens is not None and input_tokens < loss_tokens:
        raise ValueError(
            "cached-decode checkpoint input-token count cannot be smaller than "
            "its supervised-token count"
        )
    rl_accounting = None
    if stage == "rl":
        raw_rl_accounting = checkpoint.get("rl_accounting")
        if not isinstance(raw_rl_accounting, Mapping):
            raise ValueError("cached-decode RL checkpoint has no rl_accounting mapping")
        realized_updates = raw_rl_accounting.get("realized_optimizer_updates")
        if (
            isinstance(realized_updates, bool)
            or not isinstance(realized_updates, int)
            or realized_updates < 0
        ):
            raise ValueError(
                "cached-decode RL checkpoint realized_optimizer_updates must be non-negative"
            )
        rl_accounting = dict(raw_rl_accounting)

    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise TypeError("cached-decode checkpoint has no state_dict mapping")
    model = LocalAgentLM(cfg).eval()
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "cached-decode checkpoint state_dict does not strictly match selected model"
        ) from exc
    return model, {
        "bytes": identity_before["bytes"],
        "checkpoint": _config_source_label(source),
        "checkpoint_sha256": identity_before["sha256"],
        "conversation_prompt_contract": checkpoint.get("conversation_prompt_contract"),
        "auxiliary_heads": auxiliary_heads,
        "history_field": history_field,
        "input_tokens_seen": input_tokens,
        "lineage": lineage,
        "rl_accounting": rl_accounting,
        "stage": stage,
        "step": step,
        "tokenizer": tokenizer,
        "tokenizer_source_path": tokenizer_source_path,
        "tokens_seen": loss_tokens,
        "training_steps": step + 1,
    }


def _training_lineage_export(
    checkpoint_info: Mapping[str, Any],
    training_artifact_sha256: Sequence[str],
    training_artifact_identities: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the exact lineage sidecar consumed by fresh external-eval contract v2."""

    from localagent.data.prompt_contract import OPENAI_FULL_CATALOG_V1
    from localagent.eval.external_action_contract import (
        SCHEMA_VERSION,
        TRAINING_LINEAGE_KIND,
    )

    hashes = list(training_artifact_sha256)
    if len(set(hashes)) != len(hashes) or any(
        not isinstance(item, str)
        or len(item) != 64
        or any(character not in "0123456789abcdef" for character in item)
        for item in hashes
    ):
        raise ValueError("training_artifact_sha256 must contain unique lowercase SHA-256 values")
    identities: list[dict[str, Any]] | None = None
    if training_artifact_identities is not None:
        identities = []
        for index, raw_identity in enumerate(training_artifact_identities):
            if not isinstance(raw_identity, Mapping):
                raise ValueError(f"training artifact identity {index} must be a mapping")
            if set(raw_identity) != {"artifact_kind", "bytes", "path", "sha256"}:
                raise ValueError(f"training artifact identity {index} has unexpected fields")
            path = raw_identity["path"]
            artifact_kind = raw_identity["artifact_kind"]
            if not isinstance(path, str) or not path or not Path(path).is_absolute():
                raise ValueError(f"training artifact identity {index} path must be absolute")
            if not isinstance(artifact_kind, str) or not artifact_kind:
                raise ValueError(f"training artifact identity {index} kind must be non-empty")
            expected = {
                "bytes": raw_identity["bytes"],
                "sha256": raw_identity["sha256"],
            }
            observed = _bundle_artifact_identity(path)
            if expected != observed:
                raise RuntimeError(f"training artifact changed before export: {path}")
            identities.append(
                {
                    "artifact_kind": artifact_kind,
                    "bytes": observed["bytes"],
                    "path": path,
                    "sha256": observed["sha256"],
                }
            )
        if [identity["sha256"] for identity in identities] != hashes:
            raise ValueError(
                "training artifact identities do not match training_artifact_sha256 order"
            )
        if len({identity["path"] for identity in identities}) != len(identities):
            raise ValueError("training artifact identity paths must be unique")

    lineage = dict(checkpoint_info["lineage"])
    required_lineage_keys = {
        "version",
        "stage",
        "config_sha256",
        "model_config_sha256",
        "data_sha256",
        "tokenizer_sha256",
        "git",
    }
    optional_lineage_keys = {"parent_checkpoint_sha256"}
    missing = sorted(required_lineage_keys - set(lineage))
    extra = sorted(set(lineage) - required_lineage_keys - optional_lineage_keys)
    if missing or extra:
        raise ValueError(
            "checkpoint lineage cannot populate localagent_training_lineage_export: "
            f"missing={missing}, extra={extra}"
        )
    git = lineage["git"]
    git_keys = {"commit", "repository_sha256", "dirty", "worktree_sha256"}
    if not isinstance(git, Mapping) or set(git) != git_keys:
        raise ValueError(
            "checkpoint lineage.git cannot populate localagent_training_lineage_export"
        )
    commit = git["commit"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("checkpoint lineage.git.commit must be a lowercase 40-hex commit")
    for key in ("repository_sha256", "worktree_sha256"):
        normalized = _valid_sha256(
            git[key],
            label=f"checkpoint lineage.git.{key}",
        )
        if normalized != git[key]:
            raise ValueError(f"checkpoint lineage.git.{key} must be lowercase")
    if not isinstance(git["dirty"], bool):
        raise TypeError("checkpoint lineage.git.dirty must be boolean")

    stage = str(checkpoint_info["stage"])
    prompt_contract = checkpoint_info["conversation_prompt_contract"]
    if stage == "pretrain":
        if "parent_checkpoint_sha256" in lineage:
            raise ValueError(
                "pretrain training lineage export must not declare a parent checkpoint"
            )
        if prompt_contract is not None:
            raise ValueError(
                "pretrain training lineage export requires conversation_prompt_contract=null"
            )
    elif prompt_contract != OPENAI_FULL_CATALOG_V1:
        raise ValueError(
            "posttraining lineage export requires conversation_prompt_contract="
            f"{OPENAI_FULL_CATALOG_V1!r}"
        )
    exported = {
        "kind": TRAINING_LINEAGE_KIND,
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "checkpoint_sha256": checkpoint_info["checkpoint_sha256"],
        "lineage": lineage,
        "training_artifact_sha256": hashes,
        "conversation_prompt_contract": prompt_contract,
    }
    if identities is not None:
        exported["training_artifacts"] = identities
    return exported


def _cached_runtime_metadata(
    cfg: ModelConfig,
    *,
    graph_contract: Mapping[str, Any],
    checkpoint_info: Mapping[str, Any] | None,
    tokenizer_path: Path | None,
    config_path: Path,
    default_precision: str,
    model_parameters: int,
    training_lineage_file: str | None,
) -> dict[str, Any]:
    """Build the self-contained browser/runtime contract for a cached bundle."""

    from localagent.model.tokenizer import BPETokenizer, ByteTokenizer
    from localagent.train.stage_data import tokenizer_identity

    tokenizer = None
    tokenizer_file = None
    if cfg.vocab_size == 256:
        tokenizer = ByteTokenizer()
    elif tokenizer_path is not None:
        tokenizer = BPETokenizer.from_file(tokenizer_path)
        tokenizer_file = tokenizer_path.name

    if tokenizer is not None:
        metadata = _meta_json(
            cfg,
            tokenizer,
            tokenizer_file=tokenizer_file,
            model_parameters=None,
        )
    else:
        metadata = {
            "d_model": cfg.d_model,
            "encoding": None,
            "eos_id": None,
            "max_seq_len": cfg.max_seq_len,
            "pad_id": None,
            "tokenizer_required_for_text": True,
            "vocab_size": cfg.vocab_size,
        }

    if checkpoint_info is None:
        tokenizer_metadata = (
            {
                **tokenizer_identity("byte", vocab_size=256),
                "encoding": "utf-8-bytes",
                "eos_id": 0,
                "file": None,
                "pad_id": 0,
                "verified": True,
            }
            if cfg.vocab_size == 256
            else {
                "encoding": None,
                "eos_id": None,
                "file": None,
                "kind": None,
                "pad_id": None,
                "sha256": None,
                "verified": False,
                "vocab_size": cfg.vocab_size,
            }
        )
        checkpoint_metadata = None
    else:
        tokenizer_metadata = {
            **dict(checkpoint_info["tokenizer"]),
            "file": tokenizer_file,
        }
        checkpoint_metadata = {
            "conversation_prompt_contract": checkpoint_info["conversation_prompt_contract"],
            "lineage": dict(checkpoint_info["lineage"]),
            "lineage_export": (
                {
                    "file": training_lineage_file,
                    "kind": "localagent_training_lineage_export",
                    "schema_version": 1,
                }
                if training_lineage_file is not None
                else None
            ),
            "sha256": checkpoint_info["checkpoint_sha256"],
            "stage": checkpoint_info["stage"],
            "step": checkpoint_info["step"],
        }

    metadata.update(
        {
            "artifact_type": "localagent_cached_autoregressive_onnx",
            "checkpoint": checkpoint_metadata,
            "default_precision": default_precision,
            "graph_contract": dict(graph_contract),
            "model": {
                "config": asdict(cfg),
                "config_canonical_sha256": _canonical_sha256(asdict(cfg)),
                "config_file": config_path.name,
                "parameters": model_parameters,
            },
            "schema_version": 1,
            "tokenizer": tokenizer_metadata,
        }
    )
    return metadata


def _cached_artifact_identity(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    """Return the browser pin fields from one cached-export artifact identity."""

    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must pin cached-export artifact bytes and SHA-256")
    size = value.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise RuntimeError(f"{label} must pin a positive byte size")
    try:
        sha256 = _valid_sha256(value.get("sha256"), label=f"{label} SHA-256")
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    return {"bytes": size, "sha256": sha256}


def _single_trained_cached_decode_manifest(
    provenance: Mapping[str, Any],
    *,
    provenance_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the single-model browser wrapper only from hard-passed trained provenance."""

    if (
        provenance.get("schema_version") != 1
        or provenance.get("artifact_type") != "trained_checkpoint_cached_decode_onnx"
        or provenance.get("trained") is not True
        or provenance.get("latency_only") is not False
        or provenance.get("capability_artifact") is not False
    ):
        raise RuntimeError(
            "refusing to publish a single-model wrapper for non-trained cached provenance"
        )

    model = provenance.get("model")
    if not isinstance(model, Mapping):
        raise RuntimeError("trained cached provenance has no model identity")
    model_name = model.get("name")
    pair_role = model.get("pair_role")
    if not isinstance(model_name, str) or not model_name:
        raise RuntimeError("trained cached provenance model name must be non-empty")
    if not isinstance(pair_role, str) or not pair_role:
        raise RuntimeError("trained cached provenance model role must be non-empty")

    graph_contract = provenance.get("graph_contract")
    graphs = graph_contract.get("graphs") if isinstance(graph_contract, Mapping) else None
    parity = provenance.get("parity")
    results = parity.get("results") if isinstance(parity, Mapping) else None
    if (
        not isinstance(parity, Mapping)
        or parity.get("hard_gate") is not True
        or not isinstance(graphs, Mapping)
        or set(graphs) not in ({"fp32"}, {"fp32", "fp16"})
        or not isinstance(results, Mapping)
        or set(results) != set(graphs)
    ):
        raise RuntimeError(
            "refusing to publish a single-model wrapper without complete hard trajectory parity"
        )

    artifacts = provenance.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("trained cached provenance has no artifact identities")
    for precision, precision_graphs in graphs.items():
        result = results.get(precision)
        if (
            not isinstance(result, Mapping)
            or result.get("hard_gate") is not True
            or result.get("passed") is not True
            or result.get("greedy_next_token_exact") is not True
            or result.get("reference") != "exact in-memory LocalAgentLM checkpoint weights"
        ):
            raise RuntimeError(
                f"refusing to publish a single-model wrapper: {precision} trajectory parity failed"
            )
        parity_artifacts = result.get("artifacts")
        if not isinstance(precision_graphs, Mapping) or not isinstance(
            parity_artifacts,
            Mapping,
        ):
            raise RuntimeError(f"{precision} trajectory parity has no graph identity bindings")
        for graph_kind in ("prefill", "decode"):
            graph = precision_graphs.get(graph_kind)
            graph_file = graph.get("file") if isinstance(graph, Mapping) else None
            expected_file = f"{graph_kind}.{precision}.onnx"
            if graph_file != expected_file:
                raise RuntimeError(
                    f"{precision} {graph_kind} trajectory parity names a non-canonical graph"
                )
            published_identity = _cached_artifact_identity(
                artifacts.get(graph_file),
                label=f"provenance artifact {graph_file}",
            )
            tested_identity = _cached_artifact_identity(
                parity_artifacts.get(graph_kind),
                label=f"{precision} parity artifact {graph_kind}",
            )
            if tested_identity != published_identity:
                raise RuntimeError(
                    f"{graph_file} provenance identity is not bound to trajectory parity"
                )

    identity = _cached_artifact_identity(
        provenance_identity,
        label="provenance.json",
    )
    return {
        "artifact_type": _SINGLE_TRAINED_CACHED_DECODE_MANIFEST_TYPE,
        "artifacts": {
            "provenance.json": identity,
        },
        "capability_artifact": False,
        "latency_only": False,
        "model": {
            "name": model_name,
            "pair_role": pair_role,
            "provenance": "provenance.json",
        },
        "quality_claims": [],
        "quality_evaluation": {
            "included": False,
            "required_separately": True,
        },
        "schema_version": 1,
        "trained": True,
    }


def _write_single_trained_cached_decode_manifest(
    output: Path,
    provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    """Atomically publish a provenance-pinned wrapper for the browser's single-model mode."""

    provenance_path = output / "provenance.json"
    expected_text = json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    expected_payload = expected_text.replace("\n", os.linesep).encode("utf-8")
    expected_identity = {
        "bytes": len(expected_payload),
        "sha256": hashlib.sha256(expected_payload).hexdigest(),
    }
    observed_identity = _bundle_artifact_identity(str(provenance_path))
    if observed_identity != expected_identity:
        raise RuntimeError("provenance.json changed before single-model wrapper publication")
    manifest = _single_trained_cached_decode_manifest(
        provenance,
        provenance_identity=observed_identity,
    )

    manifest_path = output / "single-decode.json"
    temporary_path = output / "single-decode.json.tmp"
    try:
        temporary_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if _bundle_artifact_identity(str(provenance_path)) != observed_identity:
            raise RuntimeError("provenance.json changed before single-model wrapper publication")
        os.replace(temporary_path, manifest_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return manifest, manifest_path


def export_cached_decode(
    config_path: str,
    out_dir: str,
    *,
    seed: int = 20260728,
    pair_role: str,
    checkpoint_path: str | None = None,
    tokenizer_path: str | None = None,
    fp16: bool = True,
    opset: int = 17,
    fixture_lengths: Sequence[int] = (1, 8, 31),
    decode_steps: int = 4,
    fp32_cache_atol: float = 1e-3,
    fp16_cache_atol: float = 1e-1,
    training_artifact_sha256: Sequence[str] | None = None,
    training_artifact_identities: Sequence[Mapping[str, Any]] | None = None,
    require_posttraining_training_artifacts: bool = False,
) -> dict[str, Any]:
    """Export parity-gated cached graphs from random or lineage-validated checkpoint weights.

    Both graphs return ``next_token`` for the existing latency harness and final-token ``logits``
    shaped ``[batch, vocab]`` for production sampling. Supplying ``training_artifact_sha256``
    additionally writes the exact ``localagent_training_lineage_export`` sidecar required by
    fresh external-eval contract v2.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not pair_role:
        raise ValueError("pair_role must be a non-empty label")
    if checkpoint_path is None and training_artifact_sha256 is not None:
        raise ValueError("training_artifact_sha256 is valid only with checkpoint-backed export")
    if checkpoint_path is None and training_artifact_identities is not None:
        raise ValueError("training_artifact_identities is valid only with checkpoint-backed export")
    if training_artifact_identities is not None and training_artifact_sha256 is None:
        raise ValueError("training_artifact_identities requires training_artifact_sha256")
    if decode_steps < 3:
        raise ValueError("decode_steps must be at least three")
    fixture_lengths = _validate_cached_fixture_lengths(fixture_lengths)
    fp32_cache_atol = _validate_cached_cache_atol(
        fp32_cache_atol,
        precision="fp32",
    )
    fp16_cache_atol = _validate_cached_cache_atol(
        fp16_cache_atol,
        precision="fp16",
    )

    source = Path(config_path)
    cfg = ModelConfig.from_yaml(str(source))
    cfg.assert_within_budget()
    _assert_text_exportable(cfg)
    if any(length + decode_steps > cfg.max_seq_len for length in fixture_lengths):
        raise ValueError("fixture trajectory exceeds model max_seq_len")
    descriptors = _cache_slot_descriptors(cfg)

    output = Path(out_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to mix artifacts in non-empty directory: {output}")
    checkpoint_info = None
    if checkpoint_path is None:
        if tokenizer_path is not None:
            raise ValueError("--tokenizer is valid only with checkpoint-backed export")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = LocalAgentLM(cfg).eval()
    else:
        model, checkpoint_info = _load_cached_decode_checkpoint(
            cfg,
            checkpoint_path,
            tokenizer_path=tokenizer_path,
        )
    if (
        require_posttraining_training_artifacts
        and checkpoint_info is not None
        and checkpoint_info["stage"] in {"midtrain", "sft", "rl"}
        and not training_artifact_identities
    ):
        raise ValueError(
            "accepted posttraining cached-decode export requires at least one "
            "training artifact file identity"
        )
    training_lineage = (
        _training_lineage_export(
            checkpoint_info,
            training_artifact_sha256,
            training_artifact_identities,
        )
        if checkpoint_info is not None and training_artifact_sha256 is not None
        else None
    )

    output.mkdir(parents=True, exist_ok=True)
    copied_config = output / "model-config.yaml"
    shutil.copyfile(source, copied_config)
    copied_tokenizer = None
    if checkpoint_info is not None and checkpoint_info["tokenizer_source_path"] is not None:
        tokenizer_source = Path(checkpoint_info["tokenizer_source_path"])
        expected_tokenizer_identity = checkpoint_info["tokenizer"]["artifact_identity"]
        observed_tokenizer_identity = _bundle_artifact_identity(str(tokenizer_source))
        if observed_tokenizer_identity != expected_tokenizer_identity:
            raise RuntimeError("cached-decode tokenizer changed before bundle copy")
        copied_tokenizer = output / "tokenizer.json"
        shutil.copyfile(tokenizer_source, copied_tokenizer)
        copied_tokenizer_identity = _bundle_artifact_identity(str(copied_tokenizer))
        if copied_tokenizer_identity != expected_tokenizer_identity:
            raise RuntimeError("cached-decode bundled tokenizer differs from validated source")
        checkpoint_info["tokenizer"]["file"] = copied_tokenizer.name
        checkpoint_info["tokenizer"]["bundled_artifact_identity"] = copied_tokenizer_identity

    state_sha256 = _state_dict_sha256(model)
    prefill_wrapper = _CachedPrefill(model, descriptors).eval()
    decode_wrapper = _CachedDecode(model, descriptors).eval()

    dummy_length = min(8, cfg.max_seq_len - decode_steps)
    dummy = _latency_fixture(cfg.vocab_size, dummy_length, 0)
    with torch.no_grad():
        _, _, dummy_caches = _cached_reference_prefill(model, dummy, descriptors)

    past_names = _cache_names(descriptors, "past_inputs")
    present_names = _cache_names(descriptors, "present_outputs")
    prefill_inputs = ["input_ids"]
    decode_inputs = ["input_ids", *past_names]
    outputs = ["next_token", "logits", *present_names]
    fp32_prefill = output / "prefill.fp32.onnx"
    fp32_decode = output / "decode.fp32.onnx"
    torch.onnx.export(
        prefill_wrapper,
        (dummy,),
        str(fp32_prefill),
        opset_version=opset,
        dynamo=False,
        input_names=prefill_inputs,
        output_names=outputs,
        dynamic_axes=_cached_dynamic_axes(descriptors, decode=False),
    )
    decode_token = torch.zeros((1, 1), dtype=torch.int64)
    torch.onnx.export(
        decode_wrapper,
        (decode_token, *dummy_caches),
        str(fp32_decode),
        opset_version=opset,
        dynamo=False,
        input_names=decode_inputs,
        output_names=outputs,
        dynamic_axes=_cached_dynamic_axes(descriptors, decode=True),
    )

    fp16_prefill = None
    fp16_decode = None
    if fp16:
        fp16_prefill = output / "prefill.fp16.onnx"
        fp16_decode = output / "decode.fp16.onnx"
        _convert_cached_fp16(fp32_prefill, fp16_prefill)
        _convert_cached_fp16(fp32_decode, fp16_decode)

    graphs: dict[str, tuple[Path, Path, str, float]] = {
        "fp32": (fp32_prefill, fp32_decode, "float32", fp32_cache_atol),
    }
    if fp16_prefill is not None and fp16_decode is not None:
        graphs["fp16"] = (
            fp16_prefill,
            fp16_decode,
            "float16",
            fp16_cache_atol,
        )
    fixtures = [
        _latency_fixture(cfg.vocab_size, length, index)
        for index, length in enumerate(fixture_lengths)
    ]
    parity: dict[str, Any] = {}
    for precision, (prefill_path, decode_path, cache_dtype, atol) in graphs.items():
        _validate_cached_graph_contract(
            prefill_path,
            input_names=prefill_inputs,
            output_names=outputs,
            cache_dtype=cache_dtype,
            decode=False,
        )
        _validate_cached_graph_contract(
            decode_path,
            input_names=decode_inputs,
            output_names=outputs,
            cache_dtype=cache_dtype,
            decode=True,
        )
        parity[precision] = _cached_trajectory_parity(
            model,
            prefill_path,
            decode_path,
            descriptors,
            fixtures,
            cache_dtype=cache_dtype,
            cache_atol=atol,
            decode_steps=decode_steps,
            reference=(
                "exact in-memory LocalAgentLM checkpoint weights"
                if checkpoint_info is not None
                else "exact in-memory LocalAgentLM random initialization"
            ),
        )

    config_dict = asdict(cfg)
    artifact_paths: dict[str, tuple[Path, str]] = {
        copied_config.name: (copied_config, "text"),
        fp32_prefill.name: (fp32_prefill, "fp32"),
        fp32_decode.name: (fp32_decode, "fp32"),
    }
    if copied_tokenizer is not None:
        artifact_paths[copied_tokenizer.name] = (copied_tokenizer, "tokenizer")
    if fp16_prefill is not None and fp16_decode is not None:
        artifact_paths[fp16_prefill.name] = (fp16_prefill, "fp16")
        artifact_paths[fp16_decode.name] = (fp16_decode, "fp16")
    artifacts = {
        name: _artifact_entry(path, precision=precision)
        for name, (path, precision) in sorted(artifact_paths.items())
    }

    for precision, result in parity.items():
        prefill_name = f"prefill.{precision}.onnx"
        decode_name = f"decode.{precision}.onnx"
        if result["artifacts"]["prefill"] != {
            "bytes": artifacts[prefill_name]["bytes"],
            "sha256": artifacts[prefill_name]["sha256"],
        }:
            raise RuntimeError(f"{prefill_name} changed after trajectory parity")
        if result["artifacts"]["decode"] != {
            "bytes": artifacts[decode_name]["bytes"],
            "sha256": artifacts[decode_name]["sha256"],
        }:
            raise RuntimeError(f"{decode_name} changed after trajectory parity")

    contract_slots = []
    for descriptor in descriptors:
        contract_slots.append(
            {
                **descriptor,
                "dtype_by_precision": {
                    precision: cache_dtype for precision, (_, _, cache_dtype, _) in graphs.items()
                },
            }
        )
    graph_contract = {
        "cache_slots": contract_slots,
        "cache_update_strategy": (
            "attention K/V append one token; short-conv state replaces its fixed-width tail"
        ),
        "decode_position": {
            "caller_position_input": False,
            "derived_from": decode_wrapper.position_source_input,
            "rule": "RoPE position = first attention past-key axis-2 length",
        },
        "decode_token_axis_fixed_one": True,
        "graphs": {
            precision: {
                "cache_dtype": cache_dtype,
                "decode": {
                    "file": decode_path.name,
                    **_cached_graph_io_contract(
                        descriptors,
                        cache_dtype=cache_dtype,
                        decode=True,
                    ),
                },
                "prefill": {
                    "file": prefill_path.name,
                    **_cached_graph_io_contract(
                        descriptors,
                        cache_dtype=cache_dtype,
                        decode=False,
                    ),
                },
            }
            for precision, (
                prefill_path,
                decode_path,
                cache_dtype,
                _,
            ) in graphs.items()
        },
        "logits": {
            "description": "unnormalized LM scores for the final input token only",
            "dtype_by_precision": {
                precision: cache_dtype for precision, (_, _, cache_dtype, _) in graphs.items()
            },
            "name": "logits",
            "shape": ["batch", cfg.vocab_size],
        },
        "next_token": {
            "decode": "compatibility argmax over the exported final-token logits",
            "dtype": "int64",
            "name": "next_token",
            "shape": ["batch"],
        },
        "prefill_projection": (
            "only the final normalized prompt feature is projected to vocabulary logits"
        ),
    }
    training_lineage_path = None
    if training_lineage is not None:
        training_lineage_path = output / "training-lineage.json"
        training_lineage_path.write_text(
            json.dumps(training_lineage, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts[training_lineage_path.name] = _artifact_entry(
            training_lineage_path,
            precision="metadata",
        )
    runtime_metadata = _cached_runtime_metadata(
        cfg,
        graph_contract=graph_contract,
        checkpoint_info=checkpoint_info,
        tokenizer_path=copied_tokenizer,
        config_path=copied_config,
        default_precision="fp16" if "fp16" in graphs else "fp32",
        model_parameters=model.num_params(),
        training_lineage_file=(
            training_lineage_path.name if training_lineage_path is not None else None
        ),
    )
    metadata_path = output / "meta.json"
    metadata_path.write_text(
        json.dumps(runtime_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts[metadata_path.name] = _artifact_entry(
        metadata_path,
        precision="metadata",
    )
    trained = checkpoint_info is not None
    quality_evaluation = (
        {
            "included": False,
            "required_separately": True,
            "scope": (
                "Export validates graph parity only; held-out CE/BPB and downstream "
                "capability metrics are separate artifacts."
            ),
        }
        if trained
        else None
    )
    if checkpoint_info is None:
        weights = {
            "checkpoint": None,
            "initialization": (
                "LocalAgentLM constructor defaults: normal(0,0.02) Linear/Embedding/Conv1d; "
                "unit norm gains; zero recurrent-loop embeddings"
            ),
            "seed": seed,
            "source": "deterministic_random_initialization",
            "state_dict_sha256": state_sha256,
        }
    else:
        weights = {
            "checkpoint": checkpoint_info["checkpoint"],
            "checkpoint_bytes": checkpoint_info["bytes"],
            "checkpoint_sha256": checkpoint_info["checkpoint_sha256"],
            "checkpoint_stage": checkpoint_info["stage"],
            "checkpoint_step": checkpoint_info["step"],
            "input_tokens_seen": checkpoint_info["input_tokens_seen"],
            "source": "strict_lineage_validated_lm_checkpoint",
            "state_dict_sha256": state_sha256,
            "tokens_seen": checkpoint_info["tokens_seen"],
        }

    provenance = {
        "artifact_type": (
            "trained_checkpoint_cached_decode_onnx"
            if trained
            else "random_weight_cached_decode_onnx"
        ),
        "artifacts": artifacts,
        # Trained LM weights do not by themselves constitute downstream capability evidence.
        "capability_artifact": False,
        "capability_metrics": None,
        "graph_contract": graph_contract,
        "latency_only": not trained,
        "model": {
            "config": config_dict,
            "config_canonical_sha256": _canonical_sha256(config_dict),
            "config_source": _config_source_label(source),
            "config_source_sha256": _sha256_path(source),
            "full_model_parameters": model.num_params(),
            "name": cfg.name,
            "pair_role": pair_role,
        },
        "parity": {
            "cache_atol_ceiling_by_precision": dict(_CACHED_CACHE_ATOL_CEILINGS),
            "fixture_contract": "ids[i]=(131*i+17+977*fixture_index) mod vocab_size",
            "fixture_length_requirement": (
                "at least two positive, distinct prompt lengths; order is preserved"
            ),
            "fixture_lengths": list(fixture_lengths),
            "hard_gate": True,
            "results": parity,
        },
        "purpose": (
            "checkpoint-backed local WebGPU/WASM autoregressive prefill and cached decode; "
            "model quality is evaluated separately"
            if trained
            else "local WebGPU/WASM autoregressive prefill and cached-decode latency only"
        ),
        "quality_claims": [],
        "schema_version": 1,
        "software": {
            "onnx": __import__("onnx").__version__,
            "onnxruntime": __import__("onnxruntime").__version__,
            "opset": opset,
            "torch": torch.__version__,
        },
        "trained": trained,
        "training_steps": (checkpoint_info["training_steps"] if checkpoint_info is not None else 0),
        "warning": _TRAINED_CACHED_WARNING if trained else _RANDOM_BACKBONE_WARNING,
        "weights": weights,
    }
    if checkpoint_info is not None:
        provenance["auxiliary_heads"] = checkpoint_info["auxiliary_heads"]
        provenance["checkpoint_lineage"] = checkpoint_info["lineage"]
        provenance["checkpoint_step"] = checkpoint_info["step"]
        provenance["input_tokens_seen"] = checkpoint_info["input_tokens_seen"]
        provenance["quality_evaluation"] = quality_evaluation
        provenance["rl_accounting"] = checkpoint_info["rl_accounting"]
        provenance["tokenizer"] = checkpoint_info["tokenizer"]
        provenance["tokens_seen"] = checkpoint_info["tokens_seen"]
        provenance["training_lineage_export"] = (
            training_lineage_path.name if training_lineage_path is not None else None
        )
    provenance_path = output / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {
        "config": cfg,
        "graph_paths": {
            precision: {
                "decode": str(decode_path),
                "prefill": str(prefill_path),
            }
            for precision, (prefill_path, decode_path, _, _) in graphs.items()
        },
        "model_parameters": model.num_params(),
        "metadata": runtime_metadata,
        "metadata_path": str(metadata_path),
        "parity": parity,
        "provenance": provenance,
        "provenance_path": str(provenance_path),
        "state_dict_sha256": state_sha256,
        "training_lineage_path": (
            str(training_lineage_path) if training_lineage_path is not None else None
        ),
    }
    if trained:
        single_manifest, single_manifest_path = _write_single_trained_cached_decode_manifest(
            output,
            provenance,
        )
        result["single_manifest"] = single_manifest
        result["single_manifest_path"] = str(single_manifest_path)
    return result


def export_random_cached_decode(
    config_path: str,
    out_dir: str,
    *,
    seed: int = 20260728,
    pair_role: str,
    fp16: bool = True,
    opset: int = 17,
    fixture_lengths: Sequence[int] = (1, 8, 31),
    decode_steps: int = 4,
    fp32_cache_atol: float = 1e-3,
    fp16_cache_atol: float = 1e-1,
) -> dict[str, Any]:
    """Preserve the deterministic random-weight cached-decode export workflow."""

    return export_cached_decode(
        config_path,
        out_dir,
        seed=seed,
        pair_role=pair_role,
        fp16=fp16,
        opset=opset,
        fixture_lengths=fixture_lengths,
        decode_steps=decode_steps,
        fp32_cache_atol=fp32_cache_atol,
        fp16_cache_atol=fp16_cache_atol,
    )


def _matched_checkpoint_preflight(
    hybrid_cfg: ModelConfig,
    attention_cfg: ModelConfig,
    *,
    hybrid_checkpoint_path: str | None,
    attention_checkpoint_path: str | None,
    tokenizer_path: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Validate both trained arms before any pair directory or graph is written."""

    supplied = (
        hybrid_checkpoint_path is not None,
        attention_checkpoint_path is not None,
    )
    if supplied[0] != supplied[1]:
        raise ValueError(
            "matched checkpoint export requires both --hybrid-checkpoint and --attention-checkpoint"
        )
    if not any(supplied):
        if tokenizer_path is not None:
            raise ValueError("--tokenizer requires a matched checkpoint pair")
        return None, None

    _, hybrid = _load_cached_decode_checkpoint(
        hybrid_cfg,
        str(hybrid_checkpoint_path),
        tokenizer_path=tokenizer_path,
    )
    _, attention = _load_cached_decode_checkpoint(
        attention_cfg,
        str(attention_checkpoint_path),
        tokenizer_path=tokenizer_path,
    )
    controlled_fields = (
        "stage",
        "step",
        "tokens_seen",
        "input_tokens_seen",
    )
    mismatches = [field for field in controlled_fields if hybrid[field] != attention[field]]
    hybrid_tokenizer = hybrid["tokenizer"].get("sha256")
    attention_tokenizer = attention["tokenizer"].get("sha256")
    if hybrid_tokenizer != attention_tokenizer:
        mismatches.append("tokenizer_sha256")
    if mismatches:
        raise ValueError(
            "matched checkpoint training provenance differs between arms: " + ", ".join(mismatches)
        )
    return hybrid, attention


def export_matched_cached_decode(
    hybrid_config_path: str,
    attention_config_path: str,
    out_dir: str,
    *,
    seed: int = 20260728,
    hybrid_checkpoint_path: str | None = None,
    attention_checkpoint_path: str | None = None,
    tokenizer_path: str | None = None,
    fp16: bool = True,
    opset: int = 17,
    fixture_lengths: Sequence[int] = (1, 8, 31),
    decode_steps: int = 4,
    parameter_tolerance: float = 0.01,
    training_artifact_sha256: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Export a matched random or lineage-validated checkpoint cache-bearing pair."""

    fixture_lengths = _validate_cached_fixture_lengths(fixture_lengths)
    hybrid_cfg = ModelConfig.from_yaml(hybrid_config_path)
    attention_cfg = ModelConfig.from_yaml(attention_config_path)
    match = _validate_latency_pair(
        hybrid_cfg,
        attention_cfg,
        parameter_tolerance=parameter_tolerance,
    )
    hybrid_checkpoint, attention_checkpoint = _matched_checkpoint_preflight(
        hybrid_cfg,
        attention_cfg,
        hybrid_checkpoint_path=hybrid_checkpoint_path,
        attention_checkpoint_path=attention_checkpoint_path,
        tokenizer_path=tokenizer_path,
    )
    trained = hybrid_checkpoint is not None
    if not trained and training_artifact_sha256 is not None:
        raise ValueError("training_artifact_sha256 requires a matched checkpoint pair")
    output = Path(out_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to mix artifacts in non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    hybrid_result = export_cached_decode(
        hybrid_config_path,
        str(output / "hybrid"),
        seed=seed,
        pair_role="hybrid_treatment",
        checkpoint_path=hybrid_checkpoint_path,
        tokenizer_path=tokenizer_path,
        fp16=fp16,
        opset=opset,
        fixture_lengths=fixture_lengths,
        decode_steps=decode_steps,
        training_artifact_sha256=training_artifact_sha256,
    )
    attention_result = export_cached_decode(
        attention_config_path,
        str(output / "attention"),
        seed=seed,
        pair_role="all_attention_control",
        checkpoint_path=attention_checkpoint_path,
        tokenizer_path=tokenizer_path,
        fp16=fp16,
        opset=opset,
        fixture_lengths=fixture_lengths,
        decode_steps=decode_steps,
        training_artifact_sha256=training_artifact_sha256,
    )
    if trained:
        for role, result, preflight in (
            ("hybrid", hybrid_result, hybrid_checkpoint),
            ("attention", attention_result, attention_checkpoint),
        ):
            if preflight is None:
                raise RuntimeError(f"{role} checkpoint preflight evidence is missing")
            exported_sha256 = result["provenance"]["weights"]["checkpoint_sha256"]
            if exported_sha256 != preflight["checkpoint_sha256"]:
                raise RuntimeError(
                    f"{role} checkpoint changed between pair preflight and graph export"
                )
    pair_artifacts: dict[str, dict[str, Any]] = {}
    for role in ("hybrid", "attention"):
        for artifact in sorted((output / role).iterdir()):
            if artifact.is_file():
                relative = artifact.relative_to(output).as_posix()
                pair_artifacts[relative] = {
                    "bytes": artifact.stat().st_size,
                    "sha256": _sha256_path(artifact),
                }
    manifest = {
        "artifact_type": (
            "matched_trained_cached_decode_suite"
            if trained
            else "matched_random_cached_decode_latency_suite"
        ),
        "artifacts": pair_artifacts,
        # Trained weights alone are not an agent-capability artifact. The paired LM-quality
        # scorecard is bound and reported separately from this deployment/parity bundle.
        "capability_artifact": False,
        "controlled_fields": sorted(set(asdict(hybrid_cfg)) - _PAIR_DIFFERENCE_FIELDS),
        "intentional_differences": {
            key: {
                "all_attention_control": asdict(attention_cfg)[key],
                "hybrid_treatment": asdict(hybrid_cfg)[key],
            }
            for key in sorted(_PAIR_DIFFERENCE_FIELDS)
        },
        "latency_only": not trained,
        "match": match,
        "models": {
            "all_attention_control": {
                "directory": "attention",
                "name": attention_cfg.name,
                "provenance": "attention/provenance.json",
            },
            "hybrid_treatment": {
                "directory": "hybrid",
                "name": hybrid_cfg.name,
                "provenance": "hybrid/provenance.json",
            },
        },
        "purpose": (
            "matched checkpoint-backed local WebGPU/WASM autoregressive cached decode; "
            "quality evidence is reported separately"
            if trained
            else "matched local WebGPU/WASM autoregressive cached-decode latency only"
        ),
        "quality_claims": [],
        "schema_version": 1,
        "trained": trained,
        "warning": _TRAINED_CACHED_WARNING if trained else _RANDOM_BACKBONE_WARNING,
    }
    if trained:
        manifest["checkpoints"] = {
            "all_attention_control": {
                "bytes": attention_checkpoint["bytes"],
                "checkpoint": attention_checkpoint["checkpoint"],
                "sha256": attention_checkpoint["checkpoint_sha256"],
                "stage": attention_checkpoint["stage"],
                "step": attention_checkpoint["step"],
                "tokens_seen": attention_checkpoint["tokens_seen"],
                "training_steps": attention_checkpoint["training_steps"],
            },
            "hybrid_treatment": {
                "bytes": hybrid_checkpoint["bytes"],
                "checkpoint": hybrid_checkpoint["checkpoint"],
                "sha256": hybrid_checkpoint["checkpoint_sha256"],
                "stage": hybrid_checkpoint["stage"],
                "step": hybrid_checkpoint["step"],
                "tokens_seen": hybrid_checkpoint["tokens_seen"],
                "training_steps": hybrid_checkpoint["training_steps"],
            },
        }
        manifest["quality_evaluation"] = {
            "included": False,
            "required_separately": True,
        }
        manifest["tokenizer"] = hybrid_checkpoint["tokenizer"]
    else:
        manifest["shared_random_seed"] = seed
    manifest_path = output / "matched-decode.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "attention": attention_result,
        "hybrid": hybrid_result,
        "manifest": manifest,
        "manifest_path": str(manifest_path),
    }


def export_matched_random_cached_decode(
    hybrid_config_path: str,
    attention_config_path: str,
    out_dir: str,
    *,
    seed: int = 20260728,
    fp16: bool = True,
    opset: int = 17,
    fixture_lengths: Sequence[int] = (1, 8, 31),
    decode_steps: int = 4,
    parameter_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Preserve the matched deterministic random-weight cached-decode workflow."""

    return export_matched_cached_decode(
        hybrid_config_path,
        attention_config_path,
        out_dir,
        seed=seed,
        fp16=fp16,
        opset=opset,
        fixture_lengths=fixture_lengths,
        decode_steps=decode_steps,
        parameter_tolerance=parameter_tolerance,
    )
