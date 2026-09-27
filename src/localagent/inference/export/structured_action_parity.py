"""Full-stack parity gate for the browser structured-action contract.

The ordinary ONNX export gate checks a few synthetic hidden-state fixtures.  This module exercises
the complete deployed action path on the frozen browser action suite:

* append one-token spaces *after* the complete natural prompt to materialize 512 graph tokens;
* consume ``hidden[natural_input_tokens - 1]`` for route and dense-tool dispatch;
* bound pointer scores and copied spans to ``[0, natural_input_tokens)``;
* pass copied spans through the exact JavaScript ``groundFromSchema`` implementation; and
* normalize the result with the exact browser benchmark helper.

The PyTorch reference uses the checkpoint's unrounded heads.  ONNX fp32/fp16 use the serialized
JSON heads exactly as the browser does, so the comparison also covers head serialization.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from localagent.agent.dense_selector import BoundSelector, DenseToolSelector
from localagent.agent.pointer_head import PointerHead
from localagent.agent.routes import ROUTES, RouteHead, route_of
from localagent.agent.toolset import STANDARD_TOOLS
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import ASSISTANT, USER, load_tokenizer


STRUCTURED_ACTION_PARITY_THRESHOLDS = {
    "head_serialization_max_abs_diff": 1e-4,
    "onnx_fp32_vs_native_fp32": {
        "hidden_max_abs_diff": 1e-3,
        "route_logits_max_abs_diff": 2e-3,
        "selector_scores_max_abs_diff": 2e-3,
        "pointer_scores_max_abs_diff": 2e-2,
        "confidence_max_abs_diff": 2e-3,
    },
    "onnx_fp16_vs_native_fp32": {
        "hidden_max_abs_diff": 5e-2,
        "route_logits_max_abs_diff": 5e-2,
        "selector_scores_max_abs_diff": 2e-2,
        # Pointer scores compose a learned arg embedding and a 384x384 projection before the
        # hidden-state dot product.  Exact pointer-span agreement remains a separate hard gate.
        "pointer_scores_max_abs_diff": 1.0,
        "confidence_max_abs_diff": 2e-2,
    },
}

_RUNTIME_ORDER = ("native_pytorch_fp32", "onnx_fp32", "onnx_fp16")
_PAIR_NAMES = {
    "onnx_fp32": "onnx_fp32_vs_native_fp32",
    "onnx_fp16": "onnx_fp16_vs_native_fp32",
}


def canonical_sha256(value: Any) -> str:
    """Hash canonical compact JSON, rejecting non-finite values."""

    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_identity(path: str | Path) -> dict[str, Any]:
    """Return the immutable byte identity of one required artifact."""

    artifact = Path(path)
    before = artifact.stat()
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = artifact.stat()
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in stable):
        raise RuntimeError(f"artifact changed while hashing: {artifact}")
    return {
        "path": str(artifact),
        "bytes": after.st_size,
        "sha256": digest.hexdigest(),
    }


def array_identity(value: np.ndarray) -> dict[str, Any]:
    """Hash one numerical array with explicit little-endian storage."""

    array = np.ascontiguousarray(value)
    dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(dtype, copy=False))
    return {
        "dtype": canonical.dtype.str,
        "shape": list(canonical.shape),
        "sha256": hashlib.sha256(canonical.tobytes()).hexdigest(),
    }


def materialize_trailing_compute(
    tokenizer: Any,
    query: str,
    *,
    target_input_tokens: int,
    max_seq_len: int,
) -> dict[str, Any]:
    """Materialize the corrected browser fixed-compute input for one natural query."""

    if target_input_tokens < 1 or target_input_tokens > max_seq_len:
        raise ValueError("target_input_tokens must be within the model context window")
    natural_ids = tokenizer.encode(f"{USER}{query}{ASSISTANT}")
    if len(natural_ids) > target_input_tokens:
        raise ValueError(
            f"natural prompt has {len(natural_ids)} tokens, above target {target_input_tokens}"
        )
    assistant_ids = tokenizer.encode(ASSISTANT)
    if not assistant_ids or natural_ids[-len(assistant_ids) :] != assistant_ids:
        raise ValueError("framed prompt does not end with the assistant marker")
    whitespace_ids = tokenizer.encode(" ")
    if len(whitespace_ids) != 1:
        raise ValueError("tokenizer must encode one neutral space as exactly one token")
    ids = [
        *natural_ids,
        *([whitespace_ids[0]] * (target_input_tokens - len(natural_ids))),
    ]
    input_array = np.asarray(ids, dtype="<i8")
    return {
        "ids": ids,
        "input_ids_sha256": hashlib.sha256(input_array.tobytes()).hexdigest(),
        "natural_input_tokens": len(natural_ids),
        "input_tokens": len(ids),
        "padding_tokens": len(ids) - len(natural_ids),
        "decision_feature_index": len(natural_ids) - 1,
        "pointer_domain": [0, len(natural_ids)],
    }


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _maximum_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    value = float(delta.max(initial=0.0))
    if not math.isfinite(value):
        raise RuntimeError("non-finite numerical parity delta")
    return value


def _js_linear(weight: np.ndarray, bias: np.ndarray, feature: np.ndarray) -> np.ndarray:
    """Mirror ``app.js::linrow`` output storage (double accumulation, Float32Array output)."""

    value = weight.astype(np.float64) @ feature.astype(np.float64)
    value += bias.astype(np.float64)
    return value.astype(np.float32)


def _js_normalize(value: np.ndarray) -> np.ndarray:
    """Mirror in-place Float32Array query normalization in ``dispatchSelect``."""

    squared_norm = math.fsum(float(item) * float(item) for item in value)
    norm = math.sqrt(squared_norm) or 1.0
    return np.asarray([float(item) / norm for item in value], dtype=np.float32)


def _js_dense_scores(
    feature: np.ndarray,
    dispatch_heads: dict[str, Any],
) -> np.ndarray:
    selector = dispatch_heads["dense_selector"]
    query = _js_linear(
        np.asarray(selector["q_proj_weight"]),
        np.asarray(selector["q_proj_bias"]),
        feature,
    )
    if selector.get("normalize_query"):
        query = _js_normalize(query)
    matrix = np.asarray(selector["tool_matrix"], dtype=np.float64)
    return matrix @ query.astype(np.float64)


def _softmax_confidence(logits: np.ndarray, index: int) -> float:
    values = logits.astype(np.float64)
    maximum = float(values.max())
    denominator = math.fsum(math.exp(float(item) - maximum) for item in values)
    return math.exp(float(values[index]) - maximum) / denominator


def _pointer_scores_native(
    pointer_head: PointerHead,
    hidden: np.ndarray,
    *,
    arg: str,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = torch.from_numpy(np.ascontiguousarray(hidden[:limit]))
    arg_index = torch.tensor(pointer_head.arg_idx[arg])
    with torch.no_grad():
        query = pointer_head.arg_emb(arg_index)
        start = features @ pointer_head.start(query)
        end = features @ pointer_head.end(query)
    return start.numpy(), end.numpy()


def _pointer_scores_exported(
    pointer_heads: dict[str, Any],
    hidden: np.ndarray,
    *,
    arg: str,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    arg_index = int(pointer_heads["arg_idx"][arg])
    query = np.asarray(pointer_heads["arg_emb"][arg_index], dtype=np.float64)
    start_query = np.asarray(pointer_heads["start_W"], dtype=np.float64) @ query
    end_query = np.asarray(pointer_heads["end_W"], dtype=np.float64) @ query
    features = hidden[:limit].astype(np.float64)
    return features @ start_query, features @ end_query


def _decode_pointer(
    tokenizer: Any,
    ids: Sequence[int],
    start_scores: np.ndarray,
    end_scores: np.ndarray,
) -> dict[str, Any]:
    start = int(np.argmax(start_scores))
    end = start + int(np.argmax(end_scores[start:]))
    return {
        "start": start,
        "end": end,
        "text": tokenizer.decode(list(ids[start : end + 1]), stop_at_eos=False),
        "start_score": float(start_scores[start]),
        "end_score": float(end_scores[end]),
        "start_scores": array_identity(start_scores),
        "end_scores": array_identity(end_scores),
    }


def _browser_ground_actions(
    *,
    app_js_path: str | Path,
    benchmark_js_path: str | Path,
    rows: Sequence[dict[str, Any]],
    node_executable: str = "node",
) -> tuple[list[dict[str, Any]], str]:
    """Ground and normalize actions with the exact exported JavaScript functions."""

    source = """
global.window = { __localAgentSkipInit: true, location: { search: "" } };
const fs = require("fs");
const app = require(process.argv[1]);
const benchmark = require(process.argv[2]);
const rows = JSON.parse(fs.readFileSync(0, "utf8"));
const output = rows.map((row) => {
  if (row.is_stop) {
    const action = { abstain: true };
    return {
      key: row.key,
      grounded_args: null,
      schema_valid: true,
      normalized_action: benchmark.normalizeBenchmarkAction(action),
    };
  }
  const args = app.groundFromSchema(row.prompt, row.schema, row.pointer_values);
  const action = { tool: row.tool, args };
  return {
    key: row.key,
    grounded_args: args,
    schema_valid: app.groundedArgsValid(args, row.schema),
    normalized_action: benchmark.normalizeBenchmarkAction(action),
  };
});
process.stdout.write(JSON.stringify(output));
"""
    completed = subprocess.run(
        [
            node_executable,
            "-e",
            source,
            str(Path(app_js_path).resolve()),
            str(Path(benchmark_js_path).resolve()),
        ],
        input=json.dumps(list(rows), allow_nan=False, separators=(",", ":")),
        capture_output=True,
        check=True,
        text=True,
    )
    version = subprocess.run(
        [node_executable, "--version"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    result = json.loads(completed.stdout)
    if not isinstance(result, list) or len(result) != len(rows):
        raise RuntimeError("browser grounding bridge returned an invalid row contract")
    return result, version


def _run_native_hidden(
    model: LocalAgentLM,
    inputs: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    batches = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            tensor = torch.from_numpy(inputs[start : start + batch_size])
            batches.append(model.forward_features(tensor).cpu().numpy())
    return np.concatenate(batches, axis=0)


def _run_onnx_hidden(
    graph_path: str | Path,
    inputs: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, list[str]]:
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(graph_path),
        providers=["CPUExecutionProvider"],
    )
    if [output.name for output in session.get_outputs()] != ["hidden"]:
        raise ValueError(f"{graph_path} must expose exactly one hidden output")
    batches = []
    for start in range(0, len(inputs), batch_size):
        batches.append(
            session.run(
                ["hidden"],
                {"input_ids": inputs[start : start + batch_size]},
            )[0]
        )
    providers = session.get_providers()
    return np.concatenate(batches, axis=0), providers


def _head_serialization_audit(
    *,
    checkpoint: dict[str, Any],
    pointer_heads: dict[str, Any],
    dispatch_heads: dict[str, Any],
    bound_selector: BoundSelector,
) -> dict[str, Any]:
    selector_model = bound_selector.model
    with torch.no_grad():
        reference_tool_matrix = torch.nn.functional.normalize(
            selector_model.t_proj(bound_selector.embs),
            dim=-1,
        ).cpu().numpy()
    deltas = {
        "route_weight": _maximum_abs_diff(
            checkpoint["route_head"]["fc.weight"].cpu().numpy(),
            np.asarray(dispatch_heads["route_head"]["weight"]),
        ),
        "route_bias": _maximum_abs_diff(
            checkpoint["route_head"]["fc.bias"].cpu().numpy(),
            np.asarray(dispatch_heads["route_head"]["bias"]),
        ),
        "selector_q_weight": _maximum_abs_diff(
            checkpoint["dense_selector"]["q_proj.weight"].cpu().numpy(),
            np.asarray(dispatch_heads["dense_selector"]["q_proj_weight"]),
        ),
        "selector_q_bias": _maximum_abs_diff(
            checkpoint["dense_selector"]["q_proj.bias"].cpu().numpy(),
            np.asarray(dispatch_heads["dense_selector"]["q_proj_bias"]),
        ),
        "selector_tool_matrix": _maximum_abs_diff(
            reference_tool_matrix,
            np.asarray(dispatch_heads["dense_selector"]["tool_matrix"]),
        ),
        "pointer_arg_embedding": _maximum_abs_diff(
            checkpoint["ptr_head"]["arg_emb.weight"].cpu().numpy(),
            np.asarray(pointer_heads["arg_emb"]),
        ),
        "pointer_start_weight": _maximum_abs_diff(
            checkpoint["ptr_head"]["start.weight"].cpu().numpy(),
            np.asarray(pointer_heads["start_W"]),
        ),
        "pointer_end_weight": _maximum_abs_diff(
            checkpoint["ptr_head"]["end.weight"].cpu().numpy(),
            np.asarray(pointer_heads["end_W"]),
        ),
    }
    maximum = max(deltas.values())
    threshold = STRUCTURED_ACTION_PARITY_THRESHOLDS[
        "head_serialization_max_abs_diff"
    ]
    return {
        "max_abs_diff_by_component": deltas,
        "maximum_abs_diff": maximum,
        "threshold_max_abs_diff": threshold,
        "passed": maximum <= threshold,
    }


def _runtime_policy(
    *,
    runtime: str,
    hidden: np.ndarray,
    natural_input_tokens: int,
    ids: Sequence[int],
    tokenizer: Any,
    route_head: torch.nn.Module,
    bound_selector: BoundSelector,
    pointer_head: PointerHead,
    pointer_heads_json: dict[str, Any],
    dispatch_heads: dict[str, Any],
    tool_schemas: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    feature = hidden[natural_input_tokens - 1]
    if runtime == "native_pytorch_fp32":
        feature_tensor = torch.from_numpy(np.ascontiguousarray(feature))
        with torch.no_grad():
            route_logits = route_head(feature_tensor).numpy()
            selector_scores = bound_selector.model(
                feature_tensor.unsqueeze(0),
                bound_selector.embs,
            )[0].numpy()
    else:
        route = dispatch_heads["route_head"]
        route_logits = _js_linear(
            np.asarray(route["weight"]),
            np.asarray(route["bias"]),
            feature,
        )
        selector_scores = _js_dense_scores(feature, dispatch_heads)

    route_index = int(np.argmax(route_logits))
    route_name = ROUTES[route_index]
    is_stop = route_index == int(dispatch_heads["route_head"]["stop_index"])
    dense_index = int(np.argmax(selector_scores))
    dense_tool = dispatch_heads["dense_selector"]["tool_names"][dense_index]
    selected_tool = None if is_stop else dense_tool
    route_confidence = _softmax_confidence(route_logits, route_index)
    selector_confidence = (float(selector_scores[dense_index]) + 1.0) / 2.0
    confidence = route_confidence if is_stop else selector_confidence

    pointer: dict[str, Any] = {}
    pointer_values: dict[str, str] = {}
    private_pointer_scores: dict[str, dict[str, np.ndarray]] = {}
    if selected_tool is not None:
        schema = tool_schemas[selected_tool]
        for arg in schema.get("properties", {}):
            if arg not in pointer_heads_json["arg_idx"]:
                continue
            if runtime == "native_pytorch_fp32":
                start_scores, end_scores = _pointer_scores_native(
                    pointer_head,
                    hidden,
                    arg=arg,
                    limit=natural_input_tokens,
                )
            else:
                start_scores, end_scores = _pointer_scores_exported(
                    pointer_heads_json,
                    hidden,
                    arg=arg,
                    limit=natural_input_tokens,
                )
            decoded = _decode_pointer(
                tokenizer,
                ids[:natural_input_tokens],
                start_scores,
                end_scores,
            )
            pointer[arg] = decoded
            pointer_values[arg] = decoded["text"]
            private_pointer_scores[arg] = {
                "start_scores": start_scores,
                "end_scores": end_scores,
            }

    return {
        "hidden": array_identity(hidden),
        "decision_hidden": array_identity(feature),
        "route": {
            "logits": [float(item) for item in route_logits],
            "logits_identity": array_identity(route_logits),
            "decision": route_name,
            "is_stop": is_stop,
            "decision_confidence": route_confidence,
        },
        "selector": {
            "scores_identity": array_identity(selector_scores),
            "dense_top1": dense_tool,
            "selected_tool": selected_tool,
            "selected_score": float(selector_scores[dense_index]),
            "selected_confidence": selector_confidence,
        },
        "pointer": pointer,
        "pointer_values": pointer_values,
        "action_confidence": confidence,
        "_route_logits": route_logits,
        "_selector_scores": selector_scores,
        "_pointer_scores": private_pointer_scores,
    }


def _strip_private_runtime(runtime: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in runtime.items() if not key.startswith("_")}


def _validate_manifest_artifacts(
    *,
    manifest: dict[str, Any],
    checkpoint_identity: dict[str, Any],
    artifact_identities: dict[str, dict[str, Any]],
) -> None:
    if manifest.get("checkpoint_sha256") != checkpoint_identity["sha256"]:
        raise ValueError("bundle manifest checkpoint identity does not match")
    for name, identity in artifact_identities.items():
        declared = manifest.get("artifacts", {}).get(name)
        observed = {"bytes": identity["bytes"], "sha256": identity["sha256"]}
        if declared is None:
            raise ValueError(f"bundle manifest does not declare {name}")
        if {"bytes": declared.get("bytes"), "sha256": declared.get("sha256")} != observed:
            raise ValueError(f"bundle manifest identity mismatch for {name}")


def build_structured_action_parity(
    *,
    checkpoint_path: str | Path,
    bundle_dir: str | Path,
    action_suite_path: str | Path,
    app_js_path: str | Path,
    benchmark_js_path: str | Path,
    target_input_tokens: int = 512,
    batch_size: int = 4,
    node_executable: str = "node",
    tools=None,
) -> dict[str, Any]:
    """Build and hard-gate one complete structured-action export parity report."""

    import onnx
    import onnxruntime as ort

    bundle = Path(bundle_dir)
    paths = {
        "action_model.onnx": bundle / "action_model.onnx",
        "action_model.fp16.onnx": bundle / "action_model.fp16.onnx",
        "tokenizer.json": bundle / "tokenizer.json",
        "heads.json": bundle / "heads.json",
        "dispatch_heads.json": bundle / "dispatch_heads.json",
        "meta.json": bundle / "meta.json",
    }
    checkpoint_identity = file_identity(checkpoint_path)
    bundle_manifest_identity = file_identity(bundle / "bundle-manifest.json")
    artifact_identities = {name: file_identity(path) for name, path in paths.items()}
    source_suite_identity = file_identity(action_suite_path)
    bundle_suite_identity = file_identity(bundle / "benchmark-cases.json")
    app_source_identity = file_identity(app_js_path)
    app_bundle_identity = file_identity(bundle / "app.js")
    benchmark_source_identity = file_identity(benchmark_js_path)
    benchmark_bundle_identity = file_identity(bundle / "benchmark.js")

    if source_suite_identity["sha256"] != bundle_suite_identity["sha256"]:
        raise ValueError("source and bundled action suites differ")
    if app_source_identity["sha256"] != app_bundle_identity["sha256"]:
        raise ValueError("source and bundled app.js differ")
    if benchmark_source_identity["sha256"] != benchmark_bundle_identity["sha256"]:
        raise ValueError("source and bundled benchmark.js differ")

    manifest = _load_json(bundle / "bundle-manifest.json")
    _validate_manifest_artifacts(
        manifest=manifest,
        checkpoint_identity=checkpoint_identity,
        artifact_identities=artifact_identities,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg_data = checkpoint["cfg"]
    cfg = ModelConfig(
        **{
            key: value
            for key, value in cfg_data.items()
            if key in ModelConfig.__dataclass_fields__
        }
    )
    cfg.assert_within_budget()
    if target_input_tokens > cfg.max_seq_len:
        raise ValueError("target_input_tokens exceeds checkpoint context length")

    tokenizer = load_tokenizer("bpe", paths["tokenizer.json"])
    if tokenizer.vocab_size != cfg.vocab_size:
        raise ValueError("bundle tokenizer vocabulary does not match checkpoint")
    tokenizer_metadata = checkpoint.get("tokenizer") or {}
    recorded_tokenizer_sha = tokenizer_metadata.get("sha256")
    if recorded_tokenizer_sha and recorded_tokenizer_sha != artifact_identities[
        "tokenizer.json"
    ]["sha256"]:
        raise ValueError("checkpoint tokenizer identity does not match bundle tokenizer")

    suite = _load_json(action_suite_path)
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("action suite must contain a non-empty cases list")
    case_ids = [case.get("id") for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("action suite case ids must be unique")

    meta = _load_json(paths["meta.json"])
    pointer_document = _load_json(paths["heads.json"])
    pointer_heads = pointer_document["pointer_head"]
    dispatch_heads = _load_json(paths["dispatch_heads.json"])
    tool_names = [tool["name"] for tool in meta["tools"]]
    dispatch_names = dispatch_heads["dense_selector"]["tool_names"]
    tool_specs = STANDARD_TOOLS if tools is None else list(tools)
    expected_names = [tool.name for tool in tool_specs]
    if tool_names != dispatch_names or tool_names != expected_names:
        raise ValueError("meta, dispatch, and native tool orders differ")
    if dispatch_heads["route_head"]["routes"] != list(ROUTES):
        raise ValueError("exported route order differs from native ROUTES")
    pointer_args = list(pointer_heads.get("args", []))
    pointer_arg_idx = {name: index for index, name in enumerate(pointer_args)}
    if pointer_heads["arg_idx"] != pointer_arg_idx:
        raise ValueError("exported pointer argument order is not self-consistent")
    tool_schemas = {tool["name"]: tool["schema"] for tool in meta["tools"]}

    model = LocalAgentLM(cfg).eval()
    model.load_state_dict(checkpoint["state_dict"])
    route_head = RouteHead(cfg.d_model).eval()
    route_head.load_state_dict(checkpoint["route_head"])
    selector_state = checkpoint["dense_selector"]
    selector_model = DenseToolSelector(
        cfg.d_model,
        emb_dim=int(selector_state["t_proj.weight"].shape[1]),
        proj=int(selector_state["q_proj.weight"].shape[0]),
    ).eval()
    selector_model.load_state_dict(selector_state)
    bound_selector = BoundSelector(
        selector_model,
        tool_specs,
        examples=checkpoint.get("examples"),
    )
    pointer_head = PointerHead(cfg.d_model, args=pointer_args).eval()
    pointer_head.load_state_dict(checkpoint["ptr_head"])
    serialization_audit = _head_serialization_audit(
        checkpoint=checkpoint,
        pointer_heads=pointer_heads,
        dispatch_heads=dispatch_heads,
        bound_selector=bound_selector,
    )

    materializations = [
        materialize_trailing_compute(
            tokenizer,
            case["query"],
            target_input_tokens=target_input_tokens,
            max_seq_len=cfg.max_seq_len,
        )
        for case in cases
    ]
    inputs = np.asarray([row["ids"] for row in materializations], dtype=np.int64)
    native_hidden = _run_native_hidden(model, inputs, batch_size=batch_size)
    onnx_fp32_hidden, fp32_providers = _run_onnx_hidden(
        paths["action_model.onnx"],
        inputs,
        batch_size=batch_size,
    )
    onnx_fp16_hidden, fp16_providers = _run_onnx_hidden(
        paths["action_model.fp16.onnx"],
        inputs,
        batch_size=batch_size,
    )
    hidden_by_runtime = {
        "native_pytorch_fp32": native_hidden,
        "onnx_fp32": onnx_fp32_hidden,
        "onnx_fp16": onnx_fp16_hidden,
    }

    internal_rows: list[dict[str, Any]] = []
    grounding_requests: list[dict[str, Any]] = []
    for case_index, (case, materialization) in enumerate(
        zip(cases, materializations, strict=True)
    ):
        runtime_results = {}
        for runtime in _RUNTIME_ORDER:
            result = _runtime_policy(
                runtime=runtime,
                hidden=hidden_by_runtime[runtime][case_index],
                natural_input_tokens=materialization["natural_input_tokens"],
                ids=materialization["ids"],
                tokenizer=tokenizer,
                route_head=route_head,
                bound_selector=bound_selector,
                pointer_head=pointer_head,
                pointer_heads_json=pointer_heads,
                dispatch_heads=dispatch_heads,
                tool_schemas=tool_schemas,
            )
            runtime_results[runtime] = result
            selected_tool = result["selector"]["selected_tool"]
            grounding_requests.append(
                {
                    "key": f"{case['id']}::{runtime}",
                    "is_stop": result["route"]["is_stop"],
                    "prompt": case["query"],
                    "tool": selected_tool,
                    "schema": (
                        {}
                        if selected_tool is None
                        else tool_schemas[selected_tool]
                    ),
                    "pointer_values": result["pointer_values"],
                }
            )
        internal_rows.append(
            {
                "case": case,
                "materialization": materialization,
                "runtime_results": runtime_results,
            }
        )

    grounded, node_version = _browser_ground_actions(
        app_js_path=app_js_path,
        benchmark_js_path=benchmark_js_path,
        rows=grounding_requests,
        node_executable=node_executable,
    )
    grounded_by_key = {row["key"]: row for row in grounded}
    if len(grounded_by_key) != len(grounding_requests):
        raise RuntimeError("browser grounding bridge returned duplicate or missing keys")

    aggregate_pairs = {
        pair: {
            "hidden_full_max_abs_diff": 0.0,
            "hidden_natural_span_max_abs_diff": 0.0,
            "decision_hidden_max_abs_diff": 0.0,
            "route_logits_max_abs_diff": 0.0,
            "selector_scores_max_abs_diff": 0.0,
            "pointer_start_scores_max_abs_diff": 0.0,
            "pointer_end_scores_max_abs_diff": 0.0,
            "route_confidence_max_abs_diff": 0.0,
            "selector_confidence_max_abs_diff": 0.0,
            "action_confidence_max_abs_diff": 0.0,
            "confidence_max_abs_diff": 0.0,
            "route_exact": 0,
            "selected_tool_exact": 0,
            "pointer_span_exact": 0,
            "pointer_span_comparisons": 0,
            "grounded_args_exact": 0,
            "final_normalized_action_exact": 0,
        }
        for pair in _PAIR_NAMES.values()
    }
    runtime_diagnostics = {
        runtime: {
            "route_correct": 0,
            "selected_tool_correct": 0,
            "schema_valid": 0,
            "exact_action": 0,
            "tool_required_exact_action": 0,
        }
        for runtime in _RUNTIME_ORDER
    }
    report_rows = []
    for case_index, internal in enumerate(internal_rows):
        case = internal["case"]
        materialization = internal["materialization"]
        runtime_results = internal["runtime_results"]
        expected = case["expected"]
        expected_action = (
            {"abstain": True}
            if expected.get("abstain") is True
            else {"tool": expected["tool"], "args": expected["args"]}
        )
        gold_route = "text" if expected.get("abstain") else route_of(expected["tool"])
        for runtime, result in runtime_results.items():
            browser = grounded_by_key[f"{case['id']}::{runtime}"]
            result["grounded_args"] = browser["grounded_args"]
            result["schema_valid"] = browser["schema_valid"]
            result["normalized_action"] = browser["normalized_action"]
            result["exact_expected_action"] = browser["normalized_action"] == expected_action
            result["route_correct"] = result["route"]["decision"] == gold_route
            result["selected_tool_correct"] = (
                expected.get("abstain") is not True
                and result["selector"]["selected_tool"] == expected.get("tool")
            )
            diagnostic = runtime_diagnostics[runtime]
            diagnostic["route_correct"] += int(result["route_correct"])
            diagnostic["selected_tool_correct"] += int(result["selected_tool_correct"])
            diagnostic["schema_valid"] += int(result["schema_valid"])
            diagnostic["exact_action"] += int(result["exact_expected_action"])
            if expected.get("abstain") is not True:
                diagnostic["tool_required_exact_action"] += int(
                    result["exact_expected_action"]
                )

        comparisons = {}
        reference = runtime_results["native_pytorch_fp32"]
        for runtime, pair_name in _PAIR_NAMES.items():
            candidate = runtime_results[runtime]
            natural = materialization["natural_input_tokens"]
            pointer_keys = sorted(set(reference["pointer"]) | set(candidate["pointer"]))
            pointer_start_delta = 0.0
            pointer_end_delta = 0.0
            pointer_span_exact_count = 0
            pointer_records = {}
            for arg in pointer_keys:
                reference_pointer = reference["pointer"].get(arg)
                candidate_pointer = candidate["pointer"].get(arg)
                if reference_pointer is None or candidate_pointer is None:
                    pointer_records[arg] = {
                        "present_in_both": False,
                        "span_exact": False,
                    }
                    continue
                reference_start = _score_array_from_pointer(
                    reference_pointer, "start_scores", reference, arg, "start"
                )
                candidate_start = _score_array_from_pointer(
                    candidate_pointer, "start_scores", candidate, arg, "start"
                )
                reference_end = _score_array_from_pointer(
                    reference_pointer, "end_scores", reference, arg, "end"
                )
                candidate_end = _score_array_from_pointer(
                    candidate_pointer, "end_scores", candidate, arg, "end"
                )
                start_delta = _maximum_abs_diff(reference_start, candidate_start)
                end_delta = _maximum_abs_diff(reference_end, candidate_end)
                span_exact = (
                    reference_pointer["start"] == candidate_pointer["start"]
                    and reference_pointer["end"] == candidate_pointer["end"]
                    and reference_pointer["text"] == candidate_pointer["text"]
                )
                pointer_start_delta = max(pointer_start_delta, start_delta)
                pointer_end_delta = max(pointer_end_delta, end_delta)
                pointer_span_exact_count += int(span_exact)
                pointer_records[arg] = {
                    "present_in_both": True,
                    "start_scores_max_abs_diff": start_delta,
                    "end_scores_max_abs_diff": end_delta,
                    "span_exact": span_exact,
                }

            comparison = {
                "hidden_full_max_abs_diff": _maximum_abs_diff(
                    hidden_by_runtime["native_pytorch_fp32"][case_index],
                    hidden_by_runtime[runtime][case_index],
                ),
                "hidden_natural_span_max_abs_diff": _maximum_abs_diff(
                    hidden_by_runtime["native_pytorch_fp32"][case_index, :natural],
                    hidden_by_runtime[runtime][case_index, :natural],
                ),
                "decision_hidden_max_abs_diff": _maximum_abs_diff(
                    hidden_by_runtime["native_pytorch_fp32"][
                        case_index, natural - 1
                    ],
                    hidden_by_runtime[runtime][case_index, natural - 1],
                ),
                "route_logits_max_abs_diff": _maximum_abs_diff(
                    reference["_route_logits"],
                    candidate["_route_logits"],
                ),
                "selector_scores_max_abs_diff": _maximum_abs_diff(
                    reference["_selector_scores"],
                    candidate["_selector_scores"],
                ),
                "pointer_start_scores_max_abs_diff": pointer_start_delta,
                "pointer_end_scores_max_abs_diff": pointer_end_delta,
                "route_confidence_max_abs_diff": abs(
                    float(reference["route"]["decision_confidence"])
                    - float(candidate["route"]["decision_confidence"])
                ),
                "selector_confidence_max_abs_diff": abs(
                    float(reference["selector"]["selected_confidence"])
                    - float(candidate["selector"]["selected_confidence"])
                ),
                "action_confidence_max_abs_diff": abs(
                    float(reference["action_confidence"])
                    - float(candidate["action_confidence"])
                ),
                "route_exact": reference["route"]["decision"] == candidate["route"]["decision"],
                "selected_tool_exact": (
                    reference["selector"]["selected_tool"]
                    == candidate["selector"]["selected_tool"]
                ),
                "pointer_span_exact": pointer_span_exact_count == len(pointer_keys),
                "pointer_span_comparisons": len(pointer_keys),
                "grounded_args_exact": (
                    reference["grounded_args"] == candidate["grounded_args"]
                ),
                "final_normalized_action_exact": (
                    reference["normalized_action"] == candidate["normalized_action"]
                ),
                "pointer": pointer_records,
            }
            comparisons[pair_name] = comparison
            aggregate = aggregate_pairs[pair_name]
            for metric in (
                "hidden_full_max_abs_diff",
                "hidden_natural_span_max_abs_diff",
                "decision_hidden_max_abs_diff",
                "route_logits_max_abs_diff",
                "selector_scores_max_abs_diff",
                "pointer_start_scores_max_abs_diff",
                "pointer_end_scores_max_abs_diff",
                "route_confidence_max_abs_diff",
                "selector_confidence_max_abs_diff",
                "action_confidence_max_abs_diff",
                "confidence_max_abs_diff",
            ):
                if metric == "confidence_max_abs_diff":
                    aggregate[metric] = max(
                        aggregate[metric],
                        comparison["route_confidence_max_abs_diff"],
                        comparison["selector_confidence_max_abs_diff"],
                        comparison["action_confidence_max_abs_diff"],
                    )
                else:
                    aggregate[metric] = max(aggregate[metric], comparison[metric])
            aggregate["route_exact"] += int(comparison["route_exact"])
            aggregate["selected_tool_exact"] += int(comparison["selected_tool_exact"])
            aggregate["pointer_span_exact"] += pointer_span_exact_count
            aggregate["pointer_span_comparisons"] += len(pointer_keys)
            aggregate["grounded_args_exact"] += int(comparison["grounded_args_exact"])
            aggregate["final_normalized_action_exact"] += int(
                comparison["final_normalized_action_exact"]
            )

        report_rows.append(
            {
                "case_id": case["id"],
                "family": case["family"],
                "query_sha256": hashlib.sha256(case["query"].encode("utf-8")).hexdigest(),
                "expected_action": expected_action,
                "input": {
                    key: value
                    for key, value in materialization.items()
                    if key != "ids"
                },
                "runtimes": {
                    runtime: _strip_private_runtime(result)
                    for runtime, result in runtime_results.items()
                },
                "comparisons": comparisons,
            }
        )

    pair_passes = {}
    total_cases = len(cases)
    for pair_name, aggregate in aggregate_pairs.items():
        thresholds = STRUCTURED_ACTION_PARITY_THRESHOLDS[pair_name]
        numerical_checks = {
            "hidden": aggregate["hidden_full_max_abs_diff"]
            <= thresholds["hidden_max_abs_diff"],
            "route_logits": aggregate["route_logits_max_abs_diff"]
            <= thresholds["route_logits_max_abs_diff"],
            "selector_scores": aggregate["selector_scores_max_abs_diff"]
            <= thresholds["selector_scores_max_abs_diff"],
            "pointer_start_scores": aggregate["pointer_start_scores_max_abs_diff"]
            <= thresholds["pointer_scores_max_abs_diff"],
            "pointer_end_scores": aggregate["pointer_end_scores_max_abs_diff"]
            <= thresholds["pointer_scores_max_abs_diff"],
            "confidence": aggregate["confidence_max_abs_diff"]
            <= thresholds["confidence_max_abs_diff"],
        }
        exact_checks = {
            "route": aggregate["route_exact"] == total_cases,
            "selected_tool": aggregate["selected_tool_exact"] == total_cases,
            "pointer_spans": (
                aggregate["pointer_span_exact"]
                == aggregate["pointer_span_comparisons"]
            ),
            "grounded_args": aggregate["grounded_args_exact"] == total_cases,
            "final_normalized_action": (
                aggregate["final_normalized_action_exact"] == total_cases
            ),
        }
        pair_passes[pair_name] = {
            "numerical_checks": numerical_checks,
            "exact_checks": exact_checks,
            "passed": all(numerical_checks.values()) and all(exact_checks.values()),
        }

    tool_required_cases = sum(
        case["expected"].get("abstain") is not True for case in cases
    )
    for diagnostic in runtime_diagnostics.values():
        diagnostic["route_accuracy"] = diagnostic["route_correct"] / total_cases
        diagnostic["selected_tool_accuracy_on_tool_rows"] = (
            diagnostic["selected_tool_correct"] / tool_required_cases
        )
        diagnostic["schema_validity_rate"] = diagnostic["schema_valid"] / total_cases
        diagnostic["exact_action_accuracy"] = diagnostic["exact_action"] / total_cases
        diagnostic["tool_required_exact_action_accuracy"] = (
            diagnostic["tool_required_exact_action"] / tool_required_cases
        )

    passed = serialization_audit["passed"] and all(
        pair["passed"] for pair in pair_passes.values()
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "localagent_full_stack_structured_action_export_parity",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "diagnostic_status": {
            "suite_role": "diagnostic_reuse",
            "suite_reused_from": "corrected WebGPU complete-action benchmark protocol",
            "independent_capability_estimate": False,
            "purpose": (
                "deployment parity and corrected feature-materialization validation only; "
                "the repeated suite score is not a new held-out capability result"
            ),
        },
        "contract": {
            "policy": "structured_one_forward",
            "target_input_tokens": target_input_tokens,
            "materialization": (
                "append one-token spaces after the complete natural prompt until exactly "
                f"{target_input_tokens} graph tokens"
            ),
            "decision_feature": "hidden[natural_input_tokens - 1]",
            "pointer_domain": "token and hidden positions [0, natural_input_tokens)",
            "route_and_selector_reference": "exact checkpoint fp32 heads",
            "onnx_head_application": (
                "rounded JSON dispatch/pointer heads with the browser Float32 output convention"
            ),
            "grounding": (
                "exact app.js groundFromSchema and groundedArgsValid executed under Node.js"
            ),
            "normalization": (
                "exact benchmark.js normalizeBenchmarkAction executed under Node.js"
            ),
            "hard_exact_gates": [
                "route decision",
                "selected tool (including null on route stop)",
                "pointer span and decoded text",
                "grounded typed arguments",
                "final normalized action",
            ],
        },
        "thresholds": STRUCTURED_ACTION_PARITY_THRESHOLDS,
        "execution": {
            "device": "cpu",
            "native_dtype": "fp32",
            "onnx_fp32_graph_internal_dtype": "fp32",
            "onnx_fp16_graph_internal_dtype": "fp16_with_fp32_hidden_output",
            "onnx_provider": "CPUExecutionProvider",
            "onnx_fp32_session_providers": fp32_providers,
            "onnx_fp16_session_providers": fp16_providers,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__,
            "python_version": sys.version.split()[0],
            "node_version": node_version,
            "platform": platform.platform(),
            "batch_size": batch_size,
        },
        "checkpoint": {
            **checkpoint_identity,
            "stage": checkpoint.get("stage"),
            "step": checkpoint.get("step"),
            "model_name": cfg.name,
            "model_parameters": model.num_params(),
            "model_config": {
                key: getattr(cfg, key) for key in ModelConfig.__dataclass_fields__
            },
        },
        "identities": {
            "bundle_manifest": bundle_manifest_identity,
            "bundle_artifacts": artifact_identities,
            "action_suite_source": source_suite_identity,
            "action_suite_bundle_copy": bundle_suite_identity,
            "app_js_source": app_source_identity,
            "app_js_bundle_copy": app_bundle_identity,
            "benchmark_js_source": benchmark_source_identity,
            "benchmark_js_bundle_copy": benchmark_bundle_identity,
        },
        "manifest_validation": {
            "checkpoint_and_required_bundle_artifacts_match": True,
            "source_and_bundle_suite_match": True,
            "source_and_bundle_app_js_match": True,
            "source_and_bundle_benchmark_js_match": True,
        },
        "head_serialization": serialization_audit,
        "aggregate": {
            "configured_cases": total_cases,
            "eligible_cases": total_cases,
            "tool_required_cases": tool_required_cases,
            "abstention_cases": total_cases - tool_required_cases,
            "all_inputs_exact_target_length": all(
                row["input_tokens"] == target_input_tokens
                for row in materializations
            ),
            "all_decision_indices_natural": all(
                row["decision_feature_index"]
                == row["natural_input_tokens"] - 1
                for row in materializations
            ),
            "all_pointer_domains_natural": all(
                row["pointer_domain"] == [0, row["natural_input_tokens"]]
                for row in materializations
            ),
            "runtime_diagnostics": runtime_diagnostics,
            "comparisons": aggregate_pairs,
            "pair_gates": pair_passes,
        },
        "cases": report_rows,
        "summary_sha256": "",
    }
    payload["summary_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "summary_sha256"}
    )
    if not passed:
        raise RuntimeError(
            "structured action export parity failed; refusing to publish a passing artifact"
        )
    return payload


def _score_array_from_pointer(
    pointer_record: dict[str, Any],
    identity_key: str,
    runtime_record: dict[str, Any],
    arg: str,
    score_kind: str,
) -> np.ndarray:
    """Recover scores retained privately while keeping the public record compact."""

    private_scores = runtime_record.get("_pointer_scores", {}).get(arg, {})
    private_key = f"{score_kind}_scores"
    if private_key not in private_scores:
        raise RuntimeError(
            f"missing private pointer scores for {arg}:{score_kind}; "
            f"public identity was {pointer_record[identity_key]}"
        )
    return private_scores[private_key]


def write_structured_action_parity(payload: dict[str, Any], path: str | Path) -> None:
    """Write one deterministic, self-hashed parity report."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
