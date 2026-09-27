from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from localagent.eval.webgpu_decode_receipt import (
    HARNESS_HTML_BYTES,
    HARNESS_HTML_SHA256,
    HARNESS_JAVASCRIPT_BYTES,
    HARNESS_JAVASCRIPT_SHA256,
    HARNESS_ORT_VENDOR_PATH,
    HARNESS_ORT_VERSION,
)

ROOT = Path(__file__).parents[1]
WEB_DIR = ROOT / "spaces" / "localagent-webgpu"
DECODE_JS = WEB_DIR / "decode-benchmark.js"
DECODE_HTML = WEB_DIR / "decode-benchmark.html"
SHA = "a" * 64
DEFAULT_V2_BUNDLE = (
    ROOT / "runs" / "webgpu" / "random-cached-decode-latency-seed-20260728-v2"
)
DEFAULT_V2_BUNDLE_ARTIFACTS = (
    DEFAULT_V2_BUNDLE / "matched-decode.json",
    *(
        DEFAULT_V2_BUNDLE / arm / filename
        for arm in ("attention", "hybrid")
        for filename in (
            "decode.fp16.onnx",
            "decode.fp32.onnx",
            "model-config.yaml",
            "prefill.fp16.onnx",
            "prefill.fp32.onnx",
            "provenance.json",
        )
    ),
)


def _manifest() -> dict:
    controlled = [
        "conv_kernel",
        "d_model",
        "dropout",
        "embed_dim",
        "max_seq_len",
        "n_heads",
        "n_kv_heads",
        "n_layers",
        "n_loops",
        "norm_eps",
        "qk_norm",
        "rope_theta",
        "tie_embeddings",
        "vocab_size",
    ]
    return {
        "schema_version": 1,
        "artifact_type": "matched_random_cached_decode_latency_suite",
        "latency_only": True,
        "trained": False,
        "capability_artifact": False,
        "quality_claims": [],
        "shared_random_seed": 19,
        "controlled_fields": controlled,
        "intentional_differences": {
            "ffn_hidden": {
                "hybrid_treatment": 64,
                "all_attention_control": 64,
            },
            "layer_types": {
                "hybrid_treatment": ["attn", "conv"],
                "all_attention_control": ["attn", "attn"],
            },
            "name": {
                "hybrid_treatment": "test-hybrid",
                "all_attention_control": "test-attention",
            },
        },
        "match": {
            "hybrid_parameters": 34_000_000,
            "attention_parameters": 34_000_000,
            "relative_parameter_delta": 0.0,
        },
        "artifacts": {
            "hybrid/provenance.json": {"bytes": 100, "sha256": SHA},
            "attention/provenance.json": {"bytes": 100, "sha256": SHA},
        },
        "models": {
            "hybrid_treatment": {
                "directory": "hybrid",
                "name": "test-hybrid",
                "provenance": "hybrid/provenance.json",
            },
            "all_attention_control": {
                "directory": "attention",
                "name": "test-attention",
                "provenance": "attention/provenance.json",
            },
        },
    }


def _typed_io(
    names: list[str],
    *,
    cache_dtype: str,
    decode: bool,
    outputs: bool,
) -> list[dict]:
    values = []
    for name in names:
        dtype = "int64" if name in {"input_ids", "next_token"} else cache_dtype
        if name == "input_ids":
            shape = ["batch", 1 if decode else "prompt_sequence"]
        elif name == "next_token":
            shape = ["batch"]
        elif name == "logits":
            shape = ["batch", "vocab_size"]
        elif name.endswith("_conv"):
            shape = ["batch", 32, 2]
        else:
            shape = ["batch", 1, "cache_sequence", 8]
        values.append({"name": name, "dtype": dtype, "shape": shape})
    return values


def _precision_graph(precision: str, slots: list[dict]) -> dict:
    cache_dtype = "float16" if precision == "fp16" else "float32"
    past = [name for slot in slots for name in slot["past_inputs"]]
    present = [name for slot in slots for name in slot["present_outputs"]]
    prefill_inputs = ["input_ids"]
    prefill_outputs = ["next_token", "logits", *present]
    decode_inputs = ["input_ids", *past]
    decode_outputs = ["next_token", "logits", *present]
    return {
        "cache_dtype": cache_dtype,
        "prefill": {
            "file": f"prefill.{precision}.onnx",
            "input_names": prefill_inputs,
            "inputs": _typed_io(
                prefill_inputs,
                cache_dtype=cache_dtype,
                decode=False,
                outputs=False,
            ),
            "output_names": prefill_outputs,
            "outputs": _typed_io(
                prefill_outputs,
                cache_dtype=cache_dtype,
                decode=False,
                outputs=True,
            ),
        },
        "decode": {
            "file": f"decode.{precision}.onnx",
            "input_names": decode_inputs,
            "inputs": _typed_io(
                decode_inputs,
                cache_dtype=cache_dtype,
                decode=True,
                outputs=False,
            ),
            "output_names": decode_outputs,
            "outputs": _typed_io(
                decode_outputs,
                cache_dtype=cache_dtype,
                decode=True,
                outputs=True,
            ),
        },
    }


def _provenance(role: str, name: str) -> dict:
    slots = [
        {
            "slot": 0,
            "loop": 0,
            "layer": 0,
            "kind": "attn",
            "past_inputs": ["past_0_key", "past_0_value"],
            "present_outputs": ["present_0_key", "present_0_value"],
            "shape": ["batch", 1, "cache_sequence", 8],
            "update": "append_one_token_along_axis_2",
            "dtype_by_precision": {"fp32": "float32", "fp16": "float16"},
        },
        {
            "slot": 1,
            "loop": 0,
            "layer": 1,
            "kind": "conv",
            "past_inputs": ["past_1_conv"],
            "present_outputs": ["present_1_conv"],
            "shape": ["batch", 32, 2],
            "update": "replace_with_latest_fixed_width_tail",
            "dtype_by_precision": {"fp32": "float32", "fp16": "float16"},
        },
    ]
    artifacts = {
        "model-config.yaml": {
            "file": "model-config.yaml",
            "bytes": 100,
            "precision": "text",
            "sha256": SHA,
        }
    }
    for precision in ("fp32", "fp16"):
        for graph in ("prefill", "decode"):
            filename = f"{graph}.{precision}.onnx"
            artifacts[filename] = {
                "file": filename,
                "bytes": 100,
                "precision": precision,
                "sha256": SHA,
            }
    config = {
        "name": name,
        "vocab_size": 256,
        "d_model": 32,
        "embed_dim": 32,
        "ffn_hidden": 64,
        "n_layers": 2,
        "n_loops": 1,
        "n_heads": 4,
        "n_kv_heads": 1,
        "max_seq_len": 2048,
        "rope_theta": 10000.0,
        "norm_eps": 1e-5,
        "tie_embeddings": True,
        "dropout": 0.0,
        "qk_norm": True,
        "conv_kernel": 3,
        "layer_types": ["attn", "conv"],
    }
    parity = {
        "fixture_contract": "ids[i]=(131*i+17+977*fixture_index) mod vocab_size",
        "hard_gate": True,
        "results": {},
    }
    for precision in ("fp32", "fp16"):
        cache_dtype = "float16" if precision == "fp16" else "float32"
        decode = [
            {
                "decode_step": index + 1,
                "next_token_exact": True,
                "cached_vs_full_context_next_token_exact": True,
                "cache_max_abs_diff": 0.0,
                "cached_vs_full_context_logits_max_abs_diff": 0.0,
                "logits_max_abs_diff": 0.0,
            }
            for index in range(3)
        ]
        parity["results"][precision] = {
            "artifacts": {
                "prefill": {"bytes": 100, "sha256": SHA},
                "decode": {"bytes": 100, "sha256": SHA},
            },
            "cache_atol": 0.1 if precision == "fp16" else 0.001,
            "cache_dtype": cache_dtype,
            "decode_steps": 3,
            "greedy_next_token_exact": True,
            "hard_gate": True,
            "final_token_logits_shape": ["batch", config["vocab_size"]],
            "logits_atol": 0.1 if precision == "fp16" else 0.001,
            "max_cache_abs_diff": 0.0,
            "max_cached_vs_full_context_logits_abs_diff": 0.0,
            "max_logits_abs_diff": 0.0,
            "passed": True,
            "per_fixture": [
                {
                    "decode": decode,
                    "input_ids_sha256": digest * 64,
                    "prefill_cache_max_abs_diff": 0.0,
                    "prefill_next_token_exact": True,
                    "prefill_cached_vs_full_context_next_token_exact": True,
                    "prefill_cached_vs_full_context_logits_max_abs_diff": 0.0,
                    "prefill_logits_max_abs_diff": 0.0,
                    "prompt_length": prompt_length,
                }
                for prompt_length, digest in ((8, "a"), (16, "b"))
            ],
            "provider": "CPUExecutionProvider",
            "reference": "exact in-memory LocalAgentLM random initialization",
            "reference_independence": {
                "onnx_logits_vs_pytorch_cached_path": True,
                "onnx_vs_pytorch_cached_path": True,
                "pytorch_cached_vs_fresh_full_context_logits": True,
                "pytorch_cached_vs_fresh_full_context_greedy_token": True,
            },
        }
    return {
        "schema_version": 1,
        "artifact_type": "random_weight_cached_decode_onnx",
        "latency_only": True,
        "trained": False,
        "training_steps": 0,
        "capability_artifact": False,
        "quality_claims": [],
        "model": {
            "name": name,
            "pair_role": role,
            "config": config,
            "config_canonical_sha256": SHA,
            "config_source_sha256": SHA,
            "full_model_parameters": 34_000_000,
        },
        "weights": {
            "source": "deterministic_random_initialization",
            "checkpoint": None,
            "seed": 19,
            "state_dict_sha256": SHA,
        },
        "graph_contract": {
            "cache_slots": slots,
            "cache_update_strategy": (
                "attention K/V append one token; short-conv state replaces its fixed-width tail"
            ),
            "prefill_projection": (
                "only the final normalized prompt feature is projected to vocabulary logits"
            ),
            "decode_token_axis_fixed_one": True,
            "decode_position": {
                "caller_position_input": False,
                "derived_from": "past_0_key",
                "rule": "RoPE position = first attention past-key axis-2 length",
            },
            "graphs": {
                precision: _precision_graph(precision, slots)
                for precision in ("fp32", "fp16")
            },
            "next_token": {
                "name": "next_token",
                "dtype": "int64",
                "shape": ["batch"],
                "decode": "compatibility argmax over the exported final-token logits",
            },
            "logits": {
                "name": "logits",
                "description": "unnormalized LM scores for the final input token only",
                "shape": ["batch", config["vocab_size"]],
                "dtype_by_precision": {
                    "fp32": "float32",
                    "fp16": "float16",
                },
            },
        },
        "artifacts": artifacts,
        "parity": parity,
    }


def _trained_manifest() -> dict:
    manifest = copy.deepcopy(_manifest())
    manifest.update(
        {
            "artifact_type": "matched_trained_cached_decode_suite",
            "latency_only": False,
            "trained": True,
            "capability_artifact": False,
            "quality_claims": [],
            "quality_evaluation": {
                "included": False,
                "required_separately": True,
            },
            "tokenizer": {
                "artifact": "data/tokenizer-webgpu-proxy-16k.json",
                "artifact_identity": {
                    "bytes": 100,
                    "sha256": "c" * 64,
                },
                "checkpoint_metadata_present": True,
                "kind": "bpe",
                "sha256": "c" * 64,
                "verified": True,
                "vocab_size": 16_384,
            },
            "checkpoints": {
                "hybrid_treatment": {
                    "bytes": 1_000,
                    "checkpoint": "runs/hybrid/latest.pt",
                    "sha256": "d" * 64,
                    "stage": "pretrain",
                    "step": 4,
                    "tokens_seen": 12_345,
                    "training_steps": 5,
                },
                "all_attention_control": {
                    "bytes": 1_100,
                    "checkpoint": "runs/attention/latest.pt",
                    "sha256": "e" * 64,
                    "stage": "pretrain",
                    "step": 4,
                    "tokens_seen": 12_345,
                    "training_steps": 5,
                },
            },
        }
    )
    manifest.pop("shared_random_seed")
    return manifest


def _trained_provenance(role: str, name: str, manifest: dict) -> dict:
    provenance = copy.deepcopy(_provenance(role, name))
    checkpoint = manifest["checkpoints"][role]
    provenance.update(
        {
            "artifact_type": "trained_checkpoint_cached_decode_onnx",
            "latency_only": False,
            "trained": True,
            "training_steps": checkpoint["training_steps"],
            "checkpoint_step": checkpoint["step"],
            "tokens_seen": checkpoint["tokens_seen"],
            "input_tokens_seen": 12_400,
            "capability_artifact": False,
            "capability_metrics": None,
            "quality_claims": [],
            "quality_evaluation": {
                "included": False,
                "required_separately": True,
                "scope": (
                    "Export validates graph parity only; held-out CE/BPB and downstream "
                    "capability metrics are separate artifacts."
                ),
            },
            "tokenizer": copy.deepcopy(manifest["tokenizer"]),
            "weights": {
                "checkpoint": checkpoint["checkpoint"],
                "checkpoint_bytes": checkpoint["bytes"],
                "checkpoint_sha256": checkpoint["sha256"],
                "checkpoint_stage": checkpoint["stage"],
                "checkpoint_step": checkpoint["step"],
                "input_tokens_seen": 12_400,
                "source": "strict_lineage_validated_lm_checkpoint",
                "state_dict_sha256": SHA,
                "tokens_seen": checkpoint["tokens_seen"],
            },
        }
    )
    provenance["model"]["config"]["vocab_size"] = manifest["tokenizer"]["vocab_size"]
    provenance["graph_contract"]["logits"]["shape"][1] = manifest["tokenizer"]["vocab_size"]
    for parity in provenance["parity"]["results"].values():
        parity["reference"] = "exact in-memory LocalAgentLM checkpoint weights"
        parity["final_token_logits_shape"][1] = manifest["tokenizer"]["vocab_size"]
    return provenance


def _single_manifest(provenance: dict) -> dict:
    role = provenance["model"]["pair_role"]
    return {
        "schema_version": 1,
        "artifact_type": "single_trained_cached_decode_suite",
        "latency_only": False,
        "trained": True,
        "capability_artifact": False,
        "quality_claims": [],
        "quality_evaluation": {
            "included": False,
            "required_separately": True,
        },
        "model": {
            "name": provenance["model"]["name"],
            "pair_role": role,
            "provenance": "provenance.json",
        },
        "artifacts": {
            "provenance.json": {
                "bytes": 4_000,
                "sha256": "9" * 64,
            },
        },
    }


def _single_export_provenance() -> dict:
    pair_manifest = _trained_manifest()
    role = "hybrid_treatment"
    provenance = _trained_provenance(
        role,
        pair_manifest["models"][role]["name"],
        pair_manifest,
    )
    tokenizer_identity = {
        "bytes": 100,
        "sha256": pair_manifest["tokenizer"]["sha256"],
    }
    provenance["tokenizer"].update(
        {
            "bundled_artifact_identity": copy.deepcopy(tokenizer_identity),
            "encoding": "bytelevel-bpe",
            "eos_id": 0,
            "file": "tokenizer.json",
            "pad_id": 0,
        }
    )
    provenance["artifacts"].update(
        {
            "meta.json": {
                "file": "meta.json",
                "bytes": 100,
                "precision": "metadata",
                "sha256": "8" * 64,
            },
            "tokenizer.json": {
                "file": "tokenizer.json",
                "bytes": tokenizer_identity["bytes"],
                "precision": "tokenizer",
                "sha256": tokenizer_identity["sha256"],
            },
        }
    )
    return provenance


def _single_runtime_metadata(provenance: dict) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "localagent_cached_autoregressive_onnx",
        "default_precision": "fp16",
        "graph_contract": copy.deepcopy(provenance["graph_contract"]),
        "model": {
            "config": copy.deepcopy(provenance["model"]["config"]),
            "config_canonical_sha256": provenance["model"]["config_canonical_sha256"],
            "config_file": "model-config.yaml",
            "parameters": provenance["model"]["full_model_parameters"],
        },
        "checkpoint": {
            "sha256": provenance["weights"]["checkpoint_sha256"],
            "stage": provenance["weights"]["checkpoint_stage"],
            "step": provenance["weights"]["checkpoint_step"],
        },
        "tokenizer": {
            "kind": provenance["tokenizer"]["kind"],
            "sha256": provenance["tokenizer"]["sha256"],
            "vocab_size": provenance["tokenizer"]["vocab_size"],
            "verified": True,
            "file": "tokenizer.json",
        },
    }


def _run_node(script: str, *arguments: object) -> dict:
    result = subprocess.run(
        [
            shutil.which("node"),
            "-e",
            script,
            str(DECODE_JS),
            *(json.dumps(value) for value in arguments),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_decode_manifest_and_precision_specific_provenance_contract():
    manifest = _manifest()
    role = "hybrid_treatment"
    provenance = _provenance(role, manifest["models"][role]["name"])
    script = """
global.window = { __localAgentSkipInit: true };
const api = require(process.argv[1]);
const manifest = JSON.parse(process.argv[2]);
const provenance = JSON.parse(process.argv[3]);
const acceptedManifest = api.validateDecodeManifest(manifest);
const acceptedProvenance = api.validateDecodeProvenance(
  provenance, "hybrid_treatment", manifest.models.hybrid_treatment, 19
);
let rejected = null;
try {
  api.validateDecodeProvenance({
    ...provenance,
    graph_contract: {
      ...provenance.graph_contract,
      graphs: {
        ...provenance.graph_contract.graphs,
        fp16: {
          ...provenance.graph_contract.graphs.fp16,
          cache_dtype: "float32",
        },
      },
    },
  }, "hybrid_treatment", manifest.models.hybrid_treatment, 19);
} catch (error) {
  rejected = error.message;
}
let shortParityRejected = null;
try {
  const shortParity = JSON.parse(JSON.stringify(provenance));
  shortParity.parity.results.fp16.decode_steps = 2;
  api.validateDecodeProvenance(
    shortParity, "hybrid_treatment", manifest.models.hybrid_treatment, 19
  );
} catch (error) {
  shortParityRejected = error.message;
}
let precisionRejected = null;
try {
  const bogusPrecision = JSON.parse(JSON.stringify(provenance));
  bogusPrecision.graph_contract.graphs.int8 =
    bogusPrecision.graph_contract.graphs.fp32;
  api.validateDecodeProvenance(
    bogusPrecision, "hybrid_treatment", manifest.models.hybrid_treatment, 19
  );
} catch (error) {
  precisionRejected = error.message;
}
let metadataRejected = null;
try {
  const malformedMetadata = JSON.parse(JSON.stringify(provenance));
  malformedMetadata.parity.fixture_lengths = {};
  api.validateDecodeProvenance(
    malformedMetadata, "hybrid_treatment", manifest.models.hybrid_treatment, 19
  );
} catch (error) {
  metadataRejected = error.message;
}
process.stdout.write(JSON.stringify({
  artifactType: acceptedManifest.artifact_type,
  slotCount: acceptedProvenance.graph_contract.cache_slots.length,
  rejected,
  shortParityRejected,
  precisionRejected,
  metadataRejected,
}));
"""
    result = _run_node(script, manifest, provenance)
    assert result["artifactType"] == "matched_random_cached_decode_latency_suite"
    assert result["slotCount"] == 2
    assert "fp16 graph contract" in result["rejected"]
    assert "trajectory parity" in result["shortParityRejected"]
    assert "unsupported precision" in result["precisionRejected"]
    assert "partially declared" in result["metadataRejected"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_trained_decode_manifest_and_provenance_are_checkpoint_and_tokenizer_bound():
    manifest = _trained_manifest()
    role = "hybrid_treatment"
    provenance = _trained_provenance(
        role,
        manifest["models"][role]["name"],
        manifest,
    )
    script = """
global.window = { __localAgentSkipInit: true };
const api = require(process.argv[1]);
const manifest = JSON.parse(process.argv[2]);
const provenance = JSON.parse(process.argv[3]);
const acceptedManifest = api.validateDecodeManifest(manifest);
const acceptedProvenance = api.validateDecodeProvenance(
  provenance,
  "hybrid_treatment",
  manifest.models.hybrid_treatment,
  manifest
);
const mode = api.decodeManifestMode(acceptedManifest);
const labels = api.decodeLabelsForMode(mode);
process.stdout.write(JSON.stringify({
  mode,
  artifactType: acceptedProvenance.artifact_type,
  checkpointSha256: acceptedProvenance.weights.checkpoint_sha256,
  tokenizerSha256: acceptedProvenance.tokenizer.sha256,
  estimand: api.summarizeDecodeRecords([], mode).estimand,
  labels,
}));
"""
    result = _run_node(script, manifest, provenance)
    assert result["mode"] == "trained"
    assert result["artifactType"] == "trained_checkpoint_cached_decode_onnx"
    assert result["checkpointSha256"] == "d" * 64
    assert result["tokenizerSha256"] == "c" * 64
    assert result["estimand"] == "trained_weight_cached_autoregressive_graph_latency"
    assert result["labels"]["benchmark_label"] == (
        "trained weights, latency only; quality scored separately"
    )
    assert result["labels"]["latency_only"] is True
    assert result["labels"]["trained_weights"] is True
    assert result["labels"]["capability_artifact"] is False
    assert result["labels"]["action_capability_claimed"] is False
    assert result["labels"]["quality_scored_separately"] is True


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_single_trained_decode_manifest_consumes_exporter_provenance_fail_closed():
    provenance = _single_export_provenance()
    manifest = _single_manifest(provenance)
    metadata = _single_runtime_metadata(provenance)
    script = """
global.window = { __localAgentSkipInit: true };
const api = require(process.argv[1]);
const manifest = JSON.parse(process.argv[2]);
const provenance = JSON.parse(process.argv[3]);
const metadata = JSON.parse(process.argv[4]);
function rejection(callback) {
  try {
    callback();
    return null;
  } catch (error) {
    return error.message;
  }
}
const acceptedManifest = api.validateSingleDecodeManifest(manifest);
const acceptedProvenance = api.validateSingleDecodeProvenance(
  provenance, acceptedManifest.model
);
const acceptedMetadata = api.validateCachedRuntimeMetadata(
  metadata, acceptedProvenance, "fp16"
);
const unpinned = JSON.parse(JSON.stringify(manifest));
delete unpinned.artifacts["provenance.json"];
const tokenizerMismatch = JSON.parse(JSON.stringify(provenance));
tokenizerMismatch.artifacts["tokenizer.json"].sha256 = "7".repeat(64);
const metadataMismatch = JSON.parse(JSON.stringify(metadata));
metadataMismatch.model.parameters += 1;
process.stdout.write(JSON.stringify({
  manifestType: acceptedManifest.artifact_type,
  model: acceptedProvenance.model.name,
  metadataType: acceptedMetadata.artifact_type,
  unpinned: rejection(() => api.validateSingleDecodeManifest(unpinned)),
  tokenizerMismatch: rejection(() =>
    api.validateSingleDecodeProvenance(tokenizerMismatch, manifest.model)
  ),
  metadataMismatch: rejection(() =>
    api.validateCachedRuntimeMetadata(metadataMismatch, provenance, "fp16")
  ),
}));
"""
    result = _run_node(script, manifest, provenance, metadata)
    assert result["manifestType"] == "single_trained_cached_decode_suite"
    assert result["model"] == provenance["model"]["name"]
    assert result["metadataType"] == "localagent_cached_autoregressive_onnx"
    assert "pin exactly its provenance" in result["unpinned"]
    assert "bundled tokenizer" in result["tokenizerMismatch"]
    assert "runtime metadata disagrees" in result["metadataMismatch"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_single_decode_query_schedule_and_webgpu_evidence_are_explicit():
    script = """
global.window = { __localAgentSkipInit: true };
const api = require(process.argv[1]);
function rejection(callback) {
  try {
    callback();
    return null;
  } catch (error) {
    return error.message;
  }
}
const query = api.validateSingleDecodeQuery({
  provenance: "trained/provenance.json",
  sha256: "a".repeat(64),
  bytes: "4096",
});
const schedule = api.buildDecodeSchedule(
  ["trained-1m"], [128, 512], 30, "single", "measured", "single"
);
window.__localAgentDecodeAcceptanceRootSha256 = "c".repeat(64);
const acceptanceRoot = api.requestedDecodeAcceptanceRootSha256();
const evidence = api.validateRequiredWebGpuEvidence("webgpu", {
  provider_requested: "webgpu",
  provider_actual: null,
  exact_provider_request_and_session_creation_observed: true,
  execution_provider_list: ["webgpu"],
  whole_session_provider_retry: false,
  graph_wide_provider_verified: false,
  per_node_placement_verified: false,
  per_node_placement_status: "unknown",
  per_node_fallback_status: "unknown",
  ort_webgpu: { ort_adapter_available: true, ort_device_available: true },
}, 2);
process.stdout.write(JSON.stringify({
  query,
  scheduleLength: schedule.length,
  counts: schedule.reduce((result, row) => {
    result[row.input_tokens] = (result[row.input_tokens] || 0) + 1;
    return result;
  }, {}),
  providerVerified: evidence.required_verification_passed,
  providerActual: evidence.provider_actual,
  graphWideVerified: evidence.graph_wide_provider_verified,
  acceptanceRoot,
  missingHash: rejection(() => api.validateSingleDecodeQuery({
    provenance: "trained/provenance.json", bytes: 4096,
  })),
  wasm: rejection(() => api.validateRequiredWebGpuEvidence("wasm", {
    provider_requested: "wasm",
    provider_actual: "wasm",
    execution_provider_list: ["wasm"],
    whole_session_provider_retry: false,
    ort_webgpu: { ort_device_available: false },
  }, 2)),
  noDevice: rejection(() => api.validateRequiredWebGpuEvidence("webgpu", {
    provider_requested: "webgpu",
    provider_actual: null,
    exact_provider_request_and_session_creation_observed: true,
    execution_provider_list: ["webgpu"],
    whole_session_provider_retry: false,
    graph_wide_provider_verified: false,
    per_node_placement_verified: false,
    per_node_placement_status: "unknown",
    per_node_fallback_status: "unknown",
    ort_webgpu: { ort_adapter_available: true, ort_device_available: false },
  }, 2)),
}));
"""
    result = _run_node(script)
    assert result["query"] == {
        "provenance": "trained/provenance.json",
        "sha256": "a" * 64,
        "bytes": 4096,
    }
    assert result["scheduleLength"] == 60
    assert result["counts"] == {"128": 30, "512": 30}
    assert result["providerVerified"] is True
    assert result["providerActual"] is None
    assert result["graphWideVerified"] is False
    assert result["acceptanceRoot"] == "c" * 64
    assert "SHA-256" in result["missingHash"]
    assert "requires backend=webgpu" in result["wasm"]
    assert "requires two exact-provider sessions" in result["noDevice"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_single_acceptance_protocol_is_exact_and_requires_posttraining_lineage():
    script = """
global.window = { __localAgentSkipInit: true };
const api = require(process.argv[1]);
function rejection(callback) {
  try {
    callback();
    return null;
  } catch (error) {
    return error.message;
  }
}
const provenance = {
  weights: { checkpoint_stage: "sft", checkpoint_sha256: "a".repeat(64) },
  tokenizer: { sha256: "c".repeat(64) },
  checkpoint_lineage: {
    stage: "sft",
    tokenizer_sha256: "c".repeat(64),
  },
  training_lineage_export: "training-lineage.json",
};
const lineage = {
  kind: "localagent_training_lineage_export",
  schema_version: 1,
  stage: "sft",
  checkpoint_sha256: "a".repeat(64),
  lineage: provenance.checkpoint_lineage,
  conversation_prompt_contract: "openai_full_catalog_v1",
  training_artifact_sha256: ["b".repeat(64)],
  training_artifacts: [{
    artifact_kind: "localagent_sft_conversation_artifact",
    bytes: 123,
    path: "/artifacts/sft.json",
    sha256: "b".repeat(64),
  }],
};
process.stdout.write(JSON.stringify({
  protocol: api.validateDecodeProtocolSettings(
    { outputTokens: 32, warmups: 3, repetitions: 30 }, true
  ),
  alteredOutputs: rejection(() => api.validateDecodeProtocolSettings(
    { outputTokens: 64, warmups: 3, repetitions: 30 }, true
  )),
  alteredWarmups: rejection(() => api.validateDecodeProtocolSettings(
    { outputTokens: 32, warmups: 4, repetitions: 30 }, true
  )),
  alteredRepetitions: rejection(() => api.validateDecodeProtocolSettings(
    { outputTokens: 32, warmups: 3, repetitions: 31 }, true
  )),
  exploratory: api.validateDecodeProtocolSettings(
    { outputTokens: 64, warmups: 4, repetitions: 31 }, false
  ),
  lineage: api.validateAcceptanceTrainingLineage(provenance, lineage, true),
  emptyLineage: rejection(() => api.validateAcceptanceTrainingLineage(
    provenance, { ...lineage, training_artifact_sha256: [], training_artifacts: [] }, true
  )),
  stageDrift: rejection(() => api.validateAcceptanceTrainingLineage(
    provenance, { ...lineage, stage: "pretrain" }, true
  )),
  pretrain: api.validateAcceptanceTrainingLineage({
    weights: { checkpoint_stage: "pretrain", checkpoint_sha256: "c".repeat(64) },
    training_lineage_export: null,
  }, null, true),
}));
"""
    result = _run_node(script)
    assert result["protocol"] == {"outputTokens": 32, "warmups": 3, "repetitions": 30}
    assert "exactly 32 output tokens" in result["alteredOutputs"]
    assert "exactly 32 output tokens" in result["alteredWarmups"]
    assert "exactly 32 output tokens" in result["alteredRepetitions"]
    assert result["exploratory"] == {"outputTokens": 64, "warmups": 4, "repetitions": 31}
    assert result["lineage"]["training_artifact_sha256"] == ["b" * 64]
    assert result["lineage"]["training_artifacts"][0]["path"] == "/artifacts/sft.json"
    assert "non-empty" in result["emptyLineage"]
    assert "file-identity/SHA-256 lineage" in result["stageDrift"]
    assert result["pretrain"] is None


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_decode_contract_rejects_mixed_modes_and_unbound_trained_claims():
    random_manifest = _manifest()
    trained_manifest = _trained_manifest()
    role = "hybrid_treatment"
    random_provenance = _provenance(
        role,
        random_manifest["models"][role]["name"],
    )
    trained_provenance = _trained_provenance(
        role,
        trained_manifest["models"][role]["name"],
        trained_manifest,
    )
    script = """
global.window = { __localAgentSkipInit: true };
const api = require(process.argv[1]);
const randomManifest = JSON.parse(process.argv[2]);
const trainedManifest = JSON.parse(process.argv[3]);
const randomProvenance = JSON.parse(process.argv[4]);
const trainedProvenance = JSON.parse(process.argv[5]);
function rejection(callback) {
  try {
    callback();
    return null;
  } catch (error) {
    return error.message;
  }
}
const ambiguousRandom = JSON.parse(JSON.stringify(randomManifest));
ambiguousRandom.checkpoints = trainedManifest.checkpoints;
ambiguousRandom.tokenizer = trainedManifest.tokenizer;
ambiguousRandom.quality_evaluation = trainedManifest.quality_evaluation;
const mixedTrained = { ...trainedManifest, latency_only: true };
const capabilityClaim = { ...trainedManifest, action_capability_claimed: true };
const badCheckpoint = JSON.parse(JSON.stringify(trainedProvenance));
badCheckpoint.weights.checkpoint_sha256 = "f".repeat(64);
const badTokenizer = JSON.parse(JSON.stringify(trainedProvenance));
badTokenizer.tokenizer.sha256 = "f".repeat(64);
badTokenizer.tokenizer.artifact_identity.sha256 = "f".repeat(64);
const randomReference = JSON.parse(JSON.stringify(trainedProvenance));
randomReference.parity.results.fp32.reference =
  "exact in-memory LocalAgentLM random initialization";
const actionClaim = { ...trainedProvenance, action_capability_claimed: true };
process.stdout.write(JSON.stringify({
  ambiguousRandom: rejection(() => api.validateDecodeManifest(ambiguousRandom)),
  mixedTrained: rejection(() => api.validateDecodeManifest(mixedTrained)),
  capabilityClaim: rejection(() => api.validateDecodeManifest(capabilityClaim)),
  randomUnderTrained: rejection(() => api.validateDecodeProvenance(
    randomProvenance, "hybrid_treatment",
    trainedManifest.models.hybrid_treatment, trainedManifest
  )),
  trainedUnderRandom: rejection(() => api.validateDecodeProvenance(
    trainedProvenance, "hybrid_treatment",
    randomManifest.models.hybrid_treatment, randomManifest
  )),
  badCheckpoint: rejection(() => api.validateDecodeProvenance(
    badCheckpoint, "hybrid_treatment",
    trainedManifest.models.hybrid_treatment, trainedManifest
  )),
  badTokenizer: rejection(() => api.validateDecodeProvenance(
    badTokenizer, "hybrid_treatment",
    trainedManifest.models.hybrid_treatment, trainedManifest
  )),
  randomReference: rejection(() => api.validateDecodeProvenance(
    randomReference, "hybrid_treatment",
    trainedManifest.models.hybrid_treatment, trainedManifest
  )),
  actionClaim: rejection(() => api.validateDecodeProvenance(
    actionClaim, "hybrid_treatment",
    trainedManifest.models.hybrid_treatment, trainedManifest
  )),
}));
"""
    result = _run_node(
        script,
        random_manifest,
        trained_manifest,
        random_provenance,
        trained_provenance,
    )
    assert "mixed or ambiguous" in result["ambiguousRandom"]
    assert "mixed or ambiguous" in result["mixedTrained"]
    assert "must be exactly false" in result["capabilityClaim"]
    assert "trained, non-capability" in result["randomUnderTrained"]
    assert "untrained latency-only" in result["trainedUnderRandom"]
    assert "checkpoint pin" in result["badCheckpoint"]
    assert "tokenizer provenance disagrees" in result["badTokenizer"]
    assert "trajectory parity" in result["randomReference"]
    assert "must be exactly false" in result["actionClaim"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_decode_output_locations_are_provider_explicit_and_cache_resident():
    script = """
global.window = { __localAgentSkipInit: true };
const api = require(process.argv[1]);
const names = ["present_0_key", "present_0_value", "present_1_conv"];
process.stdout.write(JSON.stringify({
  webgpu: api.decodeSessionOptions("webgpu", names),
  wasm: api.decodeSessionOptions("wasm", names),
}));
"""
    result = _run_node(script)
    assert result["webgpu"]["executionProviders"] == ["webgpu"]
    assert result["wasm"]["executionProviders"] == ["wasm"]
    assert result["webgpu"]["preferredOutputLocation"]["next_token"] == "cpu"
    assert result["wasm"]["preferredOutputLocation"]["next_token"] == "cpu"
    assert result["webgpu"]["preferredOutputLocation"]["logits"] == "cpu"
    assert result["wasm"]["preferredOutputLocation"]["logits"] == "cpu"
    for name in ("present_0_key", "present_0_value", "present_1_conv"):
        assert result["webgpu"]["preferredOutputLocation"][name] == "gpu-buffer"
        assert result["wasm"]["preferredOutputLocation"][name] == "cpu"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_decode_decision_consumes_logits_and_cross_checks_compatibility_token():
    script = """
global.window = { __localAgentSkipInit: true };
const api = require(process.argv[1]);
const accepted = api.validateDecodeDecisionOutputs({
  next_token: {
    type: "int64", dims: [1], data: BigInt64Array.of(2n), location: "cpu",
  },
  logits: {
    type: "float32", dims: [1, 4],
    data: Float32Array.from([-2, 1, 9, 3]), location: "cpu",
  },
}, "fixture", 4, "fp32");
let mismatch = null;
try {
  api.validateDecodeDecisionOutputs({
    next_token: {
      type: "int64", dims: [1], data: BigInt64Array.of(1n), location: "cpu",
    },
    logits: {
      type: "float32", dims: [1, 4],
      data: Float32Array.from([-2, 1, 9, 3]), location: "cpu",
    },
  }, "fixture", 4, "fp32");
} catch (error) {
  mismatch = error.message;
}
process.stdout.write(JSON.stringify({ accepted, mismatch }));
"""
    result = _run_node(script)
    assert result["accepted"] == 2
    assert "disagrees with logits argmax" in result["mismatch"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_acceptance_disposal_and_runtime_version_fail_closed():
    script = """
global.window = { __localAgentSkipInit: true };
global.ort = { version: "1.27.0" };
const api = require(process.argv[1]);
function rejection(callback) {
  try {
    callback();
    return null;
  } catch (error) {
    return error.message;
  }
}
const cacheTensorCount = 2;
const outputTokens = 32;
const decodePasses = 31;
const tracker = {
  cache_tensors_allocated: cacheTensorCount * outputTokens,
  next_token_tensors_allocated: outputTokens,
  logits_tensors_allocated: outputTokens,
  decode_input_tensors_allocated: decodePasses,
  cache_dispose_attempted: cacheTensorCount * outputTokens,
  cache_dispose_succeeded: cacheTensorCount * outputTokens,
  cache_dispose_failed: 0,
  cache_dispose_api_unavailable: 0,
  next_token_dispose_attempted: outputTokens,
  next_token_dispose_succeeded: outputTokens,
  next_token_dispose_failed: 0,
  next_token_dispose_api_unavailable: 0,
  logits_dispose_attempted: outputTokens,
  logits_dispose_succeeded: outputTokens,
  logits_dispose_failed: 0,
  logits_dispose_api_unavailable: 0,
  decode_input_dispose_attempted: decodePasses,
  decode_input_dispose_succeeded: decodePasses,
  decode_input_dispose_failed: 0,
  decode_input_dispose_api_unavailable: 0,
  superseded_cache_tensors_released: cacheTensorCount * decodePasses,
  final_cache_tensors_released: cacheTensorCount,
};
const accepted = api.validateAcceptanceDisposalRecord({
  run_ok: true,
  cache: { tensor_count: cacheTensorCount },
  allocation_disposal: tracker,
  disposal_contract_verified: null,
  error: null,
}, true);
const unavailable = {
  run_ok: true,
  cache: { tensor_count: cacheTensorCount },
  allocation_disposal: {
    ...tracker,
    cache_dispose_succeeded: tracker.cache_dispose_succeeded - 1,
    cache_dispose_api_unavailable: 1,
  },
  disposal_contract_verified: null,
  error: null,
};
const unavailableError = rejection(() =>
  api.validateAcceptanceDisposalRecord(unavailable, true)
);
const pinned = api.verifyOrtVersionPin();
global.ort = { env: { versions: {} } };
const unknownVersion = rejection(() => api.verifyOrtVersionPin());
process.stdout.write(JSON.stringify({
  disposalVerified: accepted.disposal_contract_verified,
  unavailableError,
  unavailableRunOk: unavailable.run_ok,
  pinned,
  unknownVersion,
}));
"""
    result = _run_node(script)
    assert result["disposalVerified"] is True
    assert "unavailable or incomplete" in result["unavailableError"]
    assert result["unavailableRunOk"] is False
    assert result["pinned"]["ort_version_verified"] is True
    assert "Could not verify ONNX Runtime Web version null" in result["unknownVersion"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_acceptance_requires_all_external_challenge_machine_and_acquisition_roots():
    script = """
global.window = {
  __localAgentSkipInit: true,
  __localAgentDecodeAcceptanceMode: true,
  location: { search: "" },
};
const api = require(process.argv[1]);
function rejection(callback) {
  try {
    callback();
    return null;
  } catch (error) {
    return error.message;
  }
}
const missing = rejection(() => api.requestedDecodeAcceptanceEvidence());
Object.assign(global.window, {
  __localAgentDecodeRunChallenge: "1".repeat(64),
  __localAgentDecodeMachineConditionSha256: "2".repeat(64),
  __localAgentDecodeHarnessHtmlSha256: "3".repeat(64),
  __localAgentDecodeHarnessJavascriptSha256: "4".repeat(64),
  __localAgentDecodeOrtJavascriptSha256: "5".repeat(64),
  __localAgentDecodeOrtWasmSha256: "6".repeat(64),
});
const evidence = api.requestedDecodeAcceptanceEvidence();
process.stdout.write(JSON.stringify({ missing, evidence }));
"""
    result = _run_node(script)
    assert "Acceptance requires external run, machine, and acquisition roots" in result["missing"]
    assert result["evidence"] == {
        "run_challenge": "1" * 64,
        "machine_condition_sha256": "2" * 64,
        "html_sha256": "3" * 64,
        "javascript_sha256": "4" * 64,
        "ort_javascript_sha256": "5" * 64,
        "ort_wasm_sha256": "6" * 64,
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
@pytest.mark.skipif(
    not all(path.is_file() for path in DEFAULT_V2_BUNDLE_ARTIFACTS),
    reason="requires the exact local sealed random cached-decode v2 bundle",
)
def test_default_v2_bundle_validates_as_explicit_legacy_next_token_abi():
    bundle = DEFAULT_V2_BUNDLE
    manifest_path = bundle / "matched-decode.json"
    manifest = json.loads(manifest_path.read_text())
    provenances = {
        role: json.loads((bundle / model["provenance"]).read_text())
        for role, model in manifest["models"].items()
    }
    for relative_path, pin in manifest["artifacts"].items():
        artifact = bundle / relative_path
        assert artifact.stat().st_size == pin["bytes"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == pin["sha256"]

    script = """
global.window = { __localAgentSkipInit: true };
const api = require(process.argv[1]);
const manifest = JSON.parse(process.argv[2]);
const provenances = JSON.parse(process.argv[3]);
const accepted = api.validateDecodeManifest(manifest);
const decisionAbis = {};
for (const [role, model] of Object.entries(accepted.models)) {
  const provenance = api.validateDecodeProvenance(
    provenances[role], role, model, accepted
  );
  decisionAbis[role] = api.decodeDecisionAbi(provenance);
}
process.stdout.write(JSON.stringify({ artifactType: accepted.artifact_type, decisionAbis }));
"""
    result = _run_node(script, manifest, provenances)
    assert result["artifactType"] == "matched_random_cached_decode_latency_suite"
    assert set(result["decisionAbis"].values()) == {"legacy_exported_next_token_only"}


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_decode_schedule_is_seeded_balanced_and_summary_keeps_failures():
    records = [
        {
            "arm_id": "hybrid",
            "input_tokens": 128,
            "run_ok": True,
            "ttft_ms": 10,
            "tpot_ms": 5,
            "decode_tokens_per_second": 200,
            "prefill_ms": 9,
            "decode_inference_ms": 100,
            "model_decode_tokens_per_second": 310,
            "graph_pass_counts": {"prefill": 1, "decode": 31},
            "cache": {"final_logical_bytes": 1000},
        },
        {
            "arm_id": "hybrid",
            "input_tokens": 128,
            "run_ok": False,
            "graph_pass_counts": {"prefill": 1, "decode": 2},
            "cache": {},
        },
    ]
    script = """
global.window = { __localAgentSkipInit: true };
const api = require(process.argv[1]);
const records = JSON.parse(process.argv[2]);
const first = api.buildDecodeSchedule(
  ["hybrid", "attention"], [128, 512, 1024, 1536], 30, "paper", "measured"
);
const again = api.buildDecodeSchedule(
  ["hybrid", "attention"], [128, 512, 1024, 1536], 30, "paper", "measured"
);
const counts = {};
for (const row of first) {
  const key = `${row.arm_id}:${row.input_tokens}`;
  counts[key] = (counts[key] || 0) + 1;
}
process.stdout.write(JSON.stringify({
  same: JSON.stringify(first) === JSON.stringify(again),
  length: first.length,
  counts,
  summary: api.summarizeDecodeRecords(records),
}));
"""
    result = _run_node(script, records)
    assert result["same"] is True
    assert result["length"] == 240
    assert set(result["counts"].values()) == {30}
    summary = result["summary"]
    assert summary["quality_metrics_included"] is False
    assert summary["attempted"] == 2
    assert summary["completed"] == 1
    assert summary["failed"] == 1
    assert summary["conditions"][0]["graph_pass_counts"] == {
        "prefill": 2,
        "decode": 33,
        "total": 35,
    }


def test_decode_page_exposes_fail_closed_cached_latency_protocol():
    html_payload = DECODE_HTML.read_bytes()
    script_payload = DECODE_JS.read_bytes()
    html = html_payload.decode()
    script = script_payload.decode()

    assert len(html_payload) == HARNESS_HTML_BYTES
    assert hashlib.sha256(html_payload).hexdigest() == HARNESS_HTML_SHA256
    assert len(script_payload) == HARNESS_JAVASCRIPT_BYTES
    assert hashlib.sha256(script_payload).hexdigest() == HARNESS_JAVASCRIPT_SHA256
    assert HARNESS_ORT_VERSION == "1.27.0"
    assert f"{HARNESS_ORT_VENDOR_PATH}/ort.webgpu.min.js" in html
    assert "vendor/onnxruntime-web-${DECODE_ORT_VERSION}/" in script
    assert "ort-wasm-simd-threaded.jsep.wasm" in script
    assert "cdn.jsdelivr.net" not in html
    assert "cdn.jsdelivr.net" not in script
    assert "UNTRAINED RANDOM WEIGHTS" in html
    assert "NO CAPABILITY OR QUALITY RESULT" in html
    assert "trained weights, latency only; quality scored separately" in html
    assert 'id="decode-artifact-heading"' in html
    assert 'id="decode-artifact-disclaimer"' in html
    assert "onnxruntime-web-1.27.0" in html
    assert 'id="decode-output-tokens"' in html
    assert 'value="32"' in html
    assert 'id="decode-warmups"' in html
    assert 'min="3"' in html
    assert 'id="decode-repetitions"' in html
    assert 'min="30"' in html
    assert "?backend=wasm" in html
    assert "mode=single" in html
    assert "acceptance=1" in html
    assert "acceptance_root_sha256" in html
    assert "run_challenge" in html
    assert "machine_condition_sha256" in html
    assert "harness_html_sha256" in html
    assert "ort_wasm_sha256" in html
    assert html.count('integrity="sha384-') == 2
    assert "crossorigin=\"anonymous\"" in html
    assert "attest browser" in html
    assert "provenance_sha256" in html
    assert "provenance_bytes" in html
    assert "window.__localAgentDecodeBenchmarkResult" in html
    assert 'id="decode-result-json"' in html
    assert "validated and reduced by" in html
    assert "argmax" in html
    assert "<code>next_token</code> is only a compatibility cross-check" in html
    assert "<code>legacy_exported_next_token_only</code>" in html
    assert "when the" in html
    assert "graph exposes it, <code>logits</code>" in html

    assert 'executionProviders: [requireExplicitProvider(provider)]' in script
    assert '["next_token", "cpu"]' in script
    assert '["logits", "cpu"]' in script
    assert '"gpu-buffer"' in script
    assert "preferredOutputLocation" in script
    assert "feeds[pastName] = tensor" in script
    cache_binding = script[
        script.index("function cacheFeedsFromPresent"):
        script.index("function emptyConditionRecord")
    ]
    assert ".data" not in cache_binding
    assert "getData" not in cache_binding
    assert "disposeCacheMap(currentCache, tracker, disposed, \"superseded\")" in script
    assert "disposeCacheMap(currentCache, tracker, disposed, \"final\")" in script
    assert "prefill_attempted" in script
    assert "decode_attempted" in script
    assert "const decodeWallStarted = firstTokenAvailable" in script
    assert "record.decode_wall_ms = lastTokenAvailableAt - decodeWallStarted" in script
    assert "record.generation_wall_ms = lastTokenAvailableAt - prefillStarted" in script
    assert "pass_started_offset_ms" in script
    assert "pass_resolved_offset_ms" in script
    assert "token_available_offset_ms" in script
    assert "cache_tensors: present.metadata" in script
    assert "record.tpot_ms = record.decode_wall_ms / decodePasses" in script
    assert "model_decode_tokens_per_second" in script
    assert "validateDecodeDecisionOutputs" in script
    assert '"validated_logits_argmax"' in script
    assert "per_node_fallback_status: \"unknown\"" in script
    assert "manifest_raw_text" in script
    assert "verification_before_parse_or_ort: true" in script
    assert "trained weights, latency only; quality scored separately" in script
    assert "single_trained_cached_decode_suite" in script
    assert "validateRequiredWebGpuEvidence" in script
    assert "DECODE_ACCEPTANCE_PROTOCOL" in script
    assert "validateDecodeProtocolSettings" in script
    assert "required_verification_passed" in script
    assert "prompt_lengths_tokens" in script
    assert 'reported_percentiles: ["p50", "p95"]' in script
    assert "browserRuntimeMetadata" in script
    assert "gpuRuntimeMetadata" in script
    assert "action_capability_claimed: false" in script
    assert "window.__localAgentDecodeBenchmarkResult = payload" in script
    assert "captureDecodeHarnessIdentity" in script
    assert "requestedDecodeAcceptanceEvidence" in script
    assert "self_hosted_same_origin" in script
    assert "DECODE_EVIDENCE_SCOPE" in script
    assert "benchmark_session_id" in script
    assert "run_id" in script
    assert "case_order_seed: DECODE_DEFAULT_SEED" in script
