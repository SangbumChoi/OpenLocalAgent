from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
WEB_DIR = ROOT / "demos" / "webgpu"
BACKBONE_JS = WEB_DIR / "backbone-benchmark.js"
BACKBONE_HTML = WEB_DIR / "backbone-benchmark.html"
SHA = "a" * 64


def _pair_manifest() -> dict:
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
        "artifact_type": "matched_random_backbone_latency_suite",
        "latency_only": True,
        "trained": False,
        "capability_artifact": False,
        "quality_claims": [],
        "shared_random_seed": 19,
        "controlled_fields": controlled,
        "intentional_differences": {
            "ffn_hidden": {},
            "layer_types": {},
            "name": {},
        },
        "artifacts": {
            "hybrid/provenance.json": {"bytes": 100, "sha256": SHA},
            "attention/provenance.json": {"bytes": 100, "sha256": SHA},
        },
        "models": {
            "hybrid_treatment": {
                "directory": "hybrid",
                "name": "webgpu-35m-hybrid",
                "provenance": "hybrid/provenance.json",
            },
            "all_attention_control": {
                "directory": "attention",
                "name": "webgpu-35m-attn",
                "provenance": "attention/provenance.json",
            },
        },
    }


def _provenance(role: str, name: str) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "random_weight_hidden_backbone_onnx",
        "latency_only": True,
        "trained": False,
        "training_steps": 0,
        "capability_artifact": False,
        "quality_claims": [],
        "graph_contract": {
            "input": {"name": "input_ids", "dtype": "int64"},
            "output": {"name": "hidden", "dtype": "float32"},
            "tokenizer_asset_included": False,
        },
        "model": {
            "name": name,
            "pair_role": role,
            "config": {"name": name},
            "config_canonical_sha256": SHA,
            "full_model_parameters": 34_000_000,
        },
        "weights": {
            "source": "deterministic_random_initialization",
            "checkpoint": None,
            "seed": 19,
            "state_dict_sha256": SHA,
        },
        "artifacts": {
            "backbone.fp16.onnx": {
                "file": "backbone.fp16.onnx",
                "bytes": 100,
                "sha256": SHA,
            }
        },
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_backbone_pair_manifest_requires_exporter_latency_only_contract():
    script = """
global.window = { __localAgentSkipInit: true };
const { validateBackboneManifest } = require(process.argv[1]);
const manifest = JSON.parse(process.argv[2]);
const accepted = validateBackboneManifest(manifest);
let rejected = null;
try {
  validateBackboneManifest({ ...manifest, trained: true });
} catch (error) {
  rejected = error.message;
}
process.stdout.write(JSON.stringify({
  artifactType: accepted.artifact_type,
  models: Object.keys(accepted.models).sort(),
  rejected,
}));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(BACKBONE_JS), json.dumps(_pair_manifest())],
        check=True,
        capture_output=True,
        text=True,
    )
    contract = json.loads(result.stdout)
    assert contract["artifactType"] == "matched_random_backbone_latency_suite"
    assert contract["models"] == ["all_attention_control", "hybrid_treatment"]
    assert "latency-only" in contract["rejected"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_backbone_provenance_requires_random_hidden_graph_without_tokenizer_or_quality():
    role = "hybrid_treatment"
    model = _pair_manifest()["models"][role]
    provenance = _provenance(role, model["name"])
    script = """
global.window = { __localAgentSkipInit: true };
const { validateBackboneProvenance } = require(process.argv[1]);
const provenance = JSON.parse(process.argv[2]);
const model = JSON.parse(process.argv[3]);
const accepted = validateBackboneProvenance(
  provenance, "hybrid_treatment", model, 19
);
let rejected = null;
try {
  validateBackboneProvenance({
    ...provenance,
    graph_contract: { ...provenance.graph_contract, tokenizer_asset_included: true },
  }, "hybrid_treatment", model, 19);
} catch (error) {
  rejected = error.message;
}
process.stdout.write(JSON.stringify({
  tokenizerIncluded: accepted.graph_contract.tokenizer_asset_included,
  qualityClaims: accepted.quality_claims,
  rejected,
}));
"""
    result = subprocess.run(
        [
            shutil.which("node"),
            "-e",
            script,
            str(BACKBONE_JS),
            json.dumps(provenance),
            json.dumps(model),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    contract = json.loads(result.stdout)
    assert contract["tokenizerIncluded"] is False
    assert contract["qualityClaims"] == []
    assert "pre-tokenized hidden-only" in contract["rejected"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_backbone_schedule_is_seeded_randomized_and_balanced():
    script = """
global.window = { __localAgentSkipInit: true };
const { buildBackboneSchedule } = require(process.argv[1]);
const arms = ["hybrid", "attention"];
const lengths = [128, 512, 1024, 1536];
const build = (seed) => buildBackboneSchedule(arms, lengths, 30, seed, "measured");
const firstConditions = buildBackboneSchedule(
  arms, lengths, 1, "paper-seed", "first_inference"
);
const first = build("paper-seed");
const again = build("paper-seed");
const other = build("other-seed");
const counts = {};
for (const row of first) {
  const key = `${row.arm_id}:${row.input_tokens}`;
  counts[key] = (counts[key] || 0) + 1;
}
process.stdout.write(JSON.stringify({
  first,
  same: JSON.stringify(first) === JSON.stringify(again),
  different: JSON.stringify(first) !== JSON.stringify(other),
  counts,
  firstConditions,
}));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(BACKBONE_JS)],
        check=True,
        capture_output=True,
        text=True,
    )
    schedule = json.loads(result.stdout)
    assert schedule["same"] is True
    assert schedule["different"] is True
    assert len(schedule["first"]) == 2 * 4 * 30
    assert set(schedule["counts"].values()) == {30}
    assert set(schedule["counts"]) == {
        f"{arm}:{length}"
        for arm in ("hybrid", "attention")
        for length in (128, 512, 1024, 1536)
    }
    assert len(schedule["firstConditions"]) == 8
    assert {
        (row["arm_id"], row["input_tokens"]) for row in schedule["firstConditions"]
    } == {
        (arm, length)
        for arm in ("hybrid", "attention")
        for length in (128, 512, 1024, 1536)
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_backbone_summary_is_latency_only_and_keeps_failed_attempts():
    records = [
        {"arm_id": "hybrid", "input_tokens": 128, "run_ok": True, "inference_ms": 4},
        {"arm_id": "hybrid", "input_tokens": 128, "run_ok": True, "inference_ms": 6},
        {"arm_id": "hybrid", "input_tokens": 128, "run_ok": False, "inference_ms": 7},
    ]
    script = """
global.window = { __localAgentSkipInit: true };
const { summarizeBackboneRecords } = require(process.argv[1]);
process.stdout.write(JSON.stringify(summarizeBackboneRecords(JSON.parse(process.argv[2]))));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(BACKBONE_JS), json.dumps(records)],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)
    assert summary["estimand"] == "hidden_only_backbone_forward_latency"
    assert summary["quality_metrics_included"] is False
    assert summary["attempted"] == 3
    assert summary["completed"] == 2
    assert summary["failed"] == 1
    condition = summary["conditions"][0]
    assert condition["inference_latency_ms"]["count"] == 2
    assert condition["inference_latency_ms"]["p50"] == 5
    assert not any("accuracy" in key or "success" in key for key in summary)


def test_backbone_page_exposes_reproducible_cold_warm_raw_protocol():
    html = BACKBONE_HTML.read_text()
    script = BACKBONE_JS.read_text()

    assert "UNTRAINED" not in html  # prose is readable, not a fabricated metric banner
    assert "untrained random weights" in html.lower()
    assert "latency-only" in html.lower()
    assert "no tokenizer asset" in html
    assert 'id="backbone-warmups"' in html
    assert 'min="3"' in html
    assert 'value="3"' in html
    assert 'id="backbone-repetitions"' in html
    assert 'min="30"' in html
    assert 'value="30"' in html
    assert "?backend=wasm" in html
    assert "onnxruntime-web@1.27.0" in html
    assert "window.__localAgentBackboneBenchmarkResult" in html
    assert 'id="backbone-result-json"' in html

    assert "new URLSearchParams(window.location.search).get(\"manifest\")" in script
    assert "executionProviders: [provider]" in script
    assert 'outputNames[0] !== "hidden"' in script
    assert "latency harness accepts hidden-only graphs and rejects logits" in script
    assert "ids[i]=(131*i+17) mod vocab_size" in script
    assert 'input_semantics: "deterministic_pretokenized_ids"' in script
    assert "first_for_condition = true" in script
    assert "first_ever_for_graph" in script
    assert "hidden?.dispose?.()" in script
    assert "first_inference_records" in script
    assert "warmup_records" in script
    assert "bundle_records" in script
    assert "session_records" in script
    assert "whole_session_provider_retry: false" in script
    assert "per_node_placement_verified: false" in script
    assert 'per_node_fallback_status: "unknown"' in script
    assert "browser_cache_state: \"unknown\"" in script
    assert "unknown_no_external_expected_digest" in script
    assert "ort_version_verified" in script
    assert "graph_sha256" in script
    assert "config_sha256" in script
    assert "window.__localAgentBackboneBenchmarkResult = payload" in script
    assert "resultNode.textContent = JSON.stringify(payload)" in script
