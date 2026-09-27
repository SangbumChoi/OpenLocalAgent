from __future__ import annotations

import copy
import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import localagent.eval.webgpu_decode_campaign as campaign_module
from localagent.data.conversation_artifact import canonical_json_bytes
from localagent.eval.webgpu_decode_campaign import (
    build_webgpu_decode_campaign,
    verify_webgpu_decode_campaign_against_artifacts,
    verify_webgpu_decode_campaign_integrity_bytes,
    write_webgpu_decode_campaign,
)
from localagent.eval.webgpu_decode_receipt import (
    ACCEPTANCE_BENCHMARK,
    ACCEPTANCE_CONTEXT_LENGTHS,
    ACCEPTANCE_DECISION_ABI,
    ACCEPTANCE_OUTPUT_TOKENS,
    ACCEPTANCE_PROTOCOL_ID,
    ACCEPTANCE_REPETITIONS,
    ACCEPTANCE_SEED,
    ACCEPTANCE_SESSION_ORDER_SEED,
    ACCEPTANCE_WARMUPS,
    EVIDENCE_SCOPE,
    HARNESS_HTML_BYTES,
    HARNESS_HTML_FILE,
    HARNESS_HTML_SHA256,
    HARNESS_JAVASCRIPT_BYTES,
    HARNESS_JAVASCRIPT_FILE,
    HARNESS_JAVASCRIPT_SHA256,
    HARNESS_ORT_JAVASCRIPT_FILE,
    HARNESS_ORT_JAVASCRIPT_SHA256,
    HARNESS_ORT_WASM_FILE,
    HARNESS_ORT_VENDOR_PATH,
    HARNESS_ORT_VERSION,
    build_webgpu_decode_receipt,
    verify_webgpu_decode_receipt_bytes,
    write_webgpu_decode_receipt,
)


ROOT = Path(__file__).parents[1]
CHECKPOINT_SHA256 = "a" * 64
TOKENIZER_SHA256 = "b" * 64
PREFILL_SHA256 = "d" * 64
DECODE_SHA256 = "e" * 64
TRAINING_SHA256 = "f" * 64
ARM_ID = "accepted-webgpu-1m"
PAIR_ROLE = "accepted_checkpoint"
PREFILL_BYTES = 12_345
DECODE_BYTES = 23_456
CACHE_TENSORS = 2
BENCHMARK_SESSION_ID = "11111111-1111-4111-8111-111111111111"
RUN_ID = "22222222-2222-4222-8222-222222222222"
RUN_CHALLENGE = "3" * 64
MACHINE_CONDITION_SHA256 = "4" * 64
ORT_JAVASCRIPT_SHA256 = HARNESS_ORT_JAVASCRIPT_SHA256
ORT_WASM_SHA256 = "6" * 64
ORT_JAVASCRIPT_URL = (
    f"https://fixture.invalid/{HARNESS_ORT_VENDOR_PATH}/{HARNESS_ORT_JAVASCRIPT_FILE}"
)
ORT_WASM_URL = f"https://fixture.invalid/{HARNESS_ORT_VENDOR_PATH}/{HARNESS_ORT_WASM_FILE}"
TRAINED_LABEL = "trained weights, latency only; quality scored separately"
TRAINED_LABELS = {
    "latency_only": True,
    "untrained_random_weights": False,
    "trained_weights": True,
    "capability_artifact": False,
    "action_capability_claimed": False,
    "action_capability_evaluation": False,
    "quality_evaluation": False,
    "quality_scored_separately": True,
    "artifact_manifest_latency_only": False,
    "benchmark_label": TRAINED_LABEL,
}


def _config() -> dict[str, Any]:
    return {
        "vocab_size": 256,
        "d_model": 16,
        "n_layers": 1,
        "n_loops": 1,
        "n_heads": 2,
        "n_kv_heads": 1,
        "conv_kernel": 3,
        "layer_types": ["attn"],
    }


def _raw_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _identity(raw_text: str) -> tuple[int, str]:
    payload = raw_text.encode()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _input_fixture(context: int) -> dict:
    token_ids = [(131 * index + 17) % 256 for index in range(context)]
    encoded = b"".join(struct.pack("<q", token_id) for token_id in token_ids)
    return {
        "input_tokens": context,
        "actual_tensor_tokens": context,
        "token_ids": token_ids,
        "input_ids_int64_sha256": hashlib.sha256(encoded).hexdigest(),
        "tensor_dtype": "int64",
        "tensor_dims": [1, context],
        "input_semantics": "deterministic_pretokenized_ids",
        "fixture_contract": "ids[i]=(131*i+17) mod vocab_size",
        "vocab_size": 256,
        "tokenizer_asset": None,
    }


def _allocation_disposal() -> dict[str, int]:
    return {
        "cache_tensors_allocated": CACHE_TENSORS * ACCEPTANCE_OUTPUT_TOKENS,
        "next_token_tensors_allocated": ACCEPTANCE_OUTPUT_TOKENS,
        "logits_tensors_allocated": ACCEPTANCE_OUTPUT_TOKENS,
        "decode_input_tensors_allocated": ACCEPTANCE_OUTPUT_TOKENS - 1,
        "cache_dispose_attempted": CACHE_TENSORS * ACCEPTANCE_OUTPUT_TOKENS,
        "cache_dispose_succeeded": CACHE_TENSORS * ACCEPTANCE_OUTPUT_TOKENS,
        "cache_dispose_failed": 0,
        "cache_dispose_api_unavailable": 0,
        "next_token_dispose_attempted": ACCEPTANCE_OUTPUT_TOKENS,
        "next_token_dispose_succeeded": ACCEPTANCE_OUTPUT_TOKENS,
        "next_token_dispose_failed": 0,
        "next_token_dispose_api_unavailable": 0,
        "logits_dispose_attempted": ACCEPTANCE_OUTPUT_TOKENS,
        "logits_dispose_succeeded": ACCEPTANCE_OUTPUT_TOKENS,
        "logits_dispose_failed": 0,
        "logits_dispose_api_unavailable": 0,
        "decode_input_dispose_attempted": ACCEPTANCE_OUTPUT_TOKENS - 1,
        "decode_input_dispose_succeeded": ACCEPTANCE_OUTPUT_TOKENS - 1,
        "decode_input_dispose_failed": 0,
        "decode_input_dispose_api_unavailable": 0,
        "superseded_cache_tensors_released": CACHE_TENSORS
        * (ACCEPTANCE_OUTPUT_TOKENS - 1),
        "final_cache_tensors_released": CACHE_TENSORS,
    }


def _cache_tensors(sequence: int) -> list[dict]:
    per_tensor_bytes = sequence * 16
    return [
        {
            "name": "present_0_key",
            "dtype": "float16",
            "dims": [1, 1, sequence, 8],
            "logical_bytes": per_tensor_bytes,
            "reported_location": "gpu-buffer",
        },
        {
            "name": "present_0_value",
            "dtype": "float16",
            "dims": [1, 1, sequence, 8],
            "logical_bytes": per_tensor_bytes,
            "reported_location": "gpu-buffer",
        },
    ]


def _seeded_context_order(phase: str, repetition: int) -> list[int]:
    state = 2_166_136_261
    seed = f"{ACCEPTANCE_SEED}:{phase}:{repetition}"
    for character in seed:
        state ^= ord(character)
        state = (state * 16_777_619) & 0xFFFF_FFFF
    result = list(ACCEPTANCE_CONTEXT_LENGTHS)
    for index in range(len(result) - 1, 0, -1):
        state = (state + 0x6D2B79F5) & 0xFFFF_FFFF
        value = state
        value = ((value ^ (value >> 15)) * (value | 1)) & 0xFFFF_FFFF
        mixed = ((value ^ (value >> 7)) * (value | 61)) & 0xFFFF_FFFF
        value = (value ^ ((value + mixed) & 0xFFFF_FFFF)) & 0xFFFF_FFFF
        random_value = ((value ^ (value >> 14)) & 0xFFFF_FFFF) / 4_294_967_296
        replacement = int(random_value * (index + 1))
        result[index], result[replacement] = result[replacement], result[index]
    return result


def _record(
    phase: str,
    repetition: int,
    context: int,
    global_index: int,
    order_index: int,
) -> dict:
    generated = [(context + repetition + index) % 256 for index in range(ACCEPTANCE_OUTPUT_TOKENS)]
    prefill_cache_bytes = context * 32
    decode_records = []
    inference_values = []
    previous_bytes = prefill_cache_bytes
    previous_token_offset = 0.0
    for index in range(ACCEPTANCE_OUTPUT_TOKENS - 1):
        inference_ms = 0.25 + index / 1000
        inference_values.append(inference_ms)
        pass_started_offset = previous_token_offset + 0.01
        pass_resolved_offset = pass_started_offset + inference_ms
        token_available_offset = pass_resolved_offset + 0.01
        after_bytes = previous_bytes + 32
        decode_records.append(
            {
                "pass_index": index,
                "input_token_id": generated[index],
                "output_token_id": generated[index + 1],
                "input_tokens": 1,
                "output_tokens": 1,
                "attention_cache_sequence_length": context + index + 1,
                "inference_ms": inference_ms,
                "token_available_ms": inference_ms + 0.01,
                "pass_started_offset_ms": pass_started_offset,
                "pass_resolved_offset_ms": pass_resolved_offset,
                "token_available_offset_ms": token_available_offset,
                "cache_logical_bytes_before": previous_bytes,
                "cache_logical_bytes_after": after_bytes,
                "cache_tensor_count": CACHE_TENSORS,
                "cache_tensors": _cache_tensors(context + index + 1),
                "cache_reported_locations": ["gpu-buffer"],
                "cache_bound_directly_without_readback": True,
            }
        )
        previous_bytes = after_bytes
        previous_token_offset = token_available_offset
    decode_inference_ms = sum(inference_values)
    decode_wall_ms = previous_token_offset
    tpot_ms = decode_wall_ms / (ACCEPTANCE_OUTPUT_TOKENS - 1)
    ttft_ms = 2.0 + context / 1000
    return {
        **TRAINED_LABELS,
        "phase": phase,
        "benchmark_session_id": BENCHMARK_SESSION_ID,
        "run_id": RUN_ID,
        "run_challenge": RUN_CHALLENGE,
        "global_order_index": global_index,
        "repetition": repetition,
        "order_index": order_index,
        "arm_id": ARM_ID,
        "pair_role": PAIR_ROLE,
        "input_tokens": context,
        "prompt_tokens_requested": context,
        "actual_input_tokens": context,
        "prompt_tokens_actual": context,
        "output_tokens_requested": ACCEPTANCE_OUTPUT_TOKENS,
        "actual_output_tokens": ACCEPTANCE_OUTPUT_TOKENS,
        "generated_token_ids": generated,
        "generated_token_interpretation": (
            "trained-weight logits argmax IDs used only to drive cached graph passes; not "
            "decoded, quality-scored, or interpreted as actions"
        ),
        "decision_output_abi": ACCEPTANCE_DECISION_ABI,
        "graph_pass_counts": {
            "prefill": 1,
            "decode": 31,
            "prefill_attempted": 1,
            "decode_attempted": 31,
            "total": 32,
            "total_attempted": 32,
            "expected_prefill": 1,
            "expected_decode": 31,
            "expected_total": 32,
        },
        "actual_graph_input_token_positions": context + 31,
        "graph_files": {
            "prefill": "prefill.fp16.onnx",
            "decode": "decode.fp16.onnx",
        },
        "graph_sha256": {
            "prefill": PREFILL_SHA256,
            "decode": DECODE_SHA256,
        },
        "graph_bytes": {
            "prefill": PREFILL_BYTES,
            "decode": DECODE_BYTES,
        },
        "decode_pass_records": decode_records,
        "cache": {
            "enabled": True,
            "dtype": "float16",
            "requested_residency": "gpu-buffer",
            "next_token_residency": "cpu",
            "logits_residency": "cpu",
            "token_selection_source": "validated_logits_argmax",
            "next_token_role": "compatibility_cross_check",
            "cache_data_read_to_javascript": False,
            "update_strategy": (
                "present_outputs_rebound_directly_as_past_inputs_without_cpu_materialization"
            ),
            "tensor_count": CACHE_TENSORS,
            "slot_count": 1,
            "slots": [
                {
                    "slot": 0,
                    "loop": 0,
                    "layer": 0,
                    "kind": "attn",
                    "past_inputs": ["past_0_key", "past_0_value"],
                    "present_outputs": ["present_0_key", "present_0_value"],
                    "shape": ["batch", 1, "cache_sequence", 8],
                    "dtype_by_precision": {
                        "fp16": "float16",
                        "fp32": "float32",
                    },
                    "update": "append_one_token_along_axis_2",
                }
            ],
            "prefill_tensors": _cache_tensors(context),
            "prefill_logical_bytes": prefill_cache_bytes,
            "final_tensors": _cache_tensors(context + ACCEPTANCE_OUTPUT_TOKENS - 1),
            "final_logical_bytes": previous_bytes,
        },
        "allocation_disposal": _allocation_disposal(),
        "disposal_contract_verified": True,
        "provider_requested": "webgpu",
        "provider_actual": None,
        "provider_actual_observation": (
            "not exposed; exact provider request/session creation and cache tensor locations "
            "recorded"
        ),
        "graph_wide_provider_verified": False,
        "whole_session_provider_retry": False,
        "per_node_placement_verified": False,
        "per_node_fallback_status": "unknown",
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "decode_tokens_per_second": 1000.0 / tpot_ms,
        "prefill_ms": 1.5 + context / 1000,
        "decode_inference_ms": decode_inference_ms,
        "decode_wall_ms": decode_wall_ms,
        "generation_wall_ms": ttft_ms + decode_wall_ms,
        "model_decode_tokens_per_second": 31_000.0 / decode_inference_ms,
        "run_ok": True,
        "error": None,
    }


def _records(phase: str, repetitions: int) -> list[dict]:
    output: list[dict] = []
    for repetition in range(repetitions):
        for order_index, context in enumerate(_seeded_context_order(phase, repetition)):
            output.append(
                _record(
                    phase,
                    repetition,
                    context,
                    len(output),
                    order_index,
                )
            )
    return output


def _verified_artifact_record(
    kind: str,
    path: str,
    size: int,
    sha256: str,
) -> dict:
    return {
        "artifact_kind": kind,
        "relative_path": path,
        "bytes": size,
        "actual_sha256": sha256,
        "expected_bytes": size,
        "expected_sha256": sha256,
        "bytes_verified": True,
        "hash_verified": True,
        "verification_before_parse_or_ort": True,
    }


def _provider_evidence() -> dict:
    adapter_info = {
        "vendor": "fixture-vendor",
        "architecture": "fixture-architecture",
        "device": "fixture-device",
        "description": "fixture",
        "is_fallback_adapter": False,
    }
    return {
        "provider_requested": "webgpu",
        "provider_actual": None,
        "provider_actual_observation": (
            "not exposed by ONNX Runtime Web; exact provider request and session creation observed"
        ),
        "provider_actual_scope": (
            "provider request plus session creation only; graph-wide and per-node placement "
            "are unknown"
        ),
        "exact_provider_request_and_session_creation_observed": True,
        "execution_provider_list": ["webgpu"],
        "whole_session_provider_retry": False,
        "per_node_placement_verified": False,
        "graph_wide_provider_verified": False,
        "per_node_placement_status": "unknown",
        "per_node_fallback_status": "unknown",
        "required_for_single_model": True,
        "required_verification_passed": True,
        "cache_output_location_verification_required": True,
        "verification_method": (
            "two sessions created from executionProviders=['webgpu']; ORT exposed a GPUDevice; "
            "cache tensors must report gpu-buffer; graph-wide/per-node placement remains unknown"
        ),
        "ort_webgpu": {
            "ort_adapter_available": True,
            "ort_device_available": True,
            "adapter_info": adapter_info,
        },
    }


def _result() -> dict:
    training_identity = {
        "artifact_kind": "localagent_sft_conversation_artifact",
        "bytes": 991,
        "path": "/artifacts/train.sft.json",
        "sha256": TRAINING_SHA256,
    }
    lineage_core = {
        "stage": "sft",
        "tokenizer_sha256": TOKENIZER_SHA256,
    }
    lineage = {
        "kind": "localagent_training_lineage_export",
        "schema_version": 1,
        "stage": "sft",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "lineage": lineage_core,
        "conversation_prompt_contract": "openai_full_catalog_v1",
        "training_artifact_sha256": [TRAINING_SHA256],
        "training_artifacts": [training_identity],
    }
    lineage_raw_text = _raw_text(lineage)
    lineage_bytes, lineage_sha256 = _identity(lineage_raw_text)
    provenance = {
        "schema_version": 1,
        "artifact_type": "trained_checkpoint_cached_decode_onnx",
        "trained": True,
        "latency_only": False,
        "capability_artifact": False,
        "capability_metrics": None,
        "quality_claims": [],
        "training_lineage_export": "training-lineage.json",
        "checkpoint_lineage": lineage_core,
        "weights": {
            "checkpoint_stage": "sft",
            "checkpoint_sha256": CHECKPOINT_SHA256,
        },
        "model": {
            "name": ARM_ID,
            "pair_role": PAIR_ROLE,
            "config": _config(),
        },
        "tokenizer": {"sha256": TOKENIZER_SHA256},
        "graph_contract": {
            "cache_slots": [
                {
                    "slot": 0,
                    "loop": 0,
                    "layer": 0,
                    "kind": "attn",
                    "past_inputs": ["past_0_key", "past_0_value"],
                    "present_outputs": ["present_0_key", "present_0_value"],
                    "shape": ["batch", 1, "cache_sequence", 8],
                    "dtype_by_precision": {
                        "fp16": "float16",
                        "fp32": "float32",
                    },
                    "update": "append_one_token_along_axis_2",
                }
            ],
            "graphs": {
                "fp16": {
                    "prefill": {
                        "file": "prefill.fp16.onnx",
                        "output_names": [
                            "next_token",
                            "logits",
                            "present_0_key",
                            "present_0_value",
                        ],
                    },
                    "decode": {
                        "file": "decode.fp16.onnx",
                        "output_names": [
                            "next_token",
                            "logits",
                            "present_0_key",
                            "present_0_value",
                        ],
                    },
                }
            },
            "next_token": {
                "name": "next_token",
                "dtype": "int64",
                "shape": ["batch"],
                "decode": "compatibility argmax over the exported final-token logits",
            },
            "logits": {
                "name": "logits",
                "shape": ["batch", 256],
            },
        },
        "artifacts": {
            "prefill.fp16.onnx": {
                "file": "prefill.fp16.onnx",
                "bytes": PREFILL_BYTES,
                "sha256": PREFILL_SHA256,
            },
            "decode.fp16.onnx": {
                "file": "decode.fp16.onnx",
                "bytes": DECODE_BYTES,
                "sha256": DECODE_SHA256,
            },
            "training-lineage.json": {
                "file": "training-lineage.json",
                "bytes": lineage_bytes,
                "sha256": lineage_sha256,
            },
        },
        "parity": {
            "hard_gate": True,
            "results": {
                "fp16": {
                    "hard_gate": True,
                    "passed": True,
                    "greedy_next_token_exact": True,
                    "final_token_logits_shape": ["batch", 256],
                    "reference": "exact in-memory LocalAgentLM checkpoint weights",
                    "artifacts": {
                        "prefill": {
                            "bytes": PREFILL_BYTES,
                            "sha256": PREFILL_SHA256,
                        },
                        "decode": {
                            "bytes": DECODE_BYTES,
                            "sha256": DECODE_SHA256,
                        },
                    },
                }
            },
        },
    }
    provenance_raw_text = _raw_text(provenance)
    provenance_bytes, provenance_sha256 = _identity(provenance_raw_text)
    manifest = {
        "schema_version": 1,
        "artifact_type": "single_trained_cached_decode_suite",
        "trained": True,
        "latency_only": False,
        "capability_artifact": False,
        "quality_claims": [],
        "quality_evaluation": {"included": False, "required_separately": True},
        "model": {
            "name": ARM_ID,
            "pair_role": PAIR_ROLE,
            "provenance": "provenance.json",
        },
        "artifacts": {
            "provenance.json": {
                "bytes": provenance_bytes,
                "sha256": provenance_sha256,
            }
        },
    }
    manifest_raw_text = _raw_text(manifest)
    manifest_bytes, manifest_sha256 = _identity(manifest_raw_text)
    arm = {
        "id": ARM_ID,
        "pair_role": PAIR_ROLE,
        "precision": "fp16",
        "decision_output_abi": ACCEPTANCE_DECISION_ABI,
        "provenance_file": "provenance.json",
        "provenance_sha256": provenance_sha256,
        "provenance_bytes": provenance_bytes,
        "provenance_raw_text": provenance_raw_text,
        "prefill_file": "prefill.fp16.onnx",
        "prefill_sha256": PREFILL_SHA256,
        "prefill_bytes": PREFILL_BYTES,
        "decode_file": "decode.fp16.onnx",
        "decode_sha256": DECODE_SHA256,
        "decode_bytes": DECODE_BYTES,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "config": _config(),
        "provenance": provenance,
        "training_lineage": lineage,
        "training_lineage_raw_text": lineage_raw_text,
    }
    provider = _provider_evidence()
    sessions = []
    for order_index, (graph_kind, graph_sha256, graph_bytes) in enumerate(
        (
            ("decode", DECODE_SHA256, DECODE_BYTES),
            ("prefill", PREFILL_SHA256, PREFILL_BYTES),
        )
    ):
        sessions.append(
            {
            **TRAINED_LABELS,
            "phase": "session_create",
            "benchmark_session_id": BENCHMARK_SESSION_ID,
            "run_challenge": RUN_CHALLENGE,
            "order_index": order_index,
            "arm_id": ARM_ID,
            "graph_kind": graph_kind,
            "graph_sha256": graph_sha256,
            "graph_bytes": graph_bytes,
            "session_create_ms": 5.0 + order_index,
            "preferred_output_location": {
                "next_token": "cpu",
                "logits": "cpu",
                "present_0_key": "gpu-buffer",
                "present_0_value": "gpu-buffer",
            },
            "input_names": (
                ["input_ids"]
                if graph_kind == "prefill"
                else ["input_ids", "past_0_key", "past_0_value"]
            ),
            "output_names": [
                "next_token",
                "logits",
                "present_0_key",
                "present_0_value",
            ],
            "provider_requested": "webgpu",
            "provider_actual": None,
            "provider_actual_observation": (
                "not exposed by ONNX Runtime Web; exact provider request and session creation "
                "observed"
            ),
            "provider_actual_scope": (
                "provider request plus session creation only; graph-wide and per-node placement "
                "are unknown"
            ),
            "execution_provider_list": ["webgpu"],
            "exact_provider_request_and_session_creation_observed": True,
            "graph_wide_provider_verified": False,
            "per_node_placement_verified": False,
            "per_node_placement_status": "unknown",
            "per_node_fallback_status": "unknown",
            "whole_session_provider_retry": False,
            "cache_residency_requested": "gpu-buffer",
            "next_token_residency_requested": "cpu",
            "logits_residency_requested": "cpu",
            "ort_webgpu": {
                "ort_adapter_available": True,
                "ort_device_available": True,
                "adapter_info": provider["ort_webgpu"]["adapter_info"],
            },
            "error": None,
            }
        )
    warmups = _records("warmup", ACCEPTANCE_WARMUPS)
    measured = _records("measured", ACCEPTANCE_REPETITIONS)
    artifacts = [
        {
            "artifact_kind": "single_decode_manifest",
            "relative_path": "single-decode.json",
            "bytes": manifest_bytes,
            "actual_sha256": manifest_sha256,
            "expected_bytes": None,
            "expected_sha256": manifest_sha256,
            "bytes_verified": None,
            "hash_verified": True,
            "hash_verification_status": "verified_by_external_acceptance_root",
            "verification_before_parse_or_ort": True,
        },
        _verified_artifact_record(
            "model_provenance",
            "provenance.json",
            provenance_bytes,
            provenance_sha256,
        ),
        _verified_artifact_record(
            "cached_prefill_onnx_graph",
            "prefill.fp16.onnx",
            PREFILL_BYTES,
            PREFILL_SHA256,
        ),
        _verified_artifact_record(
            "cached_decode_onnx_graph",
            "decode.fp16.onnx",
            DECODE_BYTES,
            DECODE_SHA256,
        ),
        _verified_artifact_record(
            "training_lineage",
            "training-lineage.json",
            lineage_bytes,
            lineage_sha256,
        ),
    ]
    return {
        **TRAINED_LABELS,
        "schema_version": 1,
        "benchmark": ACCEPTANCE_BENCHMARK,
        "status": "complete",
        "created_at": "2026-07-29T12:00:00.000Z",
        "warning": TRAINED_LABEL,
        "metadata": {
            "protocol_version": "cached-decode-latency-0.2",
            "benchmark_mode": "single",
            "artifact_mode": "trained",
            "benchmark_label": TRAINED_LABEL,
            "acceptance_mode": True,
            "acceptance_wrapper_manifest_sha256": manifest_sha256,
            "acceptance_protocol": {
                "id": ACCEPTANCE_PROTOCOL_ID,
                "context_lengths": list(ACCEPTANCE_CONTEXT_LENGTHS),
                "output_tokens_per_condition": ACCEPTANCE_OUTPUT_TOKENS,
                "warmups_per_condition": ACCEPTANCE_WARMUPS,
                "measured_repetitions_per_condition": ACCEPTANCE_REPETITIONS,
                "case_order_seed": ACCEPTANCE_SEED,
                "exact": True,
            },
            "benchmark_session_id": BENCHMARK_SESSION_ID,
            "run_id": RUN_ID,
            "run_challenge": RUN_CHALLENGE,
            "external_machine_condition_sha256": MACHINE_CONDITION_SHA256,
            "evidence_scope": dict(EVIDENCE_SCOPE),
            "acceptance_acquisition_roots": {
                "html_sha256": HARNESS_HTML_SHA256,
                "javascript_sha256": HARNESS_JAVASCRIPT_SHA256,
                "ort_javascript_sha256": ORT_JAVASCRIPT_SHA256,
                "ort_wasm_sha256": ORT_WASM_SHA256,
            },
            "harness_identity": {
                "schema_version": 2,
                "html": {
                    "relative_path": HARNESS_HTML_FILE,
                    "url": f"https://fixture.invalid/{HARNESS_HTML_FILE}",
                    "bytes": HARNESS_HTML_BYTES,
                    "sha256": HARNESS_HTML_SHA256,
                    "external_expected_sha256": HARNESS_HTML_SHA256,
                    "hash_verified": True,
                },
                "javascript": {
                    "relative_path": HARNESS_JAVASCRIPT_FILE,
                    "url": f"https://fixture.invalid/{HARNESS_JAVASCRIPT_FILE}",
                    "bytes": HARNESS_JAVASCRIPT_BYTES,
                    "sha256": HARNESS_JAVASCRIPT_SHA256,
                    "external_expected_sha256": HARNESS_JAVASCRIPT_SHA256,
                    "hash_verified": True,
                },
                "ort": {
                    "javascript": {
                        "relative_path": HARNESS_ORT_JAVASCRIPT_FILE,
                        "url": ORT_JAVASCRIPT_URL,
                        "bytes": 67_237,
                        "sha256": ORT_JAVASCRIPT_SHA256,
                        "external_expected_sha256": ORT_JAVASCRIPT_SHA256,
                        "hash_verified": True,
                    },
                    "wasm": {
                        "relative_path": HARNESS_ORT_WASM_FILE,
                        "url": ORT_WASM_URL,
                        "bytes": 1_000_000,
                        "sha256": ORT_WASM_SHA256,
                        "external_expected_sha256": ORT_WASM_SHA256,
                        "hash_verified": True,
                    },
                    "self_hosted_same_origin": True,
                    "version_pin": HARNESS_ORT_VERSION,
                    "version_reported": HARNESS_ORT_VERSION,
                    "version_verified": True,
                },
            },
            "action_capability_evaluated": False,
            "action_capability_claimed": False,
            "estimand": "prefill_and_iterative_cache_bearing_graph_latency",
            "ttft_boundary": (
                "immediately before prefill session.run through validated CPU logits argmax "
                "availability"
            ),
            "tpot_boundary": (
                "wall time from the first validated CPU logits argmax through the final iterative "
                "validated CPU logits argmax divided by output_tokens_minus_one"
            ),
            "prefill_ms_boundary": (
                "immediately before prefill session.run through promise resolution"
            ),
            "decode_inference_ms_boundary": (
                "sum of immediately-before decode session.run through promise resolution"
            ),
            "model_decode_tokens_per_second_definition": (
                "(output_tokens_minus_one)*1000/summed_decode_inference_ms"
            ),
            "excluded_from_latency": (
                "manifest/config/provenance/graph fetch and hashing, session creation, "
                "deterministic prompt construction, summary rendering, download serialization, "
                "and all quality evaluation"
            ),
            "provider": provider,
            "required_webgpu_provider_verification": True,
            "ort_version_pin": "1.27.0",
            "ort_version_reported": "1.27.0",
            "ort_version_verified": True,
            "ort_version_verification_status": "matches_script_pin",
            "ort_script_url": ORT_JAVASCRIPT_URL,
            "ort_wasm_url": ORT_WASM_URL,
            "cross_origin_isolated": False,
            "shared_array_buffer_available": False,
            "ort_wasm_num_threads": 1,
            "browser": {
                "user_agent": "fixture",
                "user_agent_brands": None,
                "mobile": None,
                "platform": "fixture",
                "language": "en",
                "languages": ["en"],
                "hardware_concurrency": 8,
                "device_memory_gb": 16,
            },
            "gpu": {
                "navigator_gpu_available": True,
                "ort_webgpu": provider["ort_webgpu"],
                "device_label": "fixture",
                "device_features": ["shader-f16"],
            },
            "user_agent": "fixture",
            "language": "en",
            "hardware_concurrency": 8,
            "device_memory_gb": 16,
            "timer": "performance.now",
            "concurrency": 1,
            "tab_visibility_required": True,
            "run_once_reload_required": True,
            "manifest_url": "https://fixture.invalid/single-decode.json",
            "manifest_raw_text": manifest_raw_text,
            "manifest_sha256": manifest_sha256,
            "manifest": manifest,
            "verified_identities": {
                "manifest_sha256": manifest_sha256,
                "artifacts": [],
                "arms": [],
            },
            "context_lengths": list(ACCEPTANCE_CONTEXT_LENGTHS),
            "prompt_lengths_tokens": list(ACCEPTANCE_CONTEXT_LENGTHS),
            "context_condition": "exact_prefill_input_tensor_sequence_length",
            "input_semantics": "deterministic_pretokenized_ids",
            "input_fixture_contract": "ids[i]=(131*i+17) mod vocab_size",
            "tokenizer_asset": {
                "loaded_by_benchmark": False,
                "input_ids_are_pretokenized": True,
                "provenance_pin": None,
            },
            "output_tokens_per_condition": ACCEPTANCE_OUTPUT_TOKENS,
            "reported_percentiles": ["p50", "p95"],
            "warmups_per_condition": ACCEPTANCE_WARMUPS,
            "measured_repetitions_per_condition": ACCEPTANCE_REPETITIONS,
            "case_order_seed": ACCEPTANCE_SEED,
            "session_order_seed": ACCEPTANCE_SESSION_ORDER_SEED,
            "decision_output_abi": ACCEPTANCE_DECISION_ABI,
            "greedy_selection": "validated_logits_argmax",
            "graph_pass_contract": {
                "prefill_per_condition": 1,
                "decode_per_condition": 31,
                "total_per_condition": 32,
                "first_token_source": (
                    "prefill.logits argmax; next_token compatibility cross-check"
                ),
                "remaining_token_source": (
                    "decode.logits argmax; next_token compatibility cross-check"
                ),
            },
            "cache_contract": {
                "enabled": True,
                "webgpu_cache_residency": "gpu-buffer",
                "wasm_cache_residency": "cpu",
                "next_token_residency": "cpu",
                "logits_residency": "cpu",
                "token_selection_source": "validated_logits_argmax",
                "update_strategy": (
                    "present_outputs_rebound_directly_as_past_inputs_without_cpu_materialization"
                ),
                "cache_data_read_to_javascript": False,
                "superseded_and_final_cache_disposal_attempted": True,
            },
            "arm_count": 1,
            "arms": [arm],
            "warmups_excluded_from_summary": True,
            "page_to_ready_ms": 100.0,
        },
        "artifact_verification_records": artifacts,
        "session_records": sessions,
        "input_preparation_record": {
            **TRAINED_LABELS,
            "phase": "input_preparation",
            "duration_ms": 1.0,
            "input_semantics": "deterministic_pretokenized_ids",
            "fixture_contract": "ids[i]=(131*i+17) mod vocab_size",
            "vocab_size": 256,
            "tokenizer_asset": None,
            "requested_context_lengths": list(ACCEPTANCE_CONTEXT_LENGTHS),
            "all_actual_lengths_verified": True,
        },
        "inputs": [_input_fixture(context) for context in ACCEPTANCE_CONTEXT_LENGTHS],
        "warmup_records": warmups,
        "records": measured,
        "summary": {
            "estimand": "trained_weight_cached_autoregressive_graph_latency",
            "quality_metrics_included": False,
            "attempted": len(measured),
            "completed": len(measured),
            "failed": 0,
            "conditions": [],
        },
        "failures": [],
        "errors": [],
    }


def _payload(result: dict | None = None) -> bytes:
    return (json.dumps(result or _result(), separators=(",", ":")) + "\n").encode()


def _root(result: dict) -> str:
    return result["metadata"]["manifest_sha256"]


def _build_receipt(
    result: dict,
    *,
    expected_root: str | None = None,
    expected_run_challenge: str = RUN_CHALLENGE,
) -> dict:
    return build_webgpu_decode_receipt(
        _payload(result),
        expected_wrapper_manifest_sha256=expected_root or _root(result),
        expected_checkpoint_sha256=CHECKPOINT_SHA256,
        expected_run_challenge=expected_run_challenge,
        expected_machine_condition_sha256=MACHINE_CONDITION_SHA256,
        expected_harness_html_sha256=HARNESS_HTML_SHA256,
        expected_harness_javascript_sha256=HARNESS_JAVASCRIPT_SHA256,
        expected_ort_javascript_sha256=ORT_JAVASCRIPT_SHA256,
        expected_ort_wasm_sha256=ORT_WASM_SHA256,
    )


def _external_cli_args() -> list[str]:
    return [
        "--expected-run-challenge",
        RUN_CHALLENGE,
        "--expected-machine-condition-sha256",
        MACHINE_CONDITION_SHA256,
        "--expected-harness-html-sha256",
        HARNESS_HTML_SHA256,
        "--expected-harness-javascript-sha256",
        HARNESS_JAVASCRIPT_SHA256,
        "--expected-ort-javascript-sha256",
        ORT_JAVASCRIPT_SHA256,
        "--expected-ort-wasm-sha256",
        ORT_WASM_SHA256,
    ]


def _reself_hash(receipt: dict) -> bytes:
    unsigned = dict(receipt)
    unsigned.pop("receipt_self_sha256", None)
    receipt["receipt_self_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    return canonical_json_bytes(receipt)


def test_acceptance_result_builds_canonical_exact_receipt():
    result = _result()
    receipt = _build_receipt(result)
    canonical = canonical_json_bytes(receipt)

    assert verify_webgpu_decode_receipt_bytes(canonical) == receipt
    assert receipt["verified"] is True
    assert receipt["protocol"]["warmup_records"] == 12
    assert receipt["protocol"]["measured_records"] == 120
    assert receipt["model"]["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert receipt["model"]["training_artifacts"][0]["sha256"] == TRAINING_SHA256
    assert receipt["bundle"]["wrapper_manifest_sha256"] == _root(result)
    assert receipt["bundle"]["external_wrapper_manifest_sha256"] == _root(result)
    assert [row["input_tokens"] for row in receipt["metrics_by_context"]] == list(
        ACCEPTANCE_CONTEXT_LENGTHS
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("manifest", "manifest raw-text SHA-256 mismatch"),
        ("root", "zero digest"),
        ("provenance", "parsed provenance disagrees"),
        ("stage", "parsed provenance disagrees"),
        ("lineage", "parsed training lineage disagrees"),
        ("repetitions", "browser protocol fields differ"),
        ("cardinality", "measured record cardinality mismatch"),
        ("passes", "graph-pass counts"),
        ("metric", "wall-clock decode rate"),
        ("cache_bytes", "cache logical-byte totals"),
        ("tensor_bytes", "logical bytes do not match dtype/dimensions"),
        ("disposal", "disposal accounting"),
        ("capability", "metadata fields differ"),
        ("runtime", "acquisition bytes"),
        ("seed", "browser protocol fields differ"),
        ("harness", "external acquisition root"),
        ("record_identity", "successful run"),
        ("float_bool", "must be finite"),
    ],
)
def test_acceptance_result_rejects_semantic_or_evidence_drift(
    mutation: str,
    message: str,
):
    result = copy.deepcopy(_result())
    expected_root = _root(result)
    if mutation == "manifest":
        result["metadata"]["manifest_raw_text"] += " "
    elif mutation == "root":
        expected_root = "0" * 64
    elif mutation == "provenance":
        result["metadata"]["arms"][0]["provenance"]["trained"] = False
    elif mutation == "stage":
        result["metadata"]["arms"][0]["provenance"]["weights"]["checkpoint_stage"] = "pretrain"
    elif mutation == "lineage":
        result["metadata"]["arms"][0]["training_lineage"]["stage"] = "pretrain"
    elif mutation == "repetitions":
        result["metadata"]["measured_repetitions_per_condition"] = 31
    elif mutation == "cardinality":
        result["records"].pop()
    elif mutation == "passes":
        result["records"][0]["graph_pass_counts"]["decode"] = 30
    elif mutation == "metric":
        result["records"][0]["decode_tokens_per_second"] = 1e12
    elif mutation == "cache_bytes":
        result["records"][0]["cache"]["final_logical_bytes"] += 1
    elif mutation == "tensor_bytes":
        result["records"][0]["cache"]["final_tensors"][0]["logical_bytes"] += 16
        result["records"][0]["cache"]["final_logical_bytes"] += 16
        result["records"][0]["decode_pass_records"][-1]["cache_logical_bytes_after"] += 16
    elif mutation == "disposal":
        result["records"][0]["allocation_disposal"]["cache_dispose_api_unavailable"] = 1
    elif mutation == "capability":
        result["metadata"]["capability_artifact_type"] = "action-evaluation"
    elif mutation == "runtime":
        result["metadata"]["ort_version_reported"] = None
        result["metadata"]["ort_version_verified"] = None
    elif mutation == "seed":
        result["metadata"]["case_order_seed"] = "different-seed"
    elif mutation == "harness":
        result["metadata"]["harness_identity"]["javascript"]["sha256"] = "0" * 64
    elif mutation == "record_identity":
        result["records"][0]["run_id"] = "33333333-3333-4333-8333-333333333333"
    elif mutation == "float_bool":
        result["records"][0]["tpot_ms"] = True

    with pytest.raises(ValueError, match=message):
        _build_receipt(result, expected_root=expected_root)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda result: result["records"][0]["cache"]["prefill_tensors"][0].update(
                {"dims": [1, 2, 128, 4]}
            ),
            "exact cache-slot shape",
        ),
        (
            lambda result: result["records"][0]["decode_pass_records"][1].update(
                {
                    "pass_started_offset_ms": 0.0,
                    "pass_resolved_offset_ms": (
                        result["records"][0]["decode_pass_records"][1]["inference_ms"]
                    ),
                    "token_available_offset_ms": (
                        result["records"][0]["decode_pass_records"][1]["inference_ms"]
                        + 0.01
                    ),
                }
            ),
            "boundaries overlap",
        ),
        (
            lambda result: result["records"][0]["graph_pass_counts"].update(
                {"prefill": True}
            ),
            "graph-pass counts",
        ),
        (
            lambda result: result["records"][0].update({"global_order_index": False}),
            "non-negative integer",
        ),
    ],
)
def test_raw_verifier_rejects_axis_timing_and_boolean_exploits(mutate, message: str):
    result = _result()
    mutate(result)
    with pytest.raises(ValueError, match=message):
        _build_receipt(result)


def test_raw_verifier_binds_seeded_case_and_session_order():
    result = _result()
    result["records"][0], result["records"][1] = result["records"][1], result["records"][0]
    result["records"][0]["global_order_index"] = 0
    result["records"][1]["global_order_index"] = 1
    with pytest.raises(ValueError, match="seeded schedule"):
        _build_receipt(result)

    result = _result()
    result["session_records"].reverse()
    for index, session in enumerate(result["session_records"]):
        session["order_index"] = index
    with pytest.raises(ValueError, match="session creation order"):
        _build_receipt(result)


def test_raw_verifier_requires_external_challenge_machine_and_acquisition_roots():
    result = _result()
    with pytest.raises(ValueError, match="zero digest"):
        _build_receipt(result, expected_run_challenge="0" * 64)

    result = _result()
    result["metadata"]["external_machine_condition_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="machine condition"):
        _build_receipt(result)

    result = _result()
    result["metadata"]["harness_identity"]["ort"]["wasm"]["hash_verified"] = False
    with pytest.raises(ValueError, match="external acquisition root"):
        _build_receipt(result)


@pytest.mark.parametrize(
    "container",
    [
        "result",
        "metadata",
        "session",
        "session_ort",
        "session_adapter",
        "preferred_output_location",
        "record",
        "graph_pass_counts",
        "decode_pass",
        "cache",
        "cache_tensor",
        "allocation_disposal",
    ],
)
def test_raw_verifier_rejects_unvalidated_trace_nonce_everywhere(container: str):
    result = _result()
    targets = {
        "result": result,
        "metadata": result["metadata"],
        "session": result["session_records"][0],
        "session_ort": result["session_records"][0]["ort_webgpu"],
        "session_adapter": result["session_records"][0]["ort_webgpu"]["adapter_info"],
        "preferred_output_location": result["session_records"][0][
            "preferred_output_location"
        ],
        "record": result["records"][0],
        "graph_pass_counts": result["records"][0]["graph_pass_counts"],
        "decode_pass": result["records"][0]["decode_pass_records"][0],
        "cache": result["records"][0]["cache"],
        "cache_tensor": result["records"][0]["cache"]["prefill_tensors"][0],
        "allocation_disposal": result["records"][0]["allocation_disposal"],
    }
    targets[container]["unvalidated_trace_nonce"] = container
    with pytest.raises(ValueError, match="fields differ"):
        _build_receipt(result)


def test_receipt_verifier_rejects_minimal_fabrication_even_when_self_hashed():
    fabricated = {
        "schema_version": 1,
        "kind": "localagent_webgpu_cached_decode_acceptance_receipt",
        "verified": True,
    }
    payload = _reself_hash(fabricated)
    with pytest.raises(ValueError, match="receipt fields differ"):
        verify_webgpu_decode_receipt_bytes(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update({"verified": False}), "receipt identity"),
        (
            lambda value: value["runtime"].update({"graph_wide_provider_verified": True}),
            "runtime evidence",
        ),
        (
            lambda value: value["record_contract"].update(
                {"all_metric_algebra_verified": False}
            ),
            "record contract",
        ),
        (
            lambda value: value["model"].update({"training_artifacts": []}),
            "training identity/hash",
        ),
        (
            lambda value: value["protocol"].update({"measured_records": 4}),
            "exact acceptance protocol",
        ),
        (
            lambda value: value["bundle"].update({"wrapper_manifest_sha256": "0" * 64}),
            "recorded external root",
        ),
    ],
)
def test_receipt_verifier_rejects_semantic_mutation_after_reself_hash(mutate, message: str):
    result = _result()
    receipt = _build_receipt(result)
    mutate(receipt)

    with pytest.raises(ValueError, match=message):
        verify_webgpu_decode_receipt_bytes(_reself_hash(receipt))


def test_verifier_cli_requires_root_and_writes_the_same_canonical_receipt(tmp_path: Path):
    result = _result()
    result_path = tmp_path / "result.json"
    receipt_path = tmp_path / "receipt.json"
    result_path.write_bytes(_payload(result))
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")

    missing = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_webgpu_decode_result.py"),
            str(result_path),
            "--output",
            str(receipt_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert missing.returncode != 0
    assert "--expected-checkpoint-sha256" in missing.stderr
    assert "--expected-wrapper-manifest-sha256" in missing.stderr

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_webgpu_decode_result.py"),
            str(result_path),
            "--output",
            str(receipt_path),
            "--expected-checkpoint-sha256",
            CHECKPOINT_SHA256,
            "--expected-wrapper-manifest-sha256",
            _root(result),
            *_external_cli_args(),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    receipt = verify_webgpu_decode_receipt_bytes(receipt_path.read_bytes())
    assert receipt["result"]["sha256"] == hashlib.sha256(result_path.read_bytes()).hexdigest()
    assert f"external wrapper root: {_root(result)}" in completed.stdout
    assert "receipt file SHA-256" in completed.stdout

    original = receipt_path.read_bytes()
    refused = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_webgpu_decode_result.py"),
            str(result_path),
            "--output",
            str(receipt_path),
            "--expected-checkpoint-sha256",
            CHECKPOINT_SHA256,
            "--expected-wrapper-manifest-sha256",
            _root(result),
            *_external_cli_args(),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert refused.returncode != 0
    assert "refusing to overwrite" in refused.stderr
    assert receipt_path.read_bytes() == original

    linked_result = tmp_path / "linked-result.json"
    linked_result.symlink_to(result_path)
    linked_output = tmp_path / "linked-input-receipt.json"
    linked = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_webgpu_decode_result.py"),
            str(linked_result),
            "--output",
            str(linked_output),
            "--expected-checkpoint-sha256",
            CHECKPOINT_SHA256,
            "--expected-wrapper-manifest-sha256",
            _root(result),
            *_external_cli_args(),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert linked.returncode != 0
    assert "no-follow regular file" in linked.stderr
    assert not linked_output.exists()


def _campaign_pairs(
    tmp_path: Path,
) -> tuple[list[tuple[Path, Path]], str, list[str]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    pairs = []
    challenges = []
    wrapper_root = ""
    for run_index in range(1, 4):
        result = copy.deepcopy(_result())
        challenge = str(run_index + 6) * 64
        session_id = f"11111111-1111-4111-8111-{run_index:012d}"
        run_id = f"22222222-2222-4222-8222-{run_index:012d}"
        result["created_at"] = f"2026-07-29T12:00:0{run_index}.000Z"
        result["metadata"]["benchmark_session_id"] = session_id
        result["metadata"]["run_id"] = run_id
        result["metadata"]["run_challenge"] = challenge
        for session in result["session_records"]:
            session["benchmark_session_id"] = session_id
            session["run_challenge"] = challenge
        for record in [*result["warmup_records"], *result["records"]]:
            record["benchmark_session_id"] = session_id
            record["run_id"] = run_id
            record["run_challenge"] = challenge
            record["prefill_ms"] += run_index / 1000
            record["ttft_ms"] += run_index / 1000
            record["generation_wall_ms"] += run_index / 1000
        wrapper_root = _root(result)
        receipt = _build_receipt(
            result,
            expected_run_challenge=challenge,
        )
        raw_path = tmp_path / f"run-{run_index}.raw.json"
        receipt_path = tmp_path / f"run-{run_index}.receipt.json"
        raw_path.write_bytes(_payload(result))
        write_webgpu_decode_receipt(receipt_path, receipt)
        pairs.append((raw_path, receipt_path))
        challenges.append(challenge)
    return pairs, wrapper_root, challenges


def _campaign_kwargs(wrapper_root: str, challenges: list[str]) -> dict[str, Any]:
    return {
        "expected_checkpoint_sha256": CHECKPOINT_SHA256,
        "expected_wrapper_manifest_sha256": wrapper_root,
        "expected_run_challenges": challenges,
        "expected_machine_condition_sha256": MACHINE_CONDITION_SHA256,
        "expected_harness_html_sha256": HARNESS_HTML_SHA256,
        "expected_harness_javascript_sha256": HARNESS_JAVASCRIPT_SHA256,
        "expected_ort_javascript_sha256": ORT_JAVASCRIPT_SHA256,
        "expected_ort_wasm_sha256": ORT_WASM_SHA256,
    }


def _set_exact_tpot(result: dict, tpot_ms: float) -> None:
    decode_wall_ms = tpot_ms * (ACCEPTANCE_OUTPUT_TOKENS - 1)
    decode_rate = 1000.0 / tpot_ms
    for record in [*result["warmup_records"], *result["records"]]:
        final_pass = record["decode_pass_records"][-1]
        final_pass["token_available_offset_ms"] = decode_wall_ms
        final_pass["token_available_ms"] = (
            decode_wall_ms - final_pass["pass_started_offset_ms"]
        )
        record["decode_wall_ms"] = decode_wall_ms
        record["generation_wall_ms"] = record["ttft_ms"] + decode_wall_ms
        record["tpot_ms"] = tpot_ms
        record["decode_tokens_per_second"] = decode_rate


def _rewrite_receipt(path: Path, mutate) -> None:
    receipt = json.loads(path.read_bytes())
    mutate(receipt)
    path.write_bytes(_reself_hash(receipt))


def _reself_hash_campaign(campaign: dict) -> bytes:
    unsigned = dict(campaign)
    unsigned.pop("campaign_self_sha256", None)
    campaign["campaign_self_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    return canonical_json_bytes(campaign)


def test_three_run_campaign_revalidates_receipts_and_recomputes_exact_counts(tmp_path: Path):
    pairs, wrapper_root, challenges = _campaign_pairs(tmp_path)
    campaign = build_webgpu_decode_campaign(
        pairs,
        **_campaign_kwargs(wrapper_root, challenges),
    )

    assert (
        verify_webgpu_decode_campaign_integrity_bytes(canonical_json_bytes(campaign))
        == campaign
    )
    assert campaign["counts"] == {
        "runs": 3,
        "warmup_records": 36,
        "measured_records": 360,
        "total_records": 396,
        "warmup_graph_calls": 1_152,
        "measured_graph_calls": 11_520,
        "prefill_graph_calls": 396,
        "decode_graph_calls": 12_276,
        "graph_calls": 12_672,
    }
    assert campaign["acceptance_gate"]["passed"] is True
    assert len(campaign["receipts"]) == 3
    assert all(
        row["passed"]
        for row in campaign["acceptance_gate"]["median_of_three_by_context"]
    )
    assert campaign["acceptance_gate"]["raw_receipts_rebuilt_and_byte_compared"] is True
    assert campaign["acceptance_gate"]["distinct_normalized_execution_traces"] is True


@pytest.mark.parametrize("tpot_ms", [1000.0 / 99.99999996, 10.000000004])
def test_campaign_thresholds_use_unrounded_raw_statistics(
    tmp_path: Path,
    tpot_ms: float,
):
    pairs, wrapper_root, challenges = _campaign_pairs(tmp_path)
    for run_index, (raw_path, receipt_path) in enumerate(pairs):
        result = json.loads(raw_path.read_bytes())
        _set_exact_tpot(result, tpot_ms)
        raw_path.write_bytes(_payload(result))
        receipt = _build_receipt(
            result,
            expected_run_challenge=challenges[run_index],
        )
        metric = receipt["metrics_by_context"][0]["metrics"]
        assert metric["tpot_ms"]["p95"] == pytest.approx(tpot_ms)
        assert metric["decode_tokens_per_second"]["p50"] < 100.0
        receipt_path.write_bytes(canonical_json_bytes(receipt))

    with pytest.raises(ValueError, match="misses the latency gate"):
        build_webgpu_decode_campaign(
            pairs,
            **_campaign_kwargs(wrapper_root, challenges),
        )


def test_authoritative_campaign_verifier_rejects_resealed_metrics(tmp_path: Path):
    pairs, wrapper_root, challenges = _campaign_pairs(tmp_path)
    kwargs = _campaign_kwargs(wrapper_root, challenges)
    valid = build_webgpu_decode_campaign(pairs, **kwargs)
    forged_receipts = []
    forged_payloads = []
    for entry in valid["receipts"]:
        receipt = copy.deepcopy(entry["receipt"])
        for row in receipt["metrics_by_context"]:
            row["metrics"]["decode_tokens_per_second"]["p50"] = 1_000_000.0
            row["metrics"]["decode_tokens_per_second"]["p95"] = 1_000_000.0
            row["metrics"]["tpot_ms"]["p50"] = 0.001
            row["metrics"]["tpot_ms"]["p95"] = 0.001
        payload = _reself_hash(receipt)
        forged_payloads.append(payload)
        forged_receipts.append(verify_webgpu_decode_receipt_bytes(payload))
    raw_identities = [entry["raw_result_file"] for entry in valid["receipts"]]
    trace_hashes = [
        entry["normalized_execution_trace_sha256"] for entry in valid["receipts"]
    ]
    forged_core = campaign_module._campaign_core(
        forged_payloads,
        forged_receipts,
        raw_identities,
        trace_hashes,
        expected_checkpoint_sha256=CHECKPOINT_SHA256,
        expected_wrapper_manifest_sha256=wrapper_root,
        expected_run_challenges=challenges,
        expected_machine_condition_sha256=MACHINE_CONDITION_SHA256,
        expected_harness_html_sha256=HARNESS_HTML_SHA256,
        expected_harness_javascript_sha256=HARNESS_JAVASCRIPT_SHA256,
        expected_ort_javascript_sha256=ORT_JAVASCRIPT_SHA256,
        expected_ort_wasm_sha256=ORT_WASM_SHA256,
    )
    forged = {
        **forged_core,
        "campaign_self_sha256": hashlib.sha256(
            canonical_json_bytes(forged_core)
        ).hexdigest(),
    }
    forged_path = tmp_path / "resealed-campaign.json"
    forged_path.write_bytes(canonical_json_bytes(forged))

    assert (
        verify_webgpu_decode_campaign_integrity_bytes(forged_path.read_bytes())
        == forged
    )
    with pytest.raises(ValueError, match="artifact-backed reconstruction"):
        verify_webgpu_decode_campaign_against_artifacts(
            forged_path,
            pairs,
            **kwargs,
        )
    with pytest.raises(ValueError, match="artifact-backed reconstruction"):
        write_webgpu_decode_campaign(
            tmp_path / "must-not-publish.json",
            forged,
            pairs,
            **kwargs,
        )


def test_campaign_rejects_cloned_normalized_execution_traces(tmp_path: Path):
    pairs, wrapper_root, challenges = _campaign_pairs(tmp_path)
    source = json.loads(pairs[0][0].read_bytes())
    for run_index in (2, 3):
        result = copy.deepcopy(source)
        session_id = f"11111111-1111-4111-8111-{run_index:012d}"
        run_id = f"22222222-2222-4222-8222-{run_index:012d}"
        challenge = challenges[run_index - 1]
        result["created_at"] = f"2026-07-29T12:00:0{run_index}.000Z"
        result["metadata"]["benchmark_session_id"] = session_id
        result["metadata"]["run_id"] = run_id
        result["metadata"]["run_challenge"] = challenge
        for session in result["session_records"]:
            session["benchmark_session_id"] = session_id
            session["run_challenge"] = challenge
        for record in [*result["warmup_records"], *result["records"]]:
            record["benchmark_session_id"] = session_id
            record["run_id"] = run_id
            record["run_challenge"] = challenge
        pairs[run_index - 1][0].write_bytes(_payload(result))
        receipt = _build_receipt(result, expected_run_challenge=challenge)
        pairs[run_index - 1][1].write_bytes(canonical_json_bytes(receipt))

    with pytest.raises(ValueError, match="normalized execution traces"):
        build_webgpu_decode_campaign(
            pairs,
            **_campaign_kwargs(wrapper_root, challenges),
        )


def test_campaign_rejects_raw_receipt_mismatch_and_symlink_inputs(tmp_path: Path):
    pairs, wrapper_root, challenges = _campaign_pairs(tmp_path)
    changed = json.loads(pairs[0][0].read_bytes())
    changed["warning"] = "byte-distinct but otherwise valid"
    pairs[0][0].write_bytes(_payload(changed))
    with pytest.raises(ValueError, match="exact receipt rebuilt"):
        build_webgpu_decode_campaign(
            pairs,
            **_campaign_kwargs(wrapper_root, challenges),
        )

    pairs, wrapper_root, challenges = _campaign_pairs(tmp_path / "fresh")
    link = tmp_path / "raw-result-link.json"
    link.symlink_to(pairs[0][0])
    linked_pairs = [(link, pairs[0][1]), *pairs[1:]]
    with pytest.raises(ValueError, match="no-follow regular file"):
        build_webgpu_decode_campaign(
            linked_pairs,
            **_campaign_kwargs(wrapper_root, challenges),
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda target, first: target.update(
                {"benchmark_created_at": first["benchmark_created_at"]}
            ),
            "strictly chronological",
        ),
        (
            lambda target, first: target.update({"result": copy.deepcopy(first["result"])}),
            "raw-result SHA-256",
        ),
        (
            lambda target, first: target["execution"].update(
                {"benchmark_session_id": first["execution"]["benchmark_session_id"]}
            ),
            "session/run identities",
        ),
        (
            lambda target, first: target["model"].update({"tokenizer_sha256": "c" * 64}),
            "model differs",
        ),
        (
            lambda target, first: target["bundle"].update({"decode_sha256": "0" * 64}),
            "bundle differs",
        ),
        (
            lambda target, first: target["runtime"]["browser"].update(
                {"user_agent": "different-browser"}
            ),
            "runtime differs",
        ),
    ],
)
def test_campaign_rejects_cross_run_mismatch_after_valid_receipt_rehash(
    tmp_path: Path,
    mutate,
    message: str,
):
    pairs, wrapper_root, challenges = _campaign_pairs(tmp_path)
    first = json.loads(pairs[0][1].read_bytes())
    _rewrite_receipt(pairs[1][1], lambda target: mutate(target, first))

    with pytest.raises(ValueError, match="exact receipt rebuilt"):
        build_webgpu_decode_campaign(
            pairs,
            **_campaign_kwargs(wrapper_root, challenges),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("p50", True),
        ("p50", 99.999999),
        ("p50", float("inf")),
    ],
)
def test_campaign_rejects_bool_nonfinite_and_subthreshold_metrics(
    tmp_path: Path,
    field: str,
    value: object,
):
    pairs, wrapper_root, challenges = _campaign_pairs(tmp_path)

    def mutate(receipt: dict) -> None:
        receipt["metrics_by_context"][0]["metrics"]["decode_tokens_per_second"][field] = value

    with pytest.raises((TypeError, ValueError)):
        _rewrite_receipt(pairs[0][1], mutate)
        build_webgpu_decode_campaign(
            pairs,
            **_campaign_kwargs(wrapper_root, challenges),
        )


def test_campaign_rejects_wrong_cardinality_and_external_roots(tmp_path: Path):
    pairs, wrapper_root, challenges = _campaign_pairs(tmp_path)
    with pytest.raises(ValueError, match="exactly three"):
        build_webgpu_decode_campaign(
            pairs[:2],
            **_campaign_kwargs(wrapper_root, challenges),
        )
    with pytest.raises(ValueError, match="zero digest"):
        kwargs = _campaign_kwargs(wrapper_root, challenges)
        kwargs["expected_checkpoint_sha256"] = "0" * 64
        build_webgpu_decode_campaign(
            pairs,
            **kwargs,
        )
    with pytest.raises(ValueError, match="zero digest"):
        kwargs = _campaign_kwargs(wrapper_root, challenges)
        kwargs["expected_wrapper_manifest_sha256"] = "0" * 64
        build_webgpu_decode_campaign(
            pairs,
            **kwargs,
        )
    with pytest.raises(ValueError, match="zero digest"):
        kwargs = _campaign_kwargs(wrapper_root, challenges)
        kwargs["expected_ort_wasm_sha256"] = "0" * 64
        build_webgpu_decode_campaign(
            pairs,
            **kwargs,
        )


def test_campaign_verifier_recomputes_fields_and_writer_never_overwrites(tmp_path: Path):
    pairs, wrapper_root, challenges = _campaign_pairs(tmp_path)
    campaign = build_webgpu_decode_campaign(
        pairs,
        **_campaign_kwargs(wrapper_root, challenges),
    )
    output = tmp_path / "campaign.json"
    write_webgpu_decode_campaign(
        output,
        campaign,
        pairs,
        **_campaign_kwargs(wrapper_root, challenges),
    )
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_webgpu_decode_campaign(
            output,
            campaign,
            pairs,
            **_campaign_kwargs(wrapper_root, challenges),
        )
    assert output.read_bytes() == original
    assert not list(tmp_path.glob(".campaign.json.*.tmp"))

    mutated = copy.deepcopy(campaign)
    mutated["counts"]["graph_calls"] = True
    with pytest.raises(ValueError, match="independently recomputed"):
        verify_webgpu_decode_campaign_integrity_bytes(_reself_hash_campaign(mutated))


def test_campaign_cli_requires_external_anchors_and_reports_recomputed_counts(tmp_path: Path):
    pairs, wrapper_root, challenges = _campaign_pairs(tmp_path)
    output = tmp_path / "campaign-cli.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    command = [
        sys.executable,
        str(ROOT / "scripts" / "summarize_single_webgpu_campaign.py"),
        *(
            item
            for raw_path, receipt_path in pairs
            for item in ("--run", str(raw_path), str(receipt_path))
        ),
        "--output",
        str(output),
    ]
    missing = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert missing.returncode != 0
    assert "--expected-checkpoint-sha256" in missing.stderr
    assert "--expected-wrapper-manifest-sha256" in missing.stderr

    completed = subprocess.run(
        [
            *command,
            "--expected-checkpoint-sha256",
            CHECKPOINT_SHA256,
            "--expected-wrapper-manifest-sha256",
            wrapper_root,
            *(
                item
                for challenge in challenges
                for item in ("--run-challenge", challenge)
            ),
            "--expected-machine-condition-sha256",
            MACHINE_CONDITION_SHA256,
            "--expected-harness-html-sha256",
            HARNESS_HTML_SHA256,
            "--expected-harness-javascript-sha256",
            HARNESS_JAVASCRIPT_SHA256,
            "--expected-ort-javascript-sha256",
            ORT_JAVASCRIPT_SHA256,
            "--expected-ort-wasm-sha256",
            ORT_WASM_SHA256,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert "36 warmups, 360 measurements, 12672 graph calls" in completed.stdout
    verify_webgpu_decode_campaign_against_artifacts(
        output,
        pairs,
        **_campaign_kwargs(wrapper_root, challenges),
    )
