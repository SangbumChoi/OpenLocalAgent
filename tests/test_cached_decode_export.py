import hashlib
import json
from dataclasses import asdict

import pytest
import yaml

from openlocalagent.model import ModelConfig

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")


def _write_matched_configs(tmp_path):
    common = {
        "vocab_size": 256,
        "d_model": 32,
        "n_layers": 3,
        "n_loops": 2,
        "n_heads": 4,
        "n_kv_heads": 1,
        "max_seq_len": 32,
        "rope_theta": 10000.0,
        "norm_eps": 1e-5,
        "tie_embeddings": True,
        "dropout": 0.0,
        "qk_norm": True,
        "conv_kernel": 3,
    }
    hybrid = ModelConfig(
        name="test-cached-hybrid",
        ffn_hidden=64,
        layer_types=["conv", "conv", "attn"],
        **common,
    )
    attention = ModelConfig(
        name="test-cached-attention",
        ffn_hidden=75,
        layer_types=["attn", "attn", "attn"],
        **common,
    )
    hybrid_path = tmp_path / "hybrid.yaml"
    attention_path = tmp_path / "attention.yaml"
    hybrid_path.write_text(yaml.safe_dump(asdict(hybrid), sort_keys=True))
    attention_path.write_text(yaml.safe_dump(asdict(attention), sort_keys=True))
    return hybrid_path, attention_path


def _elem_types(path):
    graph = onnx.load(path).graph
    return (
        {value.name: value.type.tensor_type.elem_type for value in graph.input},
        {value.name: value.type.tensor_type.elem_type for value in graph.output},
    )


def test_matched_cached_decode_export_has_dynamic_cache_abi_and_trajectory_parity(tmp_path):
    from onnx import TensorProto

    from openlocalagent.inference.export.to_onnx import export_matched_random_cached_decode

    hybrid_config, attention_config = _write_matched_configs(tmp_path)
    result = export_matched_random_cached_decode(
        str(hybrid_config),
        str(attention_config),
        str(tmp_path / "pair"),
        seed=29,
        fp16=True,
        fixture_lengths=(1, 5),
        decode_steps=3,
    )

    manifest_path = tmp_path / "pair" / "matched-decode.json"
    manifest = json.loads(manifest_path.read_text())
    assert result["manifest"] == manifest
    assert manifest["artifact_type"] == "matched_random_cached_decode_latency_suite"
    assert manifest["trained"] is False
    assert manifest["latency_only"] is True
    assert manifest["capability_artifact"] is False
    assert manifest["quality_claims"] == []
    assert "UNTRAINED RANDOM WEIGHTS" in manifest["warning"]
    assert manifest["match"]["relative_parameter_delta"] < 0.01

    for role in ("hybrid", "attention"):
        model_dir = tmp_path / "pair" / role
        assert not (model_dir / "single-decode.json").exists()
        provenance = json.loads((model_dir / "provenance.json").read_text())
        assert provenance["artifact_type"] == "random_weight_cached_decode_onnx"
        assert provenance["trained"] is False
        assert provenance["training_steps"] == 0
        assert provenance["capability_artifact"] is False
        assert provenance["quality_claims"] == []
        assert provenance["parity"]["hard_gate"] is True
        assert provenance["parity"]["fixture_lengths"] == [1, 5]
        assert "at least two" in provenance["parity"]["fixture_length_requirement"]
        assert provenance["parity"]["cache_atol_ceiling_by_precision"] == {
            "fp16": 0.1,
            "fp32": 0.001,
        }
        contract = provenance["graph_contract"]
        assert contract["decode_token_axis_fixed_one"] is True
        assert contract["decode_position"]["caller_position_input"] is False
        assert contract["decode_position"]["rule"] == (
            "RoPE position = first attention past-key axis-2 length"
        )
        slots = contract["cache_slots"]
        assert len(slots) == 6
        assert [slot["slot"] for slot in slots] == list(range(6))
        if role == "hybrid":
            assert [slot["kind"] for slot in slots] == [
                "conv",
                "conv",
                "attn",
                "conv",
                "conv",
                "attn",
            ]
        else:
            assert all(slot["kind"] == "attn" for slot in slots)

        for precision, cache_type in (
            ("fp32", TensorProto.FLOAT),
            ("fp16", TensorProto.FLOAT16),
        ):
            parity = provenance["parity"]["results"][precision]
            assert parity["passed"] is True
            assert parity["hard_gate"] is True
            assert parity["greedy_next_token_exact"] is True
            assert parity["decode_steps"] == 3
            assert parity["reference_independence"] == {
                "onnx_logits_vs_pytorch_cached_path": True,
                "onnx_vs_pytorch_cached_path": True,
                "pytorch_cached_vs_fresh_full_context_logits": True,
                "pytorch_cached_vs_fresh_full_context_greedy_token": True,
            }
            assert {item["prompt_length"] for item in parity["per_fixture"]} == {1, 5}
            assert all(len(item["decode"]) == 3 for item in parity["per_fixture"])
            assert all(
                item["prefill_cached_vs_full_context_next_token_exact"]
                for item in parity["per_fixture"]
            )
            assert all(
                step["next_token_exact"] and step["cached_vs_full_context_next_token_exact"]
                for item in parity["per_fixture"]
                for step in item["decode"]
            )

            prefill_path = model_dir / f"prefill.{precision}.onnx"
            decode_path = model_dir / f"decode.{precision}.onnx"
            prefill_inputs, prefill_outputs = _elem_types(prefill_path)
            decode_inputs, decode_outputs = _elem_types(decode_path)
            assert prefill_inputs == {"input_ids": TensorProto.INT64}
            assert prefill_outputs["next_token"] == TensorProto.INT64
            assert prefill_outputs["logits"] == cache_type
            assert decode_inputs["input_ids"] == TensorProto.INT64
            assert decode_outputs["next_token"] == TensorProto.INT64
            assert decode_outputs["logits"] == cache_type
            assert all(
                dtype == cache_type
                for name, dtype in prefill_outputs.items()
                if name != "next_token"
            )
            assert all(
                dtype == cache_type for name, dtype in decode_inputs.items() if name != "input_ids"
            )
            assert all(
                dtype == cache_type
                for name, dtype in decode_outputs.items()
                if name != "next_token"
            )
            token_dims = onnx.load(decode_path).graph.input[0].type.tensor_type.shape.dim
            assert token_dims[0].dim_param == "batch"
            assert token_dims[1].dim_value == 1
            logits_dims = next(
                value for value in onnx.load(decode_path).graph.output if value.name == "logits"
            ).type.tensor_type.shape.dim
            assert logits_dims[0].dim_param == "batch"
            assert logits_dims[1].dim_value == 256
            past_key = next(
                value for value in onnx.load(decode_path).graph.input if value.name.endswith("_key")
            )
            assert past_key.type.tensor_type.shape.dim[2].dim_param == "past_sequence"

        expected_artifacts = {
            "decode.fp16.onnx",
            "decode.fp32.onnx",
            "meta.json",
            "model-config.yaml",
            "prefill.fp16.onnx",
            "prefill.fp32.onnx",
        }
        assert set(provenance["artifacts"]) == expected_artifacts
        for artifact_name, artifact in provenance["artifacts"].items():
            path = model_dir / artifact_name
            assert artifact["bytes"] == path.stat().st_size
            assert artifact["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            pair_entry = manifest["artifacts"][f"{role}/{artifact_name}"]
            assert pair_entry["bytes"] == path.stat().st_size
            assert pair_entry["sha256"] == artifact["sha256"]


def test_cached_decode_reuses_exact_graph_across_different_prompt_lengths(tmp_path):
    from openlocalagent.inference.export.to_onnx import export_random_cached_decode

    hybrid_config, _ = _write_matched_configs(tmp_path)
    result = export_random_cached_decode(
        str(hybrid_config),
        str(tmp_path / "hybrid"),
        seed=31,
        pair_role="hybrid_treatment",
        fp16=False,
        fixture_lengths=(2, 7, 11),
        decode_steps=4,
    )
    parity = result["parity"]["fp32"]
    assert parity["decode_steps"] == 4
    assert [item["prompt_length"] for item in parity["per_fixture"]] == [2, 7, 11]
    assert all(len(item["decode"]) == 4 for item in parity["per_fixture"])
    assert parity["max_cache_abs_diff"] <= parity["cache_atol"]


def test_cached_decode_rejects_short_parity_trajectory(tmp_path):
    from openlocalagent.inference.export.to_onnx import export_random_cached_decode

    hybrid_config, _ = _write_matched_configs(tmp_path)
    with pytest.raises(ValueError, match="at least three"):
        export_random_cached_decode(
            str(hybrid_config),
            str(tmp_path / "hybrid"),
            pair_role="hybrid_treatment",
            decode_steps=2,
        )
    assert not (tmp_path / "hybrid").exists()


@pytest.mark.parametrize(
    ("fixture_lengths", "message"),
    [
        ((4,), "at least two distinct"),
        ((4, 4), "must be distinct"),
    ],
)
def test_cached_decode_rejects_fixture_lengths_that_cannot_prove_dynamic_rope(
    tmp_path,
    fixture_lengths,
    message,
):
    from openlocalagent.inference.export.to_onnx import export_random_cached_decode

    hybrid_config, _ = _write_matched_configs(tmp_path)
    output = tmp_path / "hybrid"
    with pytest.raises(ValueError, match=message):
        export_random_cached_decode(
            str(hybrid_config),
            str(output),
            pair_role="hybrid_treatment",
            fixture_lengths=fixture_lengths,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fp32_cache_atol", -1.0),
        ("fp32_cache_atol", 0.0010001),
        ("fp32_cache_atol", float("nan")),
        ("fp32_cache_atol", float("inf")),
        ("fp16_cache_atol", -1.0),
        ("fp16_cache_atol", 0.10001),
        ("fp16_cache_atol", float("nan")),
        ("fp16_cache_atol", float("inf")),
    ],
)
def test_cached_decode_rejects_bypassable_cache_tolerances(
    tmp_path,
    field,
    value,
):
    from openlocalagent.inference.export.to_onnx import export_random_cached_decode

    hybrid_config, _ = _write_matched_configs(tmp_path)
    output = tmp_path / "hybrid"
    kwargs = {field: value}
    with pytest.raises(ValueError, match="must be finite"):
        export_random_cached_decode(
            str(hybrid_config),
            str(output),
            pair_role="hybrid_treatment",
            fixture_lengths=(2, 5),
            **kwargs,
        )
    assert not output.exists()
