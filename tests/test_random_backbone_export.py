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
        "n_loops": 1,
        "n_heads": 4,
        "n_kv_heads": 1,
        "max_seq_len": 64,
        "rope_theta": 10000.0,
        "norm_eps": 1e-5,
        "tie_embeddings": True,
        "dropout": 0.0,
        "qk_norm": True,
        "conv_kernel": 3,
    }
    hybrid = ModelConfig(
        name="test-hybrid",
        ffn_hidden=64,
        layer_types=["conv", "conv", "attn"],
        **common,
    )
    attention = ModelConfig(
        name="test-attention",
        ffn_hidden=75,
        layer_types=["attn", "attn", "attn"],
        **common,
    )
    hybrid_path = tmp_path / "hybrid.yaml"
    attention_path = tmp_path / "attention.yaml"
    hybrid_path.write_text(yaml.safe_dump(asdict(hybrid), sort_keys=True))
    attention_path.write_text(yaml.safe_dump(asdict(attention), sort_keys=True))
    return hybrid_path, attention_path


def test_matched_random_pair_is_hidden_only_parity_gated_and_honestly_labeled(tmp_path):
    from openlocalagent.inference.export.to_onnx import export_matched_random_backbones

    hybrid_config, attention_config = _write_matched_configs(tmp_path)
    result = export_matched_random_backbones(
        str(hybrid_config),
        str(attention_config),
        str(tmp_path / "pair"),
        seed=19,
        fp16=True,
        fixture_lengths=(1, 5),
    )

    manifest = json.loads((tmp_path / "pair" / "matched-backbones.json").read_text())
    assert result["manifest"] == manifest
    assert manifest["artifact_type"] == "matched_random_backbone_latency_suite"
    assert manifest["trained"] is False
    assert manifest["latency_only"] is True
    assert manifest["capability_artifact"] is False
    assert manifest["quality_claims"] == []
    assert "UNTRAINED RANDOM WEIGHTS" in manifest["warning"]
    assert manifest["shared_random_seed"] == 19
    assert manifest["match"]["relative_parameter_delta"] < 0.01
    assert set(manifest["intentional_differences"]) == {
        "ffn_hidden",
        "layer_types",
        "name",
    }

    for role in ("hybrid", "attention"):
        model_dir = tmp_path / "pair" / role
        provenance = json.loads((model_dir / "provenance.json").read_text())
        assert provenance["artifact_type"] == "random_weight_hidden_backbone_onnx"
        assert provenance["trained"] is False
        assert provenance["training_steps"] == 0
        assert provenance["latency_only"] is True
        assert provenance["capability_artifact"] is False
        assert provenance["capability_metrics"] is None
        assert provenance["quality_claims"] == []
        assert provenance["weights"]["checkpoint"] is None
        assert provenance["weights"]["source"] == "deterministic_random_initialization"
        assert provenance["weights"]["seed"] == 19
        assert provenance["parity"]["hard_gate"] is True
        assert all(parity["passed"] for parity in provenance["parity"]["results"].values())
        assert set(provenance["artifacts"]) == {
            "backbone.fp16.onnx",
            "backbone.fp32.onnx",
            "model-config.yaml",
        }
        assert provenance["graph_contract"]["omits"] == [
            "language_model_logits",
            "tool_heads",
            "pointer_heads",
            "route_heads",
            "kv_cache",
        ]

        for graph_name in ("backbone.fp32.onnx", "backbone.fp16.onnx"):
            graph_path = model_dir / graph_name
            graph = onnx.load(graph_path)
            assert [value.name for value in graph.graph.input] == ["input_ids"]
            assert [value.name for value in graph.graph.output] == ["hidden"]
            assert not any("lm_head" in value.name for value in graph.graph.initializer)
            artifact = provenance["artifacts"][graph_name]
            assert artifact["bytes"] == graph_path.stat().st_size
            assert artifact["sha256"] == hashlib.sha256(graph_path.read_bytes()).hexdigest()


def test_random_hidden_export_repeats_exact_state_and_fp32_graph(tmp_path):
    from openlocalagent.inference.export.to_onnx import export_random_hidden_backbone

    hybrid_config, _ = _write_matched_configs(tmp_path)
    first = export_random_hidden_backbone(
        str(hybrid_config),
        str(tmp_path / "first"),
        seed=23,
        pair_role="hybrid_treatment",
        fp16=False,
        fixture_lengths=(2, 7),
    )
    second = export_random_hidden_backbone(
        str(hybrid_config),
        str(tmp_path / "second"),
        seed=23,
        pair_role="hybrid_treatment",
        fp16=False,
        fixture_lengths=(2, 7),
    )

    assert first["state_dict_sha256"] == second["state_dict_sha256"]
    assert (tmp_path / "first" / "backbone.fp32.onnx").read_bytes() == (
        tmp_path / "second" / "backbone.fp32.onnx"
    ).read_bytes()
    assert first["provenance"] == second["provenance"]


def test_matched_random_pair_rejects_uncontrolled_config_difference(tmp_path):
    from openlocalagent.inference.export.to_onnx import export_matched_random_backbones

    hybrid_config, attention_config = _write_matched_configs(tmp_path)
    attention = yaml.safe_load(attention_config.read_text())
    attention["rope_theta"] = 20000.0
    attention_config.write_text(yaml.safe_dump(attention, sort_keys=True))

    with pytest.raises(ValueError, match="must differ in exactly"):
        export_matched_random_backbones(
            str(hybrid_config),
            str(attention_config),
            str(tmp_path / "pair"),
            fp16=False,
        )
    assert not (tmp_path / "pair").exists()
