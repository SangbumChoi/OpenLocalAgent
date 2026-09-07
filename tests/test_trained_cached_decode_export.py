import copy
import hashlib
import json
import pickle
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import yaml

from openlocalagent.model import LocalAgentLM, ModelConfig
from openlocalagent.train.stage_data import (
    LINEAGE_VERSION,
    canonical_sha256,
    tokenizer_identity,
)

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")


def _write_unsafe_checkpoint_marker(path: str) -> dict:
    Path(path).write_text("executed", encoding="utf-8")
    return {}


class _UnsafeCheckpointPayload:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return _write_unsafe_checkpoint_marker, (str(self.marker),)


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
        name="trained-cached-hybrid",
        ffn_hidden=64,
        layer_types=["conv", "conv", "attn"],
        **common,
    )
    attention = ModelConfig(
        name="trained-cached-attention",
        ffn_hidden=75,
        layer_types=["attn", "attn", "attn"],
        **common,
    )
    hybrid_path = tmp_path / "hybrid.yaml"
    attention_path = tmp_path / "attention.yaml"
    hybrid_path.write_text(
        yaml.safe_dump(asdict(hybrid), sort_keys=True),
        encoding="utf-8",
    )
    attention_path.write_text(
        yaml.safe_dump(asdict(attention), sort_keys=True),
        encoding="utf-8",
    )
    return hybrid, hybrid_path, attention, attention_path


def _save_pretrain_checkpoint(path, cfg, *, token_sha256=None, state_dict=None):
    model = LocalAgentLM(cfg).eval()
    with torch.no_grad():
        model.embed.weight.add_(0.125)
    tokenizer = tokenizer_identity("byte", vocab_size=cfg.vocab_size)
    if token_sha256 is not None:
        tokenizer["sha256"] = token_sha256
    payload = {
        "cfg": asdict(cfg),
        "state_dict": state_dict if state_dict is not None else model.state_dict(),
        "stage": "pretrain",
        "step": 4,
        "loss_history": [1.0, 0.8, 0.6, 0.5, 0.4],
        "tokens_seen": 12_345,
        "input_tokens_seen": 12_400,
        "token_accounting": {
            "input_tokens": 12_400,
            "loss_tokens": 12_345,
        },
        "tokenizer": {
            "kind": "byte",
            "path": None,
            "sha256": tokenizer["sha256"],
        },
        "lineage": {
            "version": LINEAGE_VERSION,
            "stage": "pretrain",
            "config_sha256": canonical_sha256({"stage": "pretrain"}),
            "data_sha256": canonical_sha256({"fixture": "trained-cached-decode"}),
            "model_config_sha256": canonical_sha256(asdict(cfg)),
            "git": {
                "commit": "a" * 40,
                "repository_sha256": "c" * 64,
                "dirty": False,
                "worktree_sha256": "d" * 64,
            },
            "tokenizer_sha256": tokenizer_identity(
                "byte",
                vocab_size=cfg.vocab_size,
            )["sha256"],
        },
    }
    torch.save(payload, path)
    return model, payload


def _save_stage_checkpoint(path, cfg, stage, **extra):
    model, payload = _save_pretrain_checkpoint(path, cfg)
    payload["stage"] = stage
    payload["lineage"]["stage"] = stage
    payload["lineage"]["config_sha256"] = canonical_sha256({"stage": stage})
    if stage != "pretrain":
        payload["lineage"]["parent_checkpoint_sha256"] = "b" * 64
        payload["conversation_prompt_contract"] = "openai_full_catalog_v1"
    if stage == "rl":
        payload.pop("loss_history")
        payload.pop("token_accounting")
        payload.pop("tokens_seen")
        payload.pop("input_tokens_seen")
        payload["reward_history"] = [0.0] * 5
        payload["rl_accounting"] = {
            "attempted_rollout_steps": 5,
            "realized_optimizer_updates": 0,
        }
        payload["structured_heads_available"] = False
        payload["invalidated_structured_heads"] = []
    payload.update(extra)
    torch.save(payload, path)
    return model, payload


def _minimal_single_decode_provenance() -> dict:
    graph_identity = {"bytes": 101, "sha256": "a" * 64}
    return {
        "schema_version": 1,
        "artifact_type": "trained_checkpoint_cached_decode_onnx",
        "trained": True,
        "latency_only": False,
        "capability_artifact": False,
        "model": {
            "name": "trained-single-fixture",
            "pair_role": "hybrid_treatment",
        },
        "graph_contract": {
            "graphs": {
                "fp32": {
                    "prefill": {"file": "prefill.fp32.onnx"},
                    "decode": {"file": "decode.fp32.onnx"},
                },
            },
        },
        "artifacts": {
            "prefill.fp32.onnx": copy.deepcopy(graph_identity),
            "decode.fp32.onnx": copy.deepcopy(graph_identity),
        },
        "parity": {
            "hard_gate": True,
            "results": {
                "fp32": {
                    "artifacts": {
                        "prefill": copy.deepcopy(graph_identity),
                        "decode": copy.deepcopy(graph_identity),
                    },
                    "greedy_next_token_exact": True,
                    "hard_gate": True,
                    "passed": True,
                    "reference": "exact in-memory LocalAgentLM checkpoint weights",
                },
            },
        },
    }


def test_matched_trained_cached_decode_export_binds_checkpoint_and_quality_provenance(
    tmp_path,
):
    from openlocalagent.inference.export.to_onnx import (
        _state_dict_sha256,
        export_matched_cached_decode,
    )

    hybrid_cfg, hybrid_config, attention_cfg, attention_config = _write_matched_configs(tmp_path)
    hybrid_checkpoint = tmp_path / "hybrid.pt"
    attention_checkpoint = tmp_path / "attention.pt"
    hybrid_model, _ = _save_pretrain_checkpoint(hybrid_checkpoint, hybrid_cfg)
    attention_model, _ = _save_pretrain_checkpoint(attention_checkpoint, attention_cfg)

    result = export_matched_cached_decode(
        str(hybrid_config),
        str(attention_config),
        str(tmp_path / "pair"),
        hybrid_checkpoint_path=str(hybrid_checkpoint),
        attention_checkpoint_path=str(attention_checkpoint),
        fp16=False,
        fixture_lengths=(1, 5),
        decode_steps=3,
    )

    manifest = json.loads((tmp_path / "pair" / "matched-decode.json").read_text())
    assert result["manifest"] == manifest
    assert manifest["artifact_type"] == "matched_trained_cached_decode_suite"
    assert manifest["trained"] is True
    assert manifest["latency_only"] is False
    assert manifest["capability_artifact"] is False
    assert manifest["quality_claims"] == []
    assert manifest["quality_evaluation"] == {
        "included": False,
        "required_separately": True,
    }
    assert "shared_random_seed" not in manifest
    assert manifest["tokenizer"]["verified"] is True

    expected_models = {
        "hybrid": (hybrid_checkpoint, hybrid_model, "hybrid_treatment"),
        "attention": (
            attention_checkpoint,
            attention_model,
            "all_attention_control",
        ),
    }
    for role, (checkpoint, model, pair_role) in expected_models.items():
        provenance_path = tmp_path / "pair" / role / "provenance.json"
        provenance = json.loads(provenance_path.read_text())
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert provenance["artifact_type"] == "trained_checkpoint_cached_decode_onnx"
        assert provenance["trained"] is True
        assert provenance["latency_only"] is False
        assert provenance["capability_artifact"] is False
        assert provenance["checkpoint_step"] == 4
        assert provenance["training_steps"] == 5
        assert provenance["tokens_seen"] == 12_345
        assert provenance["input_tokens_seen"] == 12_400
        assert provenance["quality_evaluation"]["included"] is False
        assert provenance["quality_evaluation"]["required_separately"] is True
        assert provenance["model"]["pair_role"] == pair_role
        assert provenance["weights"]["checkpoint_sha256"] == checkpoint_sha256
        assert provenance["weights"]["checkpoint_step"] == 4
        assert provenance["weights"]["tokens_seen"] == 12_345
        assert provenance["weights"]["state_dict_sha256"] == _state_dict_sha256(model)
        assert provenance["parity"]["results"]["fp32"]["passed"] is True
        assert provenance["parity"]["results"]["fp32"]["greedy_next_token_exact"] is True
        assert provenance["parity"]["results"]["fp32"]["final_token_logits_shape"] == [
            "batch",
            hybrid_cfg.vocab_size,
        ]
        assert (
            provenance["parity"]["results"]["fp32"]["max_logits_abs_diff"]
            <= provenance["parity"]["results"]["fp32"]["cache_atol"]
        )
        metadata = json.loads((provenance_path.parent / "meta.json").read_text())
        assert metadata["artifact_type"] == "localagent_cached_autoregressive_onnx"
        assert metadata["checkpoint"]["stage"] == "pretrain"
        assert metadata["checkpoint"]["lineage"]["version"] == LINEAGE_VERSION
        assert metadata["model"]["config_file"] == "model-config.yaml"
        assert metadata["tokenizer"]["encoding"] == "utf-8-bytes"
        assert metadata["tokenizer"]["verified"] is True
        precision_graphs = metadata["graph_contract"]["graphs"]["fp32"]
        for graph in (precision_graphs["prefill"], precision_graphs["decode"]):
            assert graph["output_names"][:2] == ["next_token", "logits"]
            assert graph["outputs"][1] == {
                "dtype": "float32",
                "name": "logits",
                "shape": ["batch", "vocab_size"],
            }
        assert set(provenance["artifacts"]) == {
            "decode.fp32.onnx",
            "meta.json",
            "model-config.yaml",
            "prefill.fp32.onnx",
        }
        for artifact_name, identity in provenance["artifacts"].items():
            artifact_path = provenance_path.parent / artifact_name
            assert identity["bytes"] == artifact_path.stat().st_size
            assert identity["sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        pair_provenance = manifest["artifacts"][f"{role}/provenance.json"]
        assert pair_provenance["bytes"] == provenance_path.stat().st_size
        assert pair_provenance["sha256"] == hashlib.sha256(provenance_path.read_bytes()).hexdigest()
        single_manifest_path = provenance_path.parent / "single-decode.json"
        single_manifest = json.loads(single_manifest_path.read_text())
        assert result[role]["single_manifest"] == single_manifest
        assert result[role]["single_manifest_path"] == str(single_manifest_path)
        assert single_manifest == {
            "artifact_type": "single_trained_cached_decode_suite",
            "artifacts": {
                "provenance.json": {
                    "bytes": provenance_path.stat().st_size,
                    "sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
                },
            },
            "capability_artifact": False,
            "latency_only": False,
            "model": {
                "name": provenance["model"]["name"],
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


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_single_decode_wrapper_is_self_consistent_and_matches_browser_contract(tmp_path):
    from openlocalagent.inference.export.to_onnx import (
        _write_single_trained_cached_decode_manifest,
    )

    output = tmp_path / "single"
    output.mkdir()
    provenance = _minimal_single_decode_provenance()
    provenance_path = output / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest, manifest_path = _write_single_trained_cached_decode_manifest(
        output,
        provenance,
    )
    first_manifest_bytes = manifest_path.read_bytes()
    repeated_manifest, repeated_path = _write_single_trained_cached_decode_manifest(
        output,
        provenance,
    )
    assert repeated_path == manifest_path
    assert repeated_manifest == manifest
    assert manifest_path.read_bytes() == first_manifest_bytes
    assert json.loads(first_manifest_bytes) == manifest
    assert manifest["model"] == {
        "name": provenance["model"]["name"],
        "pair_role": provenance["model"]["pair_role"],
        "provenance": "provenance.json",
    }
    assert manifest["artifacts"] == {
        "provenance.json": {
            "bytes": provenance_path.stat().st_size,
            "sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
        },
    }

    decode_js = Path(__file__).parents[1] / "demos" / "webgpu" / "decode-benchmark.js"
    script = """
global.window = { __localAgentSkipInit: true };
const api = require(process.argv[1]);
const manifest = JSON.parse(process.argv[2]);
const accepted = api.validateSingleDecodeManifest(manifest);
process.stdout.write(JSON.stringify({
  artifactType: accepted.artifact_type,
  model: accepted.model,
}));
"""
    result = subprocess.run(
        [
            shutil.which("node"),
            "-e",
            script,
            str(decode_js),
            json.dumps(manifest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    accepted = json.loads(result.stdout)
    assert accepted == {
        "artifactType": "single_trained_cached_decode_suite",
        "model": manifest["model"],
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda provenance: provenance["parity"].update(hard_gate=False),
            "hard trajectory parity",
        ),
        (
            lambda provenance: provenance["parity"]["results"]["fp32"].update(passed=False),
            "trajectory parity failed",
        ),
        (
            lambda provenance: provenance["parity"]["results"]["fp32"]["artifacts"][
                "decode"
            ].update(sha256="b" * 64),
            "not bound to trajectory parity",
        ),
        (
            lambda provenance: provenance.update(trained=False),
            "non-trained cached provenance",
        ),
    ],
)
def test_single_decode_wrapper_fails_closed_before_publication(
    tmp_path,
    mutate,
    message,
):
    from openlocalagent.inference.export.to_onnx import (
        _write_single_trained_cached_decode_manifest,
    )

    provenance = _minimal_single_decode_provenance()
    mutate(provenance)
    output = tmp_path / hashlib.sha256(message.encode()).hexdigest()
    output.mkdir()
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=message):
        _write_single_trained_cached_decode_manifest(output, provenance)
    assert not (output / "single-decode.json").exists()
    assert not (output / "single-decode.json.tmp").exists()


@pytest.mark.parametrize("stage", ["pretrain", "midtrain", "sft", "rl"])
def test_cached_decode_loader_accepts_lineage_validated_lm_stages(tmp_path, stage):
    from openlocalagent.inference.export.to_onnx import (
        _load_cached_decode_checkpoint,
        _training_lineage_export,
    )

    cfg, _, _, _ = _write_matched_configs(tmp_path)
    checkpoint_path = tmp_path / f"{stage}.pt"
    _save_stage_checkpoint(checkpoint_path, cfg, stage)

    _, checkpoint = _load_cached_decode_checkpoint(
        cfg,
        str(checkpoint_path),
        tokenizer_path=None,
    )

    assert checkpoint["stage"] == stage
    assert checkpoint["lineage"]["stage"] == stage
    assert checkpoint["lineage"]["version"] == LINEAGE_VERSION
    assert checkpoint["training_steps"] == 5
    sidecar = _training_lineage_export(checkpoint, ["e" * 64])
    assert sidecar["kind"] == "localagent_training_lineage_export"
    assert sidecar["stage"] == stage
    assert sidecar["checkpoint_sha256"] == checkpoint["checkpoint_sha256"]
    assert sidecar["training_artifact_sha256"] == ["e" * 64]
    assert sidecar["conversation_prompt_contract"] == (
        None if stage == "pretrain" else "openai_full_catalog_v1"
    )
    assert ("parent_checkpoint_sha256" in sidecar["lineage"]) == (stage != "pretrain")
    if stage == "rl":
        assert checkpoint["tokens_seen"] is None
        assert checkpoint["rl_accounting"]["realized_optimizer_updates"] == 0
        assert checkpoint["auxiliary_heads"]["available"] is False


def test_cached_decode_loader_rejects_pickle_code_execution(tmp_path):
    from openlocalagent.inference.export.to_onnx import _load_cached_decode_checkpoint

    cfg, _, _, _ = _write_matched_configs(tmp_path)
    checkpoint_path = tmp_path / "unsafe.pt"
    marker = tmp_path / "pickle-executed"
    torch.save(_UnsafeCheckpointPayload(marker), checkpoint_path)

    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        _load_cached_decode_checkpoint(
            cfg,
            str(checkpoint_path),
            tokenizer_path=None,
        )
    assert not marker.exists()


def test_cached_decode_loader_rejects_symlink_checkpoint(tmp_path):
    from openlocalagent.inference.export.to_onnx import _load_cached_decode_checkpoint

    cfg, _, _, _ = _write_matched_configs(tmp_path)
    checkpoint_path = tmp_path / "checkpoint.pt"
    _save_stage_checkpoint(checkpoint_path, cfg, "pretrain")
    symlink_path = tmp_path / "checkpoint-link.pt"
    symlink_path.symlink_to(checkpoint_path)

    with pytest.raises(ValueError, match="non-symlink regular file"):
        _load_cached_decode_checkpoint(
            cfg,
            str(symlink_path),
            tokenizer_path=None,
        )


def test_cached_decode_loader_validates_but_does_not_attach_sft_heads(tmp_path):
    from openlocalagent.agent.routes import RouteHead
    from openlocalagent.inference.export.to_onnx import _load_cached_decode_checkpoint

    cfg, _, _, _ = _write_matched_configs(tmp_path)
    checkpoint_path = tmp_path / "sft.pt"
    _save_stage_checkpoint(
        checkpoint_path,
        cfg,
        "sft",
        route_head=RouteHead(cfg.d_model).state_dict(),
    )

    _, checkpoint = _load_cached_decode_checkpoint(
        cfg,
        str(checkpoint_path),
        tokenizer_path=None,
    )

    heads = checkpoint["auxiliary_heads"]
    assert heads["available"] is True
    assert heads["exported"] is False
    assert heads["heads"]["route_head"]["validated"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            {"value_head": {"weight": torch.zeros(1, 1)}},
            "does not support checkpoint value/reward heads",
        ),
        (
            {
                "route_head": {"fc.weight": torch.zeros(1, 1)},
            },
            "route_head is incompatible",
        ),
    ],
)
def test_cached_decode_loader_rejects_incompatible_auxiliary_heads(
    tmp_path,
    mutation,
    message,
):
    from openlocalagent.inference.export.to_onnx import _load_cached_decode_checkpoint

    cfg, _, _, _ = _write_matched_configs(tmp_path)
    checkpoint_path = tmp_path / "sft.pt"
    _save_stage_checkpoint(checkpoint_path, cfg, "sft", **mutation)

    with pytest.raises(ValueError, match=message):
        _load_cached_decode_checkpoint(
            cfg,
            str(checkpoint_path),
            tokenizer_path=None,
        )


def test_cached_decode_loader_rejects_stale_heads_in_rl_checkpoint(tmp_path):
    from openlocalagent.agent.routes import RouteHead
    from openlocalagent.inference.export.to_onnx import _load_cached_decode_checkpoint

    cfg, _, _, _ = _write_matched_configs(tmp_path)
    checkpoint_path = tmp_path / "rl.pt"
    _save_stage_checkpoint(
        checkpoint_path,
        cfg,
        "rl",
        route_head=RouteHead(cfg.d_model).state_dict(),
        invalidated_structured_heads=["route_head"],
    )

    with pytest.raises(ValueError, match="carries stale structured heads"):
        _load_cached_decode_checkpoint(
            cfg,
            str(checkpoint_path),
            tokenizer_path=None,
        )


@pytest.mark.parametrize(
    ("lineage_update", "message"),
    [
        ({"version": LINEAGE_VERSION + 1}, "lineage version is unsupported"),
        ({"stage": "pretrain"}, "stage and lineage stage disagree"),
        ({"parent_checkpoint_sha256": None}, "parent_checkpoint_sha256"),
    ],
)
def test_cached_decode_loader_rejects_invalid_posttraining_lineage(
    tmp_path,
    lineage_update,
    message,
):
    from openlocalagent.inference.export.to_onnx import _load_cached_decode_checkpoint

    cfg, _, _, _ = _write_matched_configs(tmp_path)
    checkpoint_path = tmp_path / "sft.pt"
    _, payload = _save_stage_checkpoint(checkpoint_path, cfg, "sft")
    payload["lineage"].update(lineage_update)
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match=message):
        _load_cached_decode_checkpoint(
            cfg,
            str(checkpoint_path),
            tokenizer_path=None,
        )


def test_matched_checkpoint_export_requires_both_arms_before_writing(tmp_path):
    from openlocalagent.inference.export.to_onnx import export_matched_cached_decode

    hybrid_cfg, hybrid_config, _, attention_config = _write_matched_configs(tmp_path)
    hybrid_checkpoint = tmp_path / "hybrid.pt"
    _save_pretrain_checkpoint(hybrid_checkpoint, hybrid_cfg)
    output = tmp_path / "pair"

    with pytest.raises(ValueError, match="requires both"):
        export_matched_cached_decode(
            str(hybrid_config),
            str(attention_config),
            str(output),
            hybrid_checkpoint_path=str(hybrid_checkpoint),
            fp16=False,
        )
    assert not output.exists()


def test_trained_cached_decode_rejects_architecture_and_vocab_mismatch_before_writing(
    tmp_path,
):
    from openlocalagent.inference.export.to_onnx import export_cached_decode

    hybrid_cfg, hybrid_config, _, _ = _write_matched_configs(tmp_path)
    wrong_cfg = ModelConfig(**{**asdict(hybrid_cfg), "vocab_size": 257})
    checkpoint = tmp_path / "wrong.pt"
    _save_pretrain_checkpoint(checkpoint, wrong_cfg)
    output = tmp_path / "export"

    with pytest.raises(ValueError, match="architecture/vocabulary"):
        export_cached_decode(
            str(hybrid_config),
            str(output),
            pair_role="hybrid_treatment",
            checkpoint_path=str(checkpoint),
            fp16=False,
            fixture_lengths=(1, 5),
        )
    assert not output.exists()


def test_trained_cached_decode_rejects_tampered_tokenizer_lineage_before_writing(
    tmp_path,
):
    from openlocalagent.inference.export.to_onnx import export_cached_decode

    hybrid_cfg, hybrid_config, _, _ = _write_matched_configs(tmp_path)
    checkpoint = tmp_path / "wrong-tokenizer.pt"
    _save_pretrain_checkpoint(checkpoint, hybrid_cfg, token_sha256="a" * 64)
    output = tmp_path / "export"

    with pytest.raises(ValueError, match="tokenizer metadata and lineage hashes disagree"):
        export_cached_decode(
            str(hybrid_config),
            str(output),
            pair_role="hybrid_treatment",
            checkpoint_path=str(checkpoint),
            fp16=False,
            fixture_lengths=(1, 5),
        )
    assert not output.exists()


def test_trained_cached_decode_verifies_recorded_bpe_artifact_and_vocabulary(tmp_path):
    from openlocalagent.inference.export.to_onnx import _load_cached_decode_checkpoint
    from openlocalagent.model.tokenizer import train_bpe
    from openlocalagent.train.stage_data import sha256_file

    tokenizer_path = tmp_path / "tokenizer.json"
    corpus = [
        "Open src/app.py and summarize the implementation.",
        "Search for the browser action handler and run focused tests.",
        "<|user|>move report.md to archive/report.md<|assistant|>",
    ] * 20
    tokenizer = train_bpe(
        corpus,
        tokenizer_path,
        vocab_size=320,
        min_frequency=1,
    )
    cfg = ModelConfig(
        name="trained-bpe",
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=1,
        ffn_hidden=64,
        max_seq_len=32,
        qk_norm=True,
    )
    checkpoint_path = tmp_path / "bpe.pt"
    _, payload = _save_pretrain_checkpoint(checkpoint_path, cfg)
    tokenizer_sha256 = sha256_file(tokenizer_path)
    payload["tokenizer"] = {
        "kind": "bpe",
        "path": str(tokenizer_path),
        "sha256": tokenizer_sha256,
    }
    payload["lineage"]["tokenizer_sha256"] = tokenizer_sha256
    torch.save(payload, checkpoint_path)

    _, checkpoint = _load_cached_decode_checkpoint(
        cfg,
        str(checkpoint_path),
        tokenizer_path=None,
    )
    assert checkpoint["tokenizer"]["verified"] is True
    assert checkpoint["tokenizer"]["vocab_size"] == tokenizer.vocab_size
    assert checkpoint["tokenizer"]["artifact_identity"]["sha256"] == tokenizer_sha256

    config_path = tmp_path / "bpe.yaml"
    config_path.write_text(
        yaml.safe_dump(asdict(cfg), sort_keys=True),
        encoding="utf-8",
    )
    from openlocalagent.inference.export.to_onnx import export_cached_decode

    result = export_cached_decode(
        str(config_path),
        str(tmp_path / "bpe-export"),
        pair_role="bpe_fixture",
        checkpoint_path=str(checkpoint_path),
        fp16=False,
        fixture_lengths=(1, 5),
        decode_steps=3,
        training_artifact_sha256=["e" * 64],
    )
    copied_tokenizer = tmp_path / "bpe-export" / "tokenizer.json"
    metadata = json.loads((tmp_path / "bpe-export" / "meta.json").read_text())
    assert copied_tokenizer.read_bytes() == tokenizer_path.read_bytes()
    assert metadata["tokenizer"]["file"] == "tokenizer.json"
    assert metadata["tokenizer"]["sha256"] == tokenizer_sha256
    assert metadata["tokenizer"]["encoding"] == "bytelevel-bpe"
    assert metadata["tokenizer_file"] == "tokenizer.json"
    assert result["provenance"]["artifacts"]["tokenizer.json"]["sha256"] == tokenizer_sha256
    lineage_path = tmp_path / "bpe-export" / "training-lineage.json"
    lineage_export = json.loads(lineage_path.read_text())
    assert lineage_export == {
        "kind": "localagent_training_lineage_export",
        "schema_version": 1,
        "stage": "pretrain",
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "lineage": payload["lineage"],
        "training_artifact_sha256": ["e" * 64],
        "conversation_prompt_contract": None,
    }
    assert metadata["checkpoint"]["lineage_export"]["file"] == "training-lineage.json"
    assert result["training_lineage_path"] == str(lineage_path)

    wrong_tokenizer = tmp_path / "wrong-tokenizer.json"
    wrong_tokenizer.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 does not match"):
        _load_cached_decode_checkpoint(
            cfg,
            str(checkpoint_path),
            tokenizer_path=str(wrong_tokenizer),
        )


def test_trained_cached_decode_strictly_rejects_missing_state_key_before_writing(
    tmp_path,
):
    from openlocalagent.inference.export.to_onnx import export_cached_decode

    hybrid_cfg, hybrid_config, _, _ = _write_matched_configs(tmp_path)
    state_dict = dict(LocalAgentLM(hybrid_cfg).state_dict())
    state_dict.pop(next(iter(state_dict)))
    checkpoint = tmp_path / "missing-key.pt"
    _save_pretrain_checkpoint(checkpoint, hybrid_cfg, state_dict=state_dict)
    output = tmp_path / "export"

    with pytest.raises(ValueError, match="does not strictly match"):
        export_cached_decode(
            str(hybrid_config),
            str(output),
            pair_role="hybrid_treatment",
            checkpoint_path=str(checkpoint),
            fp16=False,
            fixture_lengths=(1, 5),
        )
    assert not output.exists()
