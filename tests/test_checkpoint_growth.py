import hashlib
import os

import pytest
import torch
import yaml

from openlocalagent.cli import main as cli_main
from openlocalagent.data.corpus.pretrain_corpus import pack_shards
from openlocalagent.model import LocalAgentLM, ModelConfig
from openlocalagent.model.tokenizer import ByteTokenizer
from openlocalagent.train.checkpoint_growth import (
    assert_growth_compatible,
    grow_checkpoint,
    parse_layer_map,
    verify_growth_checkpoint,
    write_grown_checkpoint,
)
from openlocalagent.train.pretrain import run as run_pretrain
from openlocalagent.train.stage_data import canonical_sha256, sha256_file, tokenizer_identity

TOKENIZER_SHA256 = "a" * 64


def _cfg(
    name: str,
    n_layers: int,
    *,
    layer_types: list[str] | None = None,
    d_model: int = 16,
    embed_dim: int | None = None,
    tie_embeddings: bool = True,
) -> ModelConfig:
    return ModelConfig(
        name=name,
        vocab_size=256,
        d_model=d_model,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=32,
        max_seq_len=8,
        layer_types=layer_types,
        tie_embeddings=tie_embeddings,
    )


def _checkpoint(cfg: ModelConfig, *, tokenizer_sha256: str = TOKENIZER_SHA256) -> dict:
    torch.manual_seed(7)
    model = LocalAgentLM(cfg)
    with torch.no_grad():
        for layer, block in enumerate(model.blocks):
            for parameter in block.parameters():
                parameter.fill_(layer + 1)
    return {
        "cfg": dict(cfg.__dict__),
        "state_dict": model.state_dict(),
        "optimizer": {"state": {"sentinel": "must be discarded"}},
        "step": 123,
        "tokenizer": {"kind": "byte", "sha256": tokenizer_sha256},
        "lineage": {
            "model_config_sha256": canonical_sha256(cfg.__dict__),
            "tokenizer_sha256": tokenizer_sha256,
        },
        "stage": "pretrain",
    }


def _mapped_source_key(target_key: str, layer_map: dict[int, int]) -> str:
    parts = target_key.split(".", 2)
    if len(parts) == 3 and parts[0] == "blocks":
        return f"blocks.{layer_map[int(parts[1])]}.{parts[2]}"
    return target_key


def test_grow_checkpoint_cli_writes_audited_target_state_and_discards_optimizer(
    tmp_path,
    capsys,
):
    source_cfg = _cfg("source", 2, layer_types=["attn", "conv"])
    target_cfg = _cfg("target", 4, layer_types=["attn", "attn", "conv", "conv"])
    source = tmp_path / "source.pt"
    target_yaml = tmp_path / "target.yaml"
    output = tmp_path / "grown.pt"
    source_checkpoint = _checkpoint(source_cfg)
    torch.save(source_checkpoint, source)
    target_yaml.write_text(yaml.safe_dump(target_cfg.__dict__), encoding="utf-8")
    layer_map = {0: 0, 1: 0, 2: 1, 3: 1}

    cli_main(
        [
            "grow-checkpoint",
            str(source),
            str(target_yaml),
            str(output),
            "--layer-map",
            "0:0,1:0,2:1,3:1",
        ]
    )

    assert str(output) in capsys.readouterr().out
    grown = torch.load(output, map_location="cpu", weights_only=False)
    assert set(grown) == {"cfg", "growth", "stage", "state_dict", "tokenizer"}
    assert grown["stage"] == "checkpoint_growth"
    assert "optimizer" not in grown
    assert grown["cfg"] == target_cfg.__dict__
    LocalAgentLM(target_cfg).load_state_dict(grown["state_dict"], strict=True)
    for target_key, target_tensor in grown["state_dict"].items():
        source_key = _mapped_source_key(target_key, layer_map)
        assert torch.equal(target_tensor, source_checkpoint["state_dict"][source_key])

    manifest = grown["growth"]
    assert manifest["source_checkpoint_sha256"] == sha256_file(source)
    assert manifest["target_layer_to_source_layer"] == [
        {"target_layer": 0, "source_layer": 0},
        {"target_layer": 1, "source_layer": 0},
        {"target_layer": 2, "source_layer": 1},
        {"target_layer": 3, "source_layer": 1},
    ]
    assert manifest["function_preserving"] is False
    assert manifest["optimizer_state"] == "discarded"
    assert "NOT function-preserving" in manifest["warning"]
    assert {
        "lineage",
        "optimizer",
        "stage",
        "step",
    }.issubset(manifest["discarded_payload_keys"])
    core = dict(manifest)
    recorded_manifest_sha256 = core.pop("manifest_sha256")
    assert recorded_manifest_sha256 == canonical_sha256(core)
    assert verify_growth_checkpoint(grown) == manifest

    repeated_output = tmp_path / "grown-again.pt"
    cli_main(
        [
            "grow-checkpoint",
            str(source),
            str(target_yaml),
            str(repeated_output),
            "--layer-map",
            "0:0,1:0,2:1,3:1",
        ]
    )
    capsys.readouterr()
    assert repeated_output.read_bytes() == output.read_bytes()


def test_growth_rejects_architecture_or_mapped_block_kind_changes():
    source = _cfg("source", 2, layer_types=["attn", "conv"])
    wider = _cfg(
        "wider",
        3,
        layer_types=["attn", "conv", "conv"],
        d_model=32,
    )
    with pytest.raises(ValueError, match="growth model configs are incompatible"):
        assert_growth_compatible(source, wider, {0: 0, 1: 1, 2: 1})

    wrong_kind = _cfg("wrong-kind", 3, layer_types=["attn", "conv", "attn"])
    with pytest.raises(ValueError, match="mapped block kinds must match"):
        assert_growth_compatible(source, wrong_kind, {0: 0, 1: 1, 2: 1})


def test_growth_requires_an_explicit_complete_map_and_proven_tokenizer(tmp_path):
    source_cfg = _cfg("source", 1)
    target_cfg = _cfg("target", 2)
    with pytest.raises(ValueError, match="name every target layer"):
        assert_growth_compatible(source_cfg, target_cfg, {0: 0})
    with pytest.raises(ValueError, match="appears more than once"):
        parse_layer_map("0:0,0:0")

    source_checkpoint = _checkpoint(source_cfg)
    source_checkpoint.pop("tokenizer")
    source_checkpoint.pop("lineage")
    source_path = tmp_path / "unbound-tokenizer.pt"
    torch.save(source_checkpoint, source_path)
    with pytest.raises(ValueError, match="no content-bound tokenizer identity"):
        grow_checkpoint(source_path, target_cfg, {0: 0, 1: 0})


def test_growth_verifier_rejects_tampered_manifest_or_target_state(tmp_path):
    source_cfg = _cfg("source", 1)
    target_cfg = _cfg("target", 2)
    source_path = tmp_path / "source.pt"
    torch.save(_checkpoint(source_cfg), source_path)
    grown = grow_checkpoint(source_path, target_cfg, {0: 0, 1: 0})

    grown["growth"]["warning"] = "tampered"
    with pytest.raises(ValueError, match="self-hash mismatch"):
        verify_growth_checkpoint(grown)

    grown = grow_checkpoint(source_path, target_cfg, {0: 0, 1: 0})
    first_key = next(iter(grown["state_dict"]))
    grown["state_dict"][first_key] = grown["state_dict"][first_key].clone()
    grown["state_dict"][first_key].view(-1)[0] += 1
    with pytest.raises(ValueError, match="target-state hash mismatch"):
        verify_growth_checkpoint(grown)


def test_growth_rejects_posttraining_auxiliary_state_or_incomplete_cfg(tmp_path):
    source_cfg = _cfg("source", 1)
    target_cfg = _cfg("target", 2)
    source_checkpoint = _checkpoint(source_cfg)
    source_checkpoint["stage"] = "sft"
    source_checkpoint["tool_head"] = {"weight": torch.ones(1)}
    source_path = tmp_path / "sft.pt"
    torch.save(source_checkpoint, source_path)
    with pytest.raises(ValueError, match="pretrain/midtrain backbone"):
        grow_checkpoint(source_path, target_cfg, {0: 0, 1: 0})

    source_checkpoint = _checkpoint(source_cfg)
    del source_checkpoint["cfg"]["rope_theta"]
    source_path = tmp_path / "incomplete.pt"
    torch.save(source_checkpoint, source_path)
    with pytest.raises(ValueError, match="complete current schema"):
        grow_checkpoint(source_path, target_cfg, {0: 0, 1: 0})


def test_growth_supports_factorized_untied_embeddings_and_verified_chaining(tmp_path):
    source_cfg = _cfg(
        "source",
        1,
        embed_dim=8,
        tie_embeddings=False,
    )
    middle_cfg = _cfg(
        "middle",
        2,
        embed_dim=8,
        tie_embeddings=False,
    )
    target_cfg = _cfg(
        "target",
        3,
        embed_dim=8,
        tie_embeddings=False,
    )
    source_path = tmp_path / "source.pt"
    middle_path = tmp_path / "middle.pt"
    torch.save(_checkpoint(source_cfg), source_path)
    write_grown_checkpoint(source_path, middle_cfg, {0: 0, 1: 0}, middle_path)
    chained = grow_checkpoint(middle_path, target_cfg, {0: 0, 1: 0, 2: 1})

    assert verify_growth_checkpoint(chained)["source_stage"] == "checkpoint_growth"
    LocalAgentLM(target_cfg).load_state_dict(chained["state_dict"], strict=True)


def test_growth_non_overwrite_publication_does_not_clobber_a_racing_creator(
    tmp_path,
    monkeypatch,
):
    source_cfg = _cfg("source", 1)
    target_cfg = _cfg("target", 2)
    source_path = tmp_path / "source.pt"
    output = tmp_path / "grown.pt"
    torch.save(_checkpoint(source_cfg), source_path)
    original_link = os.link

    def racing_link(source, destination):
        destination_path = destination if isinstance(destination, str) else str(destination)
        with open(destination_path, "wb") as handle:
            handle.write(b"concurrent creator")
        return original_link(source, destination)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(FileExistsError, match="already exists"):
        write_grown_checkpoint(source_path, target_cfg, {0: 0, 1: 0}, output)
    assert output.read_bytes() == b"concurrent creator"


def test_config_pretrain_init_from_growth_uses_fresh_optimizer_and_parent_hash(tmp_path):
    tokenizer_sha256 = str(tokenizer_identity("byte", vocab_size=256)["sha256"])
    source_cfg = _cfg("source", 1)
    target_cfg = _cfg("target", 2)
    source = tmp_path / "source.pt"
    grown_path = tmp_path / "grown.pt"
    model_path = tmp_path / "target.yaml"
    torch.save(_checkpoint(source_cfg, tokenizer_sha256=tokenizer_sha256), source)
    write_grown_checkpoint(source, target_cfg, {0: 0, 1: 0}, grown_path)
    grown = torch.load(grown_path, map_location="cpu", weights_only=False)
    assert "optimizer" not in grown

    model_path.write_text(yaml.safe_dump(target_cfg.__dict__), encoding="utf-8")
    shards = tmp_path / "shards"
    pack_shards(
        ["compatible checkpoint growth warm start " * 8],
        ByteTokenizer(),
        seq_len=8,
        shards_dir=str(shards),
        rows_per_shard=8,
        val_fraction=0.0,
    )
    out_dir = tmp_path / "run"
    config_path = tmp_path / "pretrain.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "stage": "pretrain",
                "model_config": str(model_path),
                "init_from": str(grown_path),
                "data": {
                    "shards_dir": str(shards),
                    "tokenizer": {"kind": "byte"},
                },
                "optim": {"lr": 0.0, "weight_decay": 0.0, "grad_clip": 1.0},
                "schedule": {
                    "type": "cosine",
                    "warmup_steps": 0,
                    "total_steps": 1,
                },
                "batch": {"micro_batch_size": 1, "grad_accum_steps": 1},
                "runtime": {
                    "device": "cpu",
                    "dtype": "fp32",
                    "seed": 23,
                    "resume": True,
                },
                "log": {"out_dir": str(out_dir)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    run_pretrain(str(config_path))

    saved = torch.load(out_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert saved["lineage"]["parent_checkpoint_sha256"] == sha256_file(grown_path)
    assert saved["step"] == 0
    for key, tensor in grown["state_dict"].items():
        assert torch.equal(saved["state_dict"][key], tensor), key
    optimizer_steps = {
        float(state["step"])
        for state in saved["optimizer"]["state"].values()
        if "step" in state
    }
    assert optimizer_steps == {1.0}
    assert hashlib.sha256(grown_path.read_bytes()).hexdigest() == (
        saved["lineage"]["parent_checkpoint_sha256"]
    )

    alias_parent = tmp_path / "mirror-parent" / "latest.pt"
    write_grown_checkpoint(source, target_cfg, {0: 0, 1: 0}, alias_parent)
    alias_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    alias_config["init_from"] = str(alias_parent)
    alias_config["log"] = {
        "out_dir": str(tmp_path / "alias-run"),
        "mirror_dir": str(alias_parent.parent),
    }
    config_path.write_text(
        yaml.safe_dump(alias_config, sort_keys=False),
        encoding="utf-8",
    )
    original_parent = hashlib.sha256(alias_parent.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="mirror checkpoint"):
        run_pretrain(str(config_path))
    assert hashlib.sha256(alias_parent.read_bytes()).hexdigest() == original_parent
