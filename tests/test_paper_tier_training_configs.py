from __future__ import annotations

import copy
from pathlib import Path

import yaml

from openlocalagent.model import ModelConfig

ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_ROOT = ROOT / "configs" / "pretrain"
DATA_ROOT = ROOT / "configs" / "data"
SEQ_LEN = 2048
TOKENS_PER_UPDATE = 2 * 8 * SEQ_LEN
MINIMUM_TOKENS = 19_628_032


def _load(name: str) -> dict:
    return yaml.safe_load((PRETRAIN_ROOT / name).read_text(encoding="utf-8"))


def _without_pair_identity(config: dict) -> dict:
    comparable = copy.deepcopy(config)
    comparable.pop("model_config")
    comparable["log"].pop("out_dir")
    return comparable


def test_shared_paper_tiers_bind_one_tokenizer_corpus_and_freeze() -> None:
    names = [
        "pretrain-paper-tier-1m.yaml",
        "pretrain-paper-tier-10m-hybrid.yaml",
        "pretrain-paper-tier-10m-attn.yaml",
        "pretrain-paper-tier-96m-hybrid.yaml",
        "pretrain-paper-tier-96m-attn.yaml",
    ]
    configs = [_load(name) for name in names]

    for config in configs:
        assert config["stage"] == "pretrain"
        assert config["data"] == {
            "shards_dir": "data/shards/paper-all",
            "min_train_tokens": MINIMUM_TOKENS,
            "corpus_freeze": {
                "spec": "configs/data/pretrain-paper-tier-freeze.yaml",
                "path": "data/shards/paper-all/tier-freeze.json",
            },
            "tokenizer": {
                "kind": "bpe",
                "path": "data/tokenizer-paper-16k.json",
            },
        }
        model = ModelConfig.from_yaml(str(ROOT / config["model_config"]))
        model.assert_within_budget()
        assert model.vocab_size == 16_384
        assert model.max_seq_len >= SEQ_LEN
        scheduled = (
            config["schedule"]["total_steps"]
            * config["batch"]["micro_batch_size"]
            * config["batch"]["grad_accum_steps"]
            * SEQ_LEN
        )
        assert scheduled >= MINIMUM_TOKENS


def test_tier_budgets_are_explicit_and_honest() -> None:
    cases = {
        "pretrain-paper-tier-1m.yaml": (20.0, 20.1),
        "pretrain-paper-tier-10m-hybrid.yaml": (5.0, 5.1),
        "pretrain-paper-tier-10m-attn.yaml": (5.0, 5.1),
        "pretrain-paper-tier-96m-hybrid.yaml": (1.0, 1.01),
        "pretrain-paper-tier-96m-attn.yaml": (1.0, 1.01),
    }
    for name, (minimum_tpp, maximum_tpp) in cases.items():
        config = _load(name)
        model = ModelConfig.from_yaml(str(ROOT / config["model_config"]))
        scheduled = config["schedule"]["total_steps"] * TOKENS_PER_UPDATE
        tokens_per_parameter = scheduled / model.estimate_params()
        assert minimum_tpp <= tokens_per_parameter < maximum_tpp


def test_tier_control_pairs_change_only_architecture_and_output() -> None:
    for tier in ("10m", "96m"):
        hybrid = _load(f"pretrain-paper-tier-{tier}-hybrid.yaml")
        attention = _load(f"pretrain-paper-tier-{tier}-attn.yaml")
        assert _without_pair_identity(hybrid) == _without_pair_identity(attention)


def test_tier_freeze_names_every_consumer_once() -> None:
    freeze = yaml.safe_load(
        (DATA_ROOT / "pretrain-paper-tier-freeze.yaml").read_text(encoding="utf-8")
    )
    consumers = freeze["training_configs"]
    assert len(consumers) == len(set(consumers)) == 5
    assert set(consumers) == {
        "configs/pretrain/pretrain-paper-tier-1m.yaml",
        "configs/pretrain/pretrain-paper-tier-10m-hybrid.yaml",
        "configs/pretrain/pretrain-paper-tier-10m-attn.yaml",
        "configs/pretrain/pretrain-paper-tier-96m-hybrid.yaml",
        "configs/pretrain/pretrain-paper-tier-96m-attn.yaml",
    }
    assert freeze["tokenizer"] == {
        "kind": "bpe",
        "path": "data/tokenizer-paper-16k.json",
        "vocab_size": 16_384,
    }
    assert freeze["expected"] == {
        "seq_len": SEQ_LEN,
        "min_train_tokens": MINIMUM_TOKENS,
        "tokenizer_training_split": "train",
    }
