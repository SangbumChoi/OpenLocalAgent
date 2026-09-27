from __future__ import annotations

import copy
from pathlib import Path

import yaml

from localagent.model import ModelConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "train"
SEEDS = (2026, 2027, 2028, 2029)
SEQ_LEN = 2048
TOTAL_STEPS = 322
MICRO_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
SCHEDULED_TOKENS = TOTAL_STEPS * MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS * SEQ_LEN


def _load(architecture: str, seed: int) -> dict:
    path = (
        CONFIG_ROOT
        / f"pretrain-webgpu-proxy-1tpp-{architecture}-seed{seed}.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _without_pair_identity(config: dict) -> dict:
    comparable = copy.deepcopy(config)
    comparable.pop("model_config")
    comparable["log"].pop("out_dir")
    return comparable


def _without_replicate_identity(config: dict) -> dict:
    comparable = copy.deepcopy(config)
    comparable["runtime"].pop("seed")
    comparable["log"].pop("out_dir")
    return comparable


def test_webgpu_proxy_pair_changes_only_architecture_and_output_directory() -> None:
    for seed in SEEDS:
        hybrid = _load("hybrid", seed)
        attention = _load("attn", seed)

        assert _without_pair_identity(hybrid) == _without_pair_identity(attention)
        assert hybrid["model_config"] == "configs/model/webgpu-10m-hybrid.yaml"
        assert attention["model_config"] == "configs/model/webgpu-10m-attn.yaml"
        assert hybrid["log"]["out_dir"] != attention["log"]["out_dir"]


def test_webgpu_proxy_replicates_change_only_seed_and_output_directory() -> None:
    for architecture in ("hybrid", "attn"):
        configs = [_load(architecture, seed) for seed in SEEDS]
        reference = _without_replicate_identity(configs[0])
        assert all(_without_replicate_identity(config) == reference for config in configs[1:])


def test_webgpu_proxy_pairs_freeze_data_order_and_scheduled_token_budget() -> None:
    for seed in SEEDS:
        for config in (_load("hybrid", seed), _load("attn", seed)):
            assert config["stage"] == "pretrain"
            assert config["data"] == {
                "shards_dir": "data/shards/webgpu-proxy-120m",
                "min_train_tokens": SCHEDULED_TOKENS,
                "tokenizer": {
                    "kind": "bpe",
                    "path": "data/tokenizer-webgpu-proxy-16k.json",
                },
            }
            assert config["batch"] == {
                "micro_batch_size": MICRO_BATCH_SIZE,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
            }
            assert config["optim"] == {
                "name": "adamw",
                "lr": 3.0e-4,
                "weight_decay": 0.1,
                "grad_clip": 1.0,
            }
            assert config["schedule"] == {
                "type": "wsd",
                "warmup_steps": 6,
                "total_steps": TOTAL_STEPS,
                "decay_frac": 0.2,
            }
            assert config["runtime"]["seed"] == seed
            assert config["runtime"]["resume"] is True

    assert SCHEDULED_TOKENS == 10_551_296


def test_webgpu_proxy_pairs_are_within_budget_and_at_least_one_tpp() -> None:
    for seed in SEEDS:
        for architecture in ("hybrid", "attn"):
            train_config = _load(architecture, seed)
            model_config = ModelConfig.from_yaml(str(ROOT / train_config["model_config"]))

            model_config.assert_within_budget()
            assert model_config.max_seq_len == SEQ_LEN
            assert SCHEDULED_TOKENS / model_config.estimate_params() >= 1.0
