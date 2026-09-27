from __future__ import annotations

from pathlib import Path

import yaml

from localagent.model import ModelConfig


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG_ROOT = ROOT / "configs" / "train"
MODEL_CONFIG = "configs/model/webgpu-10m-hybrid.yaml"
TOKENIZER = {
    "kind": "bpe",
    "path": "data/tokenizer-webgpu-proxy-16k.json",
}
TRAIN_CONVERSATIONS = "data/synth/agent_sft.jsonl"
PILOT_EVAL = "data/synth/agent_eval_pilot65.jsonl"


def _load(name: str) -> dict:
    return yaml.safe_load((TRAIN_CONFIG_ROOT / name).read_text(encoding="utf-8"))


def test_pilot_configs_form_the_seed_2027_checkpoint_chain() -> None:
    pretrain = _load("pretrain-webgpu-proxy-1tpp-hybrid-seed2027.yaml")
    midtrain = _load("midtrain-webgpu-proxy-pilot-hybrid.yaml")
    sft = _load("sft-webgpu-proxy-pilot-hybrid.yaml")
    rl = _load("rl-webgpu-proxy-pilot-hybrid.yaml")

    assert [midtrain["stage"], sft["stage"], rl["stage"]] == [
        "midtrain",
        "sft",
        "rl",
    ]
    assert midtrain["init_from"] == f"{pretrain['log']['out_dir']}/latest.pt"
    assert sft["init_from"] == f"{midtrain['log']['out_dir']}/latest.pt"
    assert rl["init_from"] == f"{sft['log']['out_dir']}/latest.pt"
    assert len(
        {
            pretrain["log"]["out_dir"],
            midtrain["log"]["out_dir"],
            sft["log"]["out_dir"],
            rl["log"]["out_dir"],
        }
    ) == 4

    for config in (midtrain, sft, rl):
        assert config["model_config"] == MODEL_CONFIG
        assert config["data"]["tokenizer"] == TOKENIZER
        assert config["runtime"]["seed"] == 2027
    assert midtrain["evaluation"]["seed"] == 12027
    assert "seed" not in sft["evaluation"]
    assert "evaluation" not in rl


def test_pilot_configs_keep_the_frozen_subset_evaluation_only() -> None:
    midtrain = _load("midtrain-webgpu-proxy-pilot-hybrid.yaml")
    sft = _load("sft-webgpu-proxy-pilot-hybrid.yaml")
    rl = _load("rl-webgpu-proxy-pilot-hybrid.yaml")

    midtrain_training_sources = {
        (source["path"], source.get("split"))
        for source in midtrain["data"]["sources"]
    }
    midtrain_eval_sources = {
        (source["path"], source.get("split"))
        for source in midtrain["data"]["eval_sources"]
    }
    midtrain_training_paths = {path for path, _ in midtrain_training_sources}
    midtrain_eval_paths = {path for path, _ in midtrain_eval_sources}
    assert TRAIN_CONVERSATIONS in midtrain_training_paths
    assert PILOT_EVAL in midtrain_eval_paths
    assert PILOT_EVAL not in midtrain_training_paths
    assert midtrain_training_sources.isdisjoint(midtrain_eval_sources)

    for config in (sft, rl):
        assert config["data"]["conversations"] == [TRAIN_CONVERSATIONS]
        assert config["data"]["eval_conversations"] == [PILOT_EVAL]
        assert PILOT_EVAL not in config["data"]["conversations"]


def test_pilot_training_and_rollout_budgets_are_bounded() -> None:
    midtrain = _load("midtrain-webgpu-proxy-pilot-hybrid.yaml")
    sft = _load("sft-webgpu-proxy-pilot-hybrid.yaml")
    rl = _load("rl-webgpu-proxy-pilot-hybrid.yaml")

    model = ModelConfig.from_yaml(str(ROOT / MODEL_CONFIG))
    model.assert_within_budget()
    assert model.estimate_params() < 100_000_000

    midtrain_token_budget = (
        midtrain["schedule"]["total_steps"]
        * midtrain["batch"]["micro_batch_size"]
        * midtrain["batch"]["grad_accum_steps"]
        * model.max_seq_len
    )
    sft_token_budget = (
        sft["schedule"]["total_steps"]
        * sft["batch"]["micro_batch_size"]
        * sft["batch"]["grad_accum_steps"]
        * sft["data"]["seq_len"]
    )
    rl_completion_budget = (
        rl["schedule"]["total_steps"]
        * rl["rollout"]["prompts_per_step"]
        * rl["rollout"]["group_size"]
        * rl["rollout"]["max_new_tokens"]
    )
    assert midtrain["schedule"]["total_steps"] == 64
    assert midtrain_token_budget == 1_048_576
    assert sft["schedule"]["total_steps"] == 320
    assert sft_token_budget == 10_485_760
    assert rl["schedule"]["total_steps"] == 32
    assert rl_completion_budget == 16_384
    assert rl["rollout"]["max_new_tokens"] == 64
