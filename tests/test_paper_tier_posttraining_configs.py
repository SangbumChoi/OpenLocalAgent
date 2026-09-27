from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from localagent.model import ModelConfig

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "configs" / "train"
ARMS = ("hybrid", "attn")
STAGES = ("midtrain", "sft", "rl")
SUPERVISED_STAGES = ("pretrain", "midtrain", "sft")
PROMPT_CONTRACT = "openai_full_catalog_v1"
PAPER_TOKENIZER = {
    "kind": "bpe",
    "path": "data/tokenizer-paper-16k.json",
}
MODEL_MAX_SEQ_LEN = 4096
MEASURED_MAX_TRAIN_PROMPT_TOKENS = 3664
MEASURED_MAX_TRAIN_ROW_TOKENS = 3685
EVAL_MAX_CONVERSATIONS = 512
EVAL_SELECTION = "greedy_uncovered_strata_then_semantic_sha256_fill_v1"
QUOTA_SAMPLING = {"mode": "quota_stratified_no_replacement_v1"}

# Exhaustive openai_full_catalog_v1 scan of the frozen 50k-row train artifact with the paper BPE.
# The artifact identity makes these measured maxima reproducible without rescanning 526 MB in CI.
MEASURED_TRAIN_ARTIFACT_SHA256 = "233f4f2d796568097897c73d4547a0129e73a8509981a308600779e3cb4cc060"

TIER_CHAINS = (
    {
        "tier": "1m",
        "arm": None,
        "model_config": "configs/model/webgpu-1m-bpe-router.yaml",
        "micro_batch_size": 2,
        "grad_accum_steps": 8,
        "midtrain_steps": 599,
        "midtrain_warmup_steps": 12,
        "sft_steps": 348,
        "sft_warmup_steps": 7,
        "sft_ckpt_every": 25,
        "eval_batch_size": 8,
        "rl_steps": 18,
        "rl_warmup_steps": 1,
        "rl_ckpt_every": 2,
    },
    {
        "tier": "10m",
        "arm": "hybrid",
        "model_config": "configs/model/webgpu-10m-hybrid-4k.yaml",
        "micro_batch_size": 2,
        "grad_accum_steps": 8,
        "midtrain_steps": 1_610,
        "midtrain_warmup_steps": 32,
        "sft_steps": 936,
        "sft_warmup_steps": 19,
        "sft_ckpt_every": 50,
        "eval_batch_size": 8,
        "rl_steps": 48,
        "rl_warmup_steps": 1,
        "rl_ckpt_every": 4,
    },
    {
        "tier": "10m",
        "arm": "attn",
        "model_config": "configs/model/webgpu-10m-attn-4k.yaml",
        "micro_batch_size": 2,
        "grad_accum_steps": 8,
        "midtrain_steps": 1_610,
        "midtrain_warmup_steps": 32,
        "sft_steps": 936,
        "sft_warmup_steps": 19,
        "sft_ckpt_every": 50,
        "eval_batch_size": 8,
        "rl_steps": 48,
        "rl_warmup_steps": 1,
        "rl_ckpt_every": 4,
    },
    {
        "tier": "96m",
        "arm": "hybrid",
        "model_config": "configs/model/webgpu-96m-hybrid.yaml",
        "micro_batch_size": 1,
        "grad_accum_steps": 16,
        "midtrain_steps": 2_909,
        "midtrain_warmup_steps": 58,
        "sft_steps": 1_690,
        "sft_warmup_steps": 34,
        "sft_ckpt_every": 100,
        "eval_batch_size": 1,
        "rl_steps": 86,
        "rl_warmup_steps": 2,
        "rl_ckpt_every": 5,
    },
    {
        "tier": "96m",
        "arm": "attn",
        "model_config": "configs/model/webgpu-96m-attn.yaml",
        "micro_batch_size": 1,
        "grad_accum_steps": 16,
        "midtrain_steps": 2_909,
        "midtrain_warmup_steps": 58,
        "sft_steps": 1_690,
        "sft_warmup_steps": 34,
        "sft_ckpt_every": 100,
        "eval_batch_size": 1,
        "rl_steps": 86,
        "rl_warmup_steps": 2,
        "rl_ckpt_every": 5,
    },
)

FROZEN_35M_POSTTRAIN_SHA256 = {
    "midtrain-paper-hybrid.yaml": (
        "0ad2c89523be0c72bd360db915ff076b5763f6c80a0ee7b0553c11f2dd840cf0"
    ),
    "midtrain-paper-attn.yaml": (
        "d700c830294d466b35a0122f5177c58964480c2acc02116441f5540e724881cf"
    ),
    "sft-paper-hybrid.yaml": ("bd5e9f348fe4c7a9fe29e7cab59c6d5888c49c14819dd854a574ba8cf0e914d0"),
    "sft-paper-attn.yaml": ("2c4ef38e9e4a53ca142cc84e091c30900afce47733648153de2e0c7afcb0de45"),
    "rl-paper-hybrid.yaml": ("0f9b56162302e7e10c7c21d96d16e09e2692808506acf8be146dcfa6fabd0806"),
    "rl-paper-attn.yaml": ("47a656c50d7c57f8cdf8b832956686971d659e69ab7c31a5b3abc54288cf2526"),
}

TRAIN_ARTIFACT = {
    "generator_config": "configs/data/agent_synth_paper_train_v2.yaml",
    "manifest": "data/synth/agent_sft_paper_train_v2.jsonl.manifest.v1.json",
    "expected_split": "train",
    "expected_rule_verified": True,
    "environment_policy": "forbid",
}
EVAL_ARTIFACT = {
    "generator_config": "configs/data/agent_synth_eval.yaml",
    "manifest": "data/synth/agent_eval.jsonl.manifest.v1.json",
    "expected_split": "eval",
    "expected_rule_verified": True,
    "environment_policy": "forbid",
}


def _tier_slug(tier: str, arm: str | None) -> str:
    return f"paper-tier-{tier}" if arm is None else f"paper-tier-{tier}-{arm}"


def _load_tier(stage: str, tier: str, arm: str | None) -> dict:
    path = TRAIN_ROOT / f"{stage}-{_tier_slug(tier, arm)}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_frozen(stage: str, arm: str) -> dict:
    path = TRAIN_ROOT / f"{stage}-paper-{arm}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _without_arm_identity(config: dict) -> dict:
    comparable = copy.deepcopy(config)
    comparable.pop("model_config")
    comparable.pop("init_from")
    comparable["log"].pop("out_dir")
    return comparable


def _without_chain_identity_or_batch(config: dict) -> dict:
    comparable = _without_arm_identity(config)
    comparable.pop("batch", None)
    comparable["schedule"].pop("warmup_steps", None)
    comparable["schedule"].pop("total_steps", None)
    comparable["log"].pop("ckpt_every", None)
    comparable.get("evaluation", {}).pop("batch_size", None)
    return comparable


def _without_tier_changes(config: dict, *, stage: str) -> dict:
    comparable = _without_arm_identity(config)
    comparable["log"].pop("ckpt_every", None)
    comparable["data"].pop("conversation_prompt_contract", None)
    comparable.get("evaluation", {}).pop("max_conversations", None)
    comparable.get("evaluation", {}).pop("selection", None)
    if comparable.get("evaluation") == {}:
        comparable.pop("evaluation")
    if stage == "midtrain":
        comparable["data"]["mixture"]["unit"] = "loss_tokens"
        comparable["schedule"]["warmup_steps"] = 100
        comparable["schedule"]["total_steps"] = 25_000
    if stage == "sft":
        comparable["data"]["seq_len"] = 2048
        comparable["data"]["shuffle"] = True
        comparable["data"].pop("sampling", None)
        comparable["schedule"]["warmup_steps"] = 200
        comparable["schedule"]["total_steps"] = 10_000
        comparable.pop("heads")
    if stage == "rl":
        comparable["rollout"].pop("prompt_sampling", None)
        comparable["schedule"].pop("warmup_steps", None)
        comparable["schedule"]["total_steps"] = 300
    return comparable


def _conversation_entries(config: dict) -> tuple[list[dict], list[dict]]:
    data = config["data"]
    if config["stage"] == "midtrain":
        train = [entry for entry in data["sources"] if entry["type"] == "conversations"]
        evaluation = [entry for entry in data["eval_sources"] if entry["type"] == "conversations"]
        return train, evaluation
    return data["conversations"], data["eval_conversations"]


def test_paired_tier_arms_match_except_model_parent_and_output_identity() -> None:
    for tier in ("10m", "96m"):
        for stage in STAGES:
            hybrid = _load_tier(stage, tier, "hybrid")
            attention = _load_tier(stage, tier, "attn")
            assert _without_arm_identity(hybrid) == _without_arm_identity(attention)


def test_every_tier_chain_has_exact_parent_continuity_and_unique_outputs() -> None:
    output_dirs: set[str] = set()
    for case in TIER_CHAINS:
        tier = case["tier"]
        arm = case["arm"]
        slug = _tier_slug(tier, arm)
        pretrain = _load_tier("pretrain", tier, arm)
        midtrain = _load_tier("midtrain", tier, arm)
        sft = _load_tier("sft", tier, arm)
        rl = _load_tier("rl", tier, arm)

        assert midtrain["init_from"] == f"{pretrain['log']['out_dir']}/latest.pt"
        assert sft["init_from"] == f"{midtrain['log']['out_dir']}/latest.pt"
        assert rl["init_from"] == f"{sft['log']['out_dir']}/latest.pt"
        assert midtrain["schedule"]["total_steps"] == pretrain["schedule"]["total_steps"]
        assert midtrain["schedule"]["warmup_steps"] == pretrain["schedule"]["warmup_steps"]

        assert pretrain["log"]["out_dir"] == f"runs/pretrain-{slug}-seed2026"
        for stage, config in zip(STAGES, (midtrain, sft, rl), strict=True):
            out_dir = config["log"]["out_dir"]
            assert out_dir == f"runs/{stage}-{slug}"
            assert out_dir not in output_dirs
            output_dirs.add(out_dir)

    assert len(output_dirs) == len(TIER_CHAINS) * len(STAGES)


def test_all_tiers_reuse_paper_sources_artifacts_and_primary_ar_policy() -> None:
    for stage in STAGES:
        reference = _load_tier(stage, "10m", "hybrid")
        for case in TIER_CHAINS:
            tier = case["tier"]
            arm = case["arm"]
            tier_config = _load_tier(stage, tier, arm)
            assert _without_chain_identity_or_batch(
                tier_config
            ) == _without_chain_identity_or_batch(reference)

            train_entries, eval_entries = _conversation_entries(tier_config)
            assert len(train_entries) == len(eval_entries) == 1
            assert train_entries[0]["path"] == "data/synth/agent_sft_paper_train_v2.jsonl"
            assert train_entries[0]["artifact"] == TRAIN_ARTIFACT
            assert eval_entries[0]["path"] == "data/synth/agent_eval.jsonl"
            assert eval_entries[0]["artifact"] == EVAL_ARTIFACT

    for case in TIER_CHAINS:
        tier = case["tier"]
        arm = case["arm"]
        midtrain = _load_tier("midtrain", tier, arm)
        assert midtrain["schedule"]["total_steps"] == case["midtrain_steps"]
        assert midtrain["schedule"]["warmup_steps"] == case["midtrain_warmup_steps"]
        assert midtrain["data"]["mixture"]["unit"] == "input_tokens"
        sft = _load_tier("sft", tier, arm)
        assert sft["schedule"]["total_steps"] == case["sft_steps"]
        assert sft["schedule"]["warmup_steps"] == case["sft_warmup_steps"]
        assert sft["evaluation"]["batch_size"] == case["eval_batch_size"]
        assert sft["evaluation"]["max_conversations"] == EVAL_MAX_CONVERSATIONS
        assert sft["evaluation"]["selection"] == EVAL_SELECTION
        assert sft["data"]["shuffle"] is False
        assert sft["data"]["sampling"] == QUOTA_SAMPLING
        assert sft["heads"] == {
            "joint_tool_pointer": False,
            "train_route_head": False,
            "train_dense_selector": False,
        }
        assert sft["log"]["ckpt_every"] == case["sft_ckpt_every"]
        rl = _load_tier("rl", tier, arm)
        assert rl["schedule"]["total_steps"] == case["rl_steps"]
        assert rl["schedule"]["warmup_steps"] == case["rl_warmup_steps"]
        assert rl["evaluation"] == {
            "max_conversations": EVAL_MAX_CONVERSATIONS,
            "selection": EVAL_SELECTION,
        }
        assert rl["rollout"]["prompt_sampling"] == QUOTA_SAMPLING
        assert rl["log"]["ckpt_every"] == case["rl_ckpt_every"]
        assert rl["environment"] == {
            "name": "canonical_toolcalls",
            "learned_judge": False,
        }


def test_10m_tier_stages_preserve_the_frozen_35m_stage_contract() -> None:
    for stage in STAGES:
        for arm in ARMS:
            tier = _load_tier(stage, "10m", arm)
            frozen = _load_frozen(stage, arm)
            assert _without_tier_changes(tier, stage=stage) == _without_tier_changes(
                frozen,
                stage=stage,
            )


def test_every_conversation_stage_uses_strict_full_catalog_paper_bpe() -> None:
    for stage in STAGES:
        for case in TIER_CHAINS:
            config = _load_tier(stage, case["tier"], case["arm"])
            assert config["data"]["conversation_prompt_contract"] == PROMPT_CONTRACT
            assert config["data"]["strict_conversation_artifacts"] is True
            assert config["data"]["tokenizer"] == PAPER_TOKENIZER


def test_supervised_stage_batch_geometry_is_tier_appropriate() -> None:
    for case in TIER_CHAINS:
        expected = {
            "micro_batch_size": case["micro_batch_size"],
            "grad_accum_steps": case["grad_accum_steps"],
        }
        for stage in SUPERVISED_STAGES:
            config = _load_tier(stage, case["tier"], case["arm"])
            assert config["batch"] == expected


def test_models_and_sft_are_4k_and_measured_rl_reserve_fits() -> None:
    manifest_path = ROOT / TRAIN_ARTIFACT["manifest"]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["output_sha256"] == MEASURED_TRAIN_ARTIFACT_SHA256
        assert manifest["rows"] == 50_000
    assert MEASURED_MAX_TRAIN_PROMPT_TOKENS == 3664
    assert MEASURED_MAX_TRAIN_ROW_TOKENS == 3685

    for case in TIER_CHAINS:
        tier = case["tier"]
        arm = case["arm"]
        expected_model_path = case["model_config"]
        for stage in STAGES:
            config = _load_tier(stage, tier, arm)
            assert config["model_config"] == expected_model_path
            model = ModelConfig.from_yaml(str(ROOT / config["model_config"]))
            model.assert_within_budget()
            assert model.max_seq_len == MODEL_MAX_SEQ_LEN

        sft = _load_tier("sft", tier, arm)
        rl = _load_tier("rl", tier, arm)
        max_new_tokens = rl["rollout"]["max_new_tokens"]
        assert sft["data"]["seq_len"] == MODEL_MAX_SEQ_LEN
        assert max_new_tokens == 256
        assert MEASURED_MAX_TRAIN_PROMPT_TOKENS + max_new_tokens <= MODEL_MAX_SEQ_LEN
        assert MEASURED_MAX_TRAIN_ROW_TOKENS <= MODEL_MAX_SEQ_LEN


def test_frozen_35m_posttraining_configs_are_unchanged_and_not_referenced() -> None:
    for name, expected_sha256 in FROZEN_35M_POSTTRAIN_SHA256.items():
        contents = (TRAIN_ROOT / name).read_bytes()
        assert hashlib.sha256(contents).hexdigest() == expected_sha256

    for stage in STAGES:
        for case in TIER_CHAINS:
            path = TRAIN_ROOT / f"{stage}-{_tier_slug(case['tier'], case['arm'])}.yaml"
            text = path.read_text(encoding="utf-8")
            assert "webgpu-35m" not in text
            assert "runs/pretrain-paper-hybrid" not in text
            assert "runs/pretrain-paper-attn" not in text
            assert "runs/midtrain-paper-hybrid" not in text
            assert "runs/midtrain-paper-attn" not in text
            assert "runs/sft-paper-hybrid" not in text
            assert "runs/sft-paper-attn" not in text
