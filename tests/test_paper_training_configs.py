from __future__ import annotations

import copy
from pathlib import Path

import yaml

from localagent.model import ModelConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "train"
DATA_CONFIG_ROOT = ROOT / "configs" / "data"
SEEDS = (2026, 2027, 2028)
SCHEDULED_TOKENS = 5231 * 2 * 8 * 2048
FULL_STEPS = 20921
FULL_SCHEDULED_TOKENS = FULL_STEPS * 2 * 8 * 2048


def _load(architecture: str, seed: int) -> dict:
    path = CONFIG_ROOT / f"pretrain-paper-5tpp-{architecture}-seed{seed}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_full(architecture: str) -> dict:
    path = CONFIG_ROOT / f"pretrain-paper-{architecture}.yaml"
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


def test_five_tpp_pairs_change_only_architecture_and_output_directory() -> None:
    for seed in SEEDS:
        hybrid = _load("hybrid", seed)
        attention = _load("attn", seed)

        assert _without_pair_identity(hybrid) == _without_pair_identity(attention)
        assert hybrid["runtime"]["seed"] == attention["runtime"]["seed"] == seed
        assert hybrid["schedule"]["total_steps"] == attention["schedule"]["total_steps"] == 5231
        assert hybrid["data"]["min_train_tokens"] == SCHEDULED_TOKENS
        assert attention["data"]["min_train_tokens"] == SCHEDULED_TOKENS


def test_five_tpp_replicates_change_only_seed_and_output_directory() -> None:
    for architecture in ("hybrid", "attn"):
        configs = [_load(architecture, seed) for seed in SEEDS]
        reference = _without_replicate_identity(configs[0])
        assert all(_without_replicate_identity(config) == reference for config in configs[1:])


def test_five_tpp_budget_is_at_least_five_tokens_per_parameter_for_both_arms() -> None:
    model_paths = {
        "hybrid": ROOT / "configs" / "model" / "webgpu-35m-hybrid.yaml",
        "attn": ROOT / "configs" / "model" / "webgpu-35m-attn.yaml",
    }
    for path in model_paths.values():
        config = ModelConfig.from_yaml(str(path))
        config.assert_within_budget()
        assert SCHEDULED_TOKENS / config.estimate_params() >= 5.0


def test_full_pairs_are_matched_and_reach_twenty_tokens_per_parameter() -> None:
    model_paths = {
        "hybrid": ROOT / "configs" / "model" / "webgpu-35m-hybrid.yaml",
        "attn": ROOT / "configs" / "model" / "webgpu-35m-attn.yaml",
    }
    hybrid = _load_full("hybrid")
    attention = _load_full("attn")

    assert _without_pair_identity(hybrid) == _without_pair_identity(attention)
    assert hybrid["schedule"]["total_steps"] == attention["schedule"]["total_steps"] == FULL_STEPS
    for path in model_paths.values():
        config = ModelConfig.from_yaml(str(path))
        config.assert_within_budget()
        assert FULL_SCHEDULED_TOKENS / config.estimate_params() >= 20.0


def test_external_benchmark_plan_matches_paper_decontamination_policy() -> None:
    plan = yaml.safe_load(
        (DATA_CONFIG_ROOT / "evaluation-benchmarks-paper.yaml").read_text(encoding="utf-8")
    )
    corpus = yaml.safe_load((DATA_CONFIG_ROOT / "pretrain-paper.yaml").read_text(encoding="utf-8"))
    assert plan["kind"] == "localagent_external_benchmark_plan"
    assert plan["schema_version"] == 1
    assert plan["purpose"] == "pretraining_prompt_only_decontamination"
    assert plan["forbid_gold_in_prompt_exports"] is True

    external_names = {"bfcl", "browsergym", "mind2web", "weblinx"}
    assert set(plan["suites"]) == external_names
    prompt_freeze = plan["prompt_freeze"]
    assert set(prompt_freeze["external_manifest_suites"]) == external_names
    assert prompt_freeze["require_adapter_audit_binding"] is True
    assert prompt_freeze["require_license_evidence_binding"] is True
    assert prompt_freeze["require_prompt_only_isolation"] is True
    assert set(prompt_freeze["config_hash_pinned_direct_suites"]) == {
        "local-realtime-actions",
        "local-browser-tasks",
        "local-agent-eval",
    }
    required = {row["name"] for row in corpus["evaluation_decontamination"]["required_suites"]}
    assert external_names <= required

    expected_adapters = {
        "bfcl": "bfcl-v4-prompt-rows-v1",
        "browsergym": "browsergym-miniwob-reset-capture-prompt-rows-v1",
        "mind2web": "mind2web-private-prompt-rows-v2",
        "weblinx": "weblinx-private-prompt-rows-v1",
    }
    for name, suite in plan["suites"].items():
        assert len(suite["revision"]) == 40
        assert suite["adapter"] == expected_adapters[name]
        assert suite["prompt_freeze_split"]
        assert suite["v1_evaluation_scope"] != "official"
    browsergym = plan["suites"]["browsergym"]
    assert len(browsergym["fixed_seeds"]) * browsergym["expected_task_variants"] == 240
    assert browsergym["expected_similarity_groups"] == 41
    assert set(browsergym["localagent_policy_exclusions"]) == {
        "click-pie",
        "click-pie-nodelay",
        "terminal",
    }
    assert "excluded_nondeterministic" not in browsergym
    assert len(browsergym["miniwob_revision"]) == 40
    assert browsergym["prompt_capture"] == {
        "status": "frozen_controlled_acquisition",
        "file": "browsergym-miniwob-reset-goals.jsonl",
        "bytes": 348_513,
        "sha256": "128f7f6be8d5b52f745523b0bca4517fdaf8107044eee5a76366464ac10079ff",
        "requirement": "freeze_before_tokenizer_fit",
    }
    assert browsergym["capture_receipt"] == {
        "status": "frozen_controlled_acquisition",
        "file": "browsergym-miniwob-reset-goals.receipt.json",
        "bytes": 6_538,
        "sha256": "b04318c36579a05d3f61a40ea09c1f1c0bd1e004a534b2b5d18305b50e68ebea",
        "receipt_self_sha256": ("e8cece5a8acf0f5e2333e004c33b035b4b31fa7ec3e3d501c43fcbbac341611a"),
        "kind": "localagent_browsergym_capture_producer_receipt",
        "schema_version": 3,
        "producer": "browsergym-miniwob-controlled-reset-goals-v3",
    }
    assert browsergym["browsergym_version"] == "0.14.3"
    assert browsergym["task_groups_sha256"] == (
        "e2a596126bc3bc37c2351b60cc4d59971628c8d9eac804c990d34515453fd3df"
    )
    assert browsergym["runtime_pins"] == {
        "playwright_version": "1.44.0",
        "chromium_revision": "1117",
        "chromium_version": "125.0.6422.26",
        "locale": "en-US",
        "timezone_id": "UTC",
        "headless": True,
        "viewport": {"width": 1280, "height": 720},
        "device_scale_factor": 1.0,
        "action_set": "highlevel-default-unused-reset-only",
        "observation_mode": "processed-dom-axtree-screenshot",
        "max_steps": 10,
        "playwright_operation_timeout_seconds": 30.0,
        "environment_manifest": {
            "kind": "localagent_browsergym_capture_environment",
            "schema_version": 1,
            "file": "browsergym-capture-runtime-darwin-arm64-py312.json",
            "bytes": 12_996,
            "sha256": ("5edf3987b09db987eabbef52324ef6d0eb87d69e7c36e94d5f88cdccddf21382"),
            "self_sha256": ("274802850e0bef5635b906a668f49e3c540e459dee1841b20a54e55ccc3863c7"),
            "distributions": 51,
            "playwright_driver_sha256": (
                "cb628761b7355e456bd2581f8a0b008d200ca3e2a6f53c466c3c245b63db26da"
            ),
        },
    }
    bfcl_sources = plan["suites"]["bfcl"]["pinned_prompt_sources"]
    assert sum(source["rows"] for source in bfcl_sources.values()) == 1000
    assert sum(source["bytes"] for source in bfcl_sources.values()) == 1_118_833
    assert all(len(source["sha256"]) == 64 for source in bfcl_sources.values())
    mind2web = plan["suites"]["mind2web"]
    assert mind2web["redistribution"] == "private_hashes_and_counts_only"
    assert sum(mind2web["heldout_splits"].values()) == 1_341
    protected_archive = mind2web["protected_test_archive"]
    expected_member_splits = {
        **{f"test_domain/test_domain_{index}.json": "cross_domain" for index in range(10)},
        **{f"test_task/test_task_{index}.json": "cross_task" for index in range(3)},
        **{f"test_website/test_website_{index}.json": "cross_website" for index in range(2)},
    }
    assert protected_archive == {
        "bytes": 567_745_122,
        "compression": "deflate",
        "encryption": "zipcrypto",
        "members": 15,
        "member_splits": expected_member_splits,
        "sha256": "8f5fbe72afab942fe97cdf7fb397e179885d89b5c16862288e9a14bc6d41ca89",
        "uncompressed_bytes": 6_107_912_752,
        "require_exact_member_set": True,
        "require_plaintext_member_hash_binding": True,
    }
    weblinx = plan["suites"]["weblinx"]
    assert weblinx["allow_training_use"] is False
    assert weblinx["expected_source_rows"] == 4_856
    assert weblinx["privacy_filter_receipt"] == {
        "filter_version": "localagent_weblinx_whole_demo_privacy_v1",
        "scanned_demos": 211,
        "accepted_demos": 146,
        "excluded_demos": 65,
        "excluded_rows": 1_936,
        "retained_rows": 2_920,
        "reason_counts": {
            "email": 61,
            "labeled_secret": 43,
            "payment_card": 11,
        },
    }
    assert weblinx["require_whole_demo_sensitive_pattern_exclusion_v1"] is True
    assert weblinx["require_private_manual_residual_privacy_review"] is True
    assert "require_whole_demo_credential_pii_exclusion" not in weblinx
    assert weblinx["pinned_prompt_sources"]["chat"]["bytes"] == 2_187_263
    assert weblinx["pinned_prompt_sources"]["splits"]["bytes"] == 38_210


def test_protected_benchmark_roots_are_gitignored() -> None:
    patterns = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert {"/private/", "/data/private/"} <= patterns
