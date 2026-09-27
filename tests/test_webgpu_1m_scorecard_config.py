from __future__ import annotations

from pathlib import Path

import yaml

from localagent.eval.agent_scorecard import (
    CONFIG_KIND,
    SCHEMA_VERSION,
    _CASE_KEYS,
    _CASE_SELECTION_KEYS,
    _CONFIG_KEYS,
    _GENERATION_KEYS,
)
from localagent.model import ModelConfig

ROOT = Path(__file__).resolve().parents[1]
SCORECARD_PATH = ROOT / "configs" / "eval" / "webgpu-1m-sft-scorecard.yaml"
TRAINING_PATH = ROOT / "configs" / "train" / "sft-paper-tier-1m.yaml"
MODEL_PATH = ROOT / "configs" / "model" / "webgpu-1m-bpe-router.yaml"


def test_webgpu_1m_sft_scorecard_binds_strict_full_catalog_eval_lane() -> None:
    scorecard = yaml.safe_load(SCORECARD_PATH.read_text(encoding="utf-8"))
    training = yaml.safe_load(TRAINING_PATH.read_text(encoding="utf-8"))
    model = ModelConfig.from_yaml(str(MODEL_PATH))

    assert set(scorecard) == _CONFIG_KEYS
    assert scorecard["kind"] == CONFIG_KIND
    assert scorecard["schema_version"] == SCHEMA_VERSION
    assert scorecard["checkpoint"] == f"{training['log']['out_dir']}/latest.pt"
    assert scorecard["training_config"] == "configs/train/sft-paper-tier-1m.yaml"
    assert scorecard["model_config"] == training["model_config"]
    assert scorecard["tokenizer"] == training["data"]["tokenizer"] == {
        "kind": "bpe",
        "path": "data/tokenizer-paper-16k.json",
    }
    assert training["data"]["conversation_prompt_contract"] == "openai_full_catalog_v1"
    assert training["data"]["strict_conversation_artifacts"] is True

    eval_entry = training["data"]["eval_conversations"]
    assert len(eval_entry) == 1
    assert set(scorecard["cases"]) == _CASE_KEYS
    cases_without_selection = {
        key: value
        for key, value in scorecard["cases"].items()
        if key != "selection"
    }
    assert cases_without_selection == {
        "path": eval_entry[0]["path"],
        **eval_entry[0]["artifact"],
    }
    assert scorecard["cases"]["expected_split"] == "eval"
    assert scorecard["cases"]["expected_rule_verified"] is True
    assert scorecard["cases"]["environment_policy"] == "forbid"
    selection = scorecard["cases"]["selection"]
    assert set(selection) == _CASE_SELECTION_KEYS
    assert selection == {
        "algorithm": training["evaluation"]["selection"],
        "max_rows": training["evaluation"]["max_conversations"],
        "expected_source_rows": 5_000,
        "expected_source_assistant_decisions": 7_963,
        "expected_source_semantic_set_sha256": (
            "02c7e08baaaa97b54f522ba3ee5f979993000de5a3c24507c7bc4a2479355999"
        ),
        "expected_selected_rows": 512,
        "expected_selected_assistant_decisions": 820,
        "expected_selected_semantic_set_sha256": (
            "5eb08ef61dcdfab5889f66ecb04c17fbce6ce2726f868ab85f1afe6120505bf3"
        ),
        "expected_audit_sha256": (
            "342abcb7ad550d4b73c726c7bbf74a68cc1fef5bda1ca68707fd8b75a16bc641"
        ),
    }

    assert set(scorecard["generation"]) == _GENERATION_KEYS
    assert scorecard["generation"] == {
        "device": "auto",
        "max_new_tokens": 96,
    }
    assert selection["expected_selected_assistant_decisions"] * 96 == 78_720
    model.assert_within_budget()
    assert model.vocab_size == 16_384
    assert model.max_seq_len == 4_096
