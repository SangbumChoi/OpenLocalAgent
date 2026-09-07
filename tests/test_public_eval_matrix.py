import json
from pathlib import Path

import pytest

from openlocalagent.data.decontam.public_eval_matrix import (
    entries_by_family,
    load_matrix,
    trainable_entries,
    validate_matrix,
)


MATRIX = Path(__file__).parents[1] / "configs/data/realistic-agent-public-eval-matrix.v1.json"


def test_public_matrix_is_source_linked_and_split_explicit() -> None:
    matrix = load_matrix(MATRIX)
    assert len(matrix["entries"]) >= 20
    assert {row["id"] for row in trainable_entries(matrix)} == {
        "androidcontrol",
        "android_in_the_wild",
        "mind2web",
        "agentnet",
        "xlam_function_calling",
        "toolace",
    }
    assert all(row["source_url"].startswith("https://") for row in matrix["entries"])
    assert all(row["paper_url"].startswith("https://") for row in matrix["entries"])
    assert all(row["split_rule"] for row in matrix["entries"])


def test_public_matrix_covers_the_realistic_modalities() -> None:
    matrix = load_matrix(MATRIX)
    assert {row["family"] for row in matrix["entries"]} == {
        "mobile",
        "browser",
        "computer",
        "tool_api",
        "terminal",
    }
    assert {row["id"] for row in entries_by_family(matrix, "mobile")} >= {
        "androidworld",
        "mobilegym",
        "mobile_safety_bench",
        "iosworld",
        "mobileworld",
        "mobile_agent_bench",
    }
    status = {row["id"]: row["local_status"] for row in matrix["entries"]}
    assert status["mobile_safety_bench"] == "manifest_audited_native_pending"
    assert status["iosworld"] == "manifest_audited_native_pending"
    mobilegym = next(row for row in matrix["entries"] if row["id"] == "mobilegym")
    assert mobilegym["source_revision"] == "093a3292d13fc4186e279af4ef1b005ac8e4d2b7"
    assert mobilegym["local_status"] == "measured_official_text_projection"
    assert {row["id"] for row in entries_by_family(matrix, "tool_api")} >= {
        "mcpmark",
        "toolsandbox",
        "enterpriseopsgym",
        "appworld",
        "appworld_ul",
        "tau_bench",
        "toolace",
    }
    agentic_bfcl = next(row for row in matrix["entries"] if row["id"] == "bfcl_v4_agentic")
    assert agentic_bfcl["train_policy"] == "eval_only"
    assert "web_search" in agentic_bfcl["modalities"]
    assert "memory_state" in agentic_bfcl["modalities"]
    assert agentic_bfcl["source_revision"] == "release-required-before-score"
    appworld_ul = next(row for row in matrix["entries"] if row["id"] == "appworld_ul")
    assert appworld_ul["train_policy"] == "eval_only"
    assert "user_simulator" in appworld_ul["modalities"]
    assert "confirmation" in appworld_ul["webgpu_projection"]


def test_public_matrix_rejects_ambiguous_training_rows() -> None:
    raw = json.loads(MATRIX.read_text(encoding="utf-8"))
    raw["entries"] = [dict(raw["entries"][0])]
    raw["entries"][0]["access_status"] = "public_runtime"
    with pytest.raises(ValueError, match="public_download"):
        validate_matrix(raw)

    raw = json.loads(MATRIX.read_text(encoding="utf-8"))
    raw["entries"] = [dict(raw["entries"][0])]
    raw["entries"][0]["license"] = "terms_review"
    with pytest.raises(ValueError, match="reviewed license"):
        validate_matrix(raw)


def test_public_matrix_rejects_duplicate_ids() -> None:
    raw = json.loads(MATRIX.read_text(encoding="utf-8"))
    raw["entries"] = [dict(raw["entries"][0]), dict(raw["entries"][0])]
    with pytest.raises(ValueError, match="duplicate matrix id"):
        validate_matrix(raw)
