from pathlib import Path

import pytest
import yaml

from localagent.data.realistic_catalog import load_catalog, train_entries, validate_catalog


CATALOG = Path(__file__).parents[1] / "configs/data/realistic-agent-eval.catalog.yaml"
TRAINING_PLAN = Path(__file__).parents[1] / "configs/data/realistic-agent-training.example.yaml"


def test_realistic_catalog_is_pinned_and_split_safe() -> None:
    catalog, fingerprint = load_catalog(CATALOG)
    assert len(fingerprint) == 64
    assert len(catalog["entries"]) >= 10
    train = train_entries(catalog)
    assert {row["id"] for row in train} == {
        "androidcontrol",
        "android_in_the_wild",
        "xlam_function_calling",
        "mind2web_train",
    }
    assert all(row["train_policy"] == "train" for row in train)
    assert all(row["train_policy"] != "train" for row in catalog["entries"] if row not in train)
    eval_ids = {row["id"] for row in catalog["entries"] if row["train_policy"] != "train"}
    assert {
        "mobilegym",
        "mobileworld",
        "memgui_bench",
        "workarena",
        "weblinx",
        "osworld_v2",
        "toolathlon_gym",
        "phoneworld",
        "gui_odyssey",
        "mobile_agent_bench",
        "agentbench_fc",
        "visualwebarena",
        "omni_act",
        "worldgui",
        "macosworld",
        "assistgui",
    } <= eval_ids

    enterprise = next(row for row in catalog["entries"] if row["id"] == "enterpriseopsgym")
    assert enterprise["source_url"] == (
        "https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym"
    )
    assert enterprise["code_url"] == "https://github.com/ServiceNow/EnterpriseOps-Gym"
    assert enterprise["scale"] == {
        "kind": "benchmark",
        "tasks": 1115,
        "domains": 8,
        "email_tasks": 104,
        "tools": 512,
        "modes": 4,
        "public_rows": 2560,
        "viewer_oracle_rows": 649,
        "card_inventory_rows": 1115,
    }
    assert enterprise["license"]["name"] == "Apache-2.0"


def test_core_mobile_and_browser_papers_use_authoritative_sources() -> None:
    catalog, _ = load_catalog(CATALOG)
    rows = {row["id"]: row for row in catalog["entries"]}
    assert rows["androidworld"]["paper_url"] == "https://arxiv.org/abs/2405.14573"
    assert rows["browsergym_miniwob"]["paper_url"] == (
        "https://openreview.net/forum?id=5298fKGmv3"
    )


def test_train_rows_require_public_download_and_verified_license() -> None:
    catalog, _ = load_catalog(CATALOG)
    broken = {**catalog, "entries": [dict(catalog["entries"][0])]}
    broken["entries"][0]["access_status"] = "public_runtime"
    with pytest.raises(ValueError, match="public_download"):
        validate_catalog(broken)

    broken = {**catalog, "entries": [dict(catalog["entries"][0])]}
    broken["entries"][0]["license"] = {
        "name": "unknown",
        "evidence": "https://example.com/license",
    }
    with pytest.raises(ValueError, match="verified license"):
        validate_catalog(broken)


def test_catalog_rejects_duplicate_ids() -> None:
    catalog, _ = load_catalog(CATALOG)
    duplicate = {**catalog, "entries": list(catalog["entries"][:2])}
    duplicate["entries"][1] = dict(duplicate["entries"][0])
    with pytest.raises(ValueError, match="duplicate catalog id"):
        validate_catalog(duplicate)


def test_training_plan_partitions_every_catalog_row() -> None:
    catalog, _ = load_catalog(CATALOG)
    plan = yaml.safe_load(TRAINING_PLAN.read_text(encoding="utf-8"))
    catalog_ids = {row["id"] for row in catalog["entries"]}
    train_ids = {row["id"] for row in catalog["entries"] if row["train_policy"] == "train"}
    assert set(plan["allowed_train_ids"]) == train_ids
    assert set(plan["forbidden_train_ids"]) == catalog_ids - train_ids
    assert not set(plan["allowed_train_ids"]) & set(plan["forbidden_train_ids"])
