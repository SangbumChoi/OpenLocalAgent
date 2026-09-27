from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.profile_enterpriseopsgym_metadata import _card_claims, profile


def _api() -> dict:
    features = [
        {"name": name}
        for name in (
            "task_id",
            "domain",
            "system_prompt",
            "user_prompt",
            "selected_tools",
            "restricted_tools",
            "mcp_endpoint",
            "number_of_runs",
            "reset_database_between_runs",
            "gym_servers_config",
            "verifiers",
        )
    ]
    splits = [
        {"name": name, "num_examples": index + 1, "num_bytes": 100 + index}
        for index, name in enumerate(
            ("calendar", "csm", "drive", "email", "hr", "hybrid", "itsm", "teams")
        )
    ]
    return {
        "id": "ServiceNow-AI/EnterpriseOps-Gym",
        "sha": "test-revision",
        "private": False,
        "gated": False,
        "cardData": {
            "license": "apache-2.0",
            "dataset_info": [
                {
                    "config_name": config,
                    "features": features,
                    "splits": splits,
                    "download_size": 1000,
                    "dataset_size": 2000,
                }
                for config in ("oracle", "plus_5_tools", "plus_10_tools", "plus_15_tools")
            ],
        },
    }


def _card() -> str:
    return """comprises **1,150 expert-curated tasks** spanning **8 enterprise domains**.
**512 tools** across 8 enterprise domains; **9.15 avg steps** per task (up to 34).
| Domain | Tasks | Avg Steps | Max Steps | Tools |
|--------|------:|----------:|----------:|------:|
| Calendar | 100 | 7.05 | 17 | 37 |
| CSM | 186 | 12.10 | 27 | 89 |
| Drive | 105 | 8.68 | 29 | 55 |
| Email | 104 | 6.25 | 22 | 79 |
| HR | 184 | 10.54 | 34 | 89 |
| ITSM | 181 | 9.00 | 31 | 93 |
| Teams | 100 | 9.41 | 18 | 70 |
| Hybrid | 155 | 7.79 | 19 | Multi-domain |
| **Total** | **1,115** | **9.15** | **34** | **512** |
"""


def test_card_claims_preserve_count_discrepancy() -> None:
    claims = _card_claims(_card())
    assert claims["about_task_count"] == 1150
    assert claims["domain_table_task_count"] == 1115
    assert claims["count_discrepancy"] is True
    assert claims["domains"]["email"]["tasks"] == 104


def test_profile_is_metadata_only_and_validates_api(tmp_path: Path) -> None:
    api = tmp_path / "api.json"
    card = tmp_path / "README.md"
    api.write_text(json.dumps(_api()), encoding="utf-8")
    card.write_text(_card(), encoding="utf-8")
    result = profile(api, card, revision="test-revision")
    assert result["config_inventory"]["oracle"]["total_examples"] == 36
    assert result["card_inventory"]["tool_count"] == 512
    assert result["source"]["parquet_rows_downloaded"] is False
    assert result["source"]["servers_started"] is False

    broken = _api()
    broken["gated"] = True
    api.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="public and ungated"):
        profile(api, card, revision="test-revision")
