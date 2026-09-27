"""Integrity checks for the pinned MCPMark metadata-only source profile."""

import json
from pathlib import Path


PROFILE = Path("docs/paper/results/raw/m496-mcpmark-current-source-profile-v1.json")


def test_m496_pins_current_source_and_suite_counts() -> None:
    payload = json.loads(PROFILE.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_mcpmark_metadata_profile"
    assert payload["source_revision"] == "cd45b7f57923b9b3985467f5139927575f83141c"
    assert payload["source"]["metadata_files"] == 239
    assert payload["suite_counts"] == {"easy": 70, "standard": 169}
    assert payload["realistic_surface_counts"] == {
        "browser_tasks": 35,
        "database_tasks": 93,
        "filesystem_tasks": 40,
        "github_tasks": 33,
        "notion_tasks": 38,
    }


def test_m496_excludes_prompt_state_and_verifier_payloads() -> None:
    payload = json.loads(PROFILE.read_text(encoding="utf-8"))
    source = payload["source"]
    assert source["description_text_retained"] is False
    assert source["state_assets_consumed"] is False
    assert source["verifiers_consumed"] is False
