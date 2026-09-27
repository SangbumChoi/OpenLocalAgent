import hashlib
import json
from pathlib import Path

import pytest

from localagent.data.mcpmark import profile_mcpmark


def _write_meta(root, service, suite, category, task, task_id="task-1"):
    path = root / "tasks" / service / suite / category / task / "meta.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "task_name": "Prompt must not enter profile",
                "category_id": category,
                "category_name": category.title(),
                "description": "secret benchmark prompt",
                "difficulty": "L1",
                "tags": ["content submission", "data extraction"],
                "mcp": [service],
                "meta_data": {"stateType": "text"},
            }
        ),
        encoding="utf-8",
    )


def test_mcpmark_profile_tracks_suites_and_services_without_prompt_text(tmp_path) -> None:
    _write_meta(tmp_path, "notion", "standard", "pages", "one", "one")
    _write_meta(tmp_path, "playwright", "easy", "browser", "two", "two")
    profile = profile_mcpmark(tmp_path, revision="test-revision")
    assert profile["source"]["metadata_files"] == 2
    assert profile["suite_counts"] == {"easy": 1, "standard": 1}
    assert profile["mcp_service_counts"] == {"notion": 1, "playwright": 1}
    assert profile["realistic_surface_counts"] == {
        "browser_tasks": 1,
        "database_tasks": 0,
        "filesystem_tasks": 0,
        "github_tasks": 0,
        "notion_tasks": 1,
    }
    assert profile["source"]["description_text_retained"] is False
    assert len(profile["profile_sha256"]) == 64


def test_mcpmark_profile_rejects_wrong_path_shape(tmp_path) -> None:
    path = tmp_path / "tasks/notion/standard/pages/meta.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="no task metadata"):
        profile_mcpmark(tmp_path)


def test_pinned_mcpmark_profile_receipt_is_self_hashed() -> None:
    receipt_path = Path(__file__).parents[1] / (
        "docs/paper/results/raw/m21-mcpmark-metadata-profile-v1.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = receipt.pop("profile_sha256")
    actual = hashlib.sha256(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert actual == expected
    assert receipt["source"]["metadata_files"] == 239
    assert receipt["realistic_surface_counts"]["notion_tasks"] == 38
