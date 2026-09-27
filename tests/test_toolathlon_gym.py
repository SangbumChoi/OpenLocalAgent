import json
import hashlib
from pathlib import Path

import pytest

from localagent.data.toolathlon_gym import profile_toolathlon_gym


def _write_config(root, name, servers, local_tools=None):
    path = root / "tasks" / "finalpool" / name / "task_config.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "needed_mcp_servers": servers,
                "needed_local_tools": local_tools or [],
                "meta": {},
            }
        ),
        encoding="utf-8",
    )


def test_toolathlon_profile_reads_only_config_inventory(tmp_path) -> None:
    _write_config(tmp_path, "email-notion", ["emails", "notion", "filesystem"], ["claim_done"])
    _write_config(tmp_path, "browser", ["playwright_with_chunk", "filesystem"])
    (tmp_path / "tasks/finalpool/email-notion/docs").mkdir(parents=True)
    (tmp_path / "tasks/finalpool/email-notion/docs/task.md").write_text(
        "secret benchmark prompt", encoding="utf-8"
    )
    profile = profile_toolathlon_gym(tmp_path, revision="test-revision")
    assert profile["source"]["config_files"] == 2
    assert profile["mcp_server_counts"] == {
        "emails": 1,
        "filesystem": 2,
        "notion": 1,
        "playwright_with_chunk": 1,
    }
    assert profile["realistic_surface_counts"] == {
        "browser_playwright_tasks": 1,
        "calendar_tasks": 0,
        "email_tasks": 1,
        "filesystem_tasks": 2,
        "notion_tasks": 1,
    }
    assert profile["source"]["descriptions_consumed"] is False
    assert profile["source"]["evaluators_consumed"] is False
    assert len(profile["profile_sha256"]) == 64


def test_toolathlon_profile_rejects_duplicate_servers(tmp_path) -> None:
    _write_config(tmp_path, "bad", ["emails", "emails"])
    with pytest.raises(ValueError, match="must not contain duplicates"):
        profile_toolathlon_gym(tmp_path)


def test_toolathlon_profile_requires_finalpool(tmp_path) -> None:
    with pytest.raises(ValueError, match="missing"):
        profile_toolathlon_gym(tmp_path)


def test_pinned_toolathlon_profile_receipt_is_self_hashed() -> None:
    receipt_path = Path(__file__).parents[1] / (
        "docs/paper/results/raw/m20-toolathlon-gym-config-profile-v1.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = receipt.pop("profile_sha256")
    actual = hashlib.sha256(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert actual == expected
    assert receipt["source"]["config_files"] == 503
    assert receipt["realistic_surface_counts"]["email_tasks"] == 258
