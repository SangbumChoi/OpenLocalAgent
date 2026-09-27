import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m359-public-realistic-eval-matrix-audit-v1.json"


def _self_hash(payload: dict) -> str:
    body = dict(payload)
    body.pop("receipt_self_sha256")
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def test_m359_binds_current_matrix_and_all_five_families() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["kind"] == "localagent_public_realistic_eval_matrix_audit"
    assert payload["matrix"]["entries"] == 27
    assert payload["matrix"]["canonical_sha256"] == (
        "5ed9202ebaa70ecc7dc73a814d77bb5308eb1281449d77dc1274ce8e07a17aef"
    )
    assert set(payload["counts"]["families"]) == {
        "mobile",
        "browser",
        "computer",
        "tool_api",
        "terminal",
    }
    assert {row["id"] for row in payload["train_eligible"]} == {
        "androidcontrol",
        "android_in_the_wild",
        "mind2web",
        "agentnet",
        "toolace",
    }


def test_m359_preserves_source_and_runtime_claim_boundary() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert len(payload["sources"]) == 27
    assert all(row["source_url"].startswith("https://") for row in payload["sources"])
    assert all(row["paper_url"].startswith("https://") for row in payload["sources"])
    boundary = payload["claim_boundary"]
    assert "No benchmark task text" in boundary
    assert "release-matched native execution" in boundary
