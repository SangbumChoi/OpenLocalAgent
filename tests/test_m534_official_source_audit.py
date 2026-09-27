import hashlib
import json
from pathlib import Path


def test_m534_source_audit_is_self_hashed_and_covers_required_surfaces() -> None:
    payload = json.loads(
        Path(
            "docs/paper/results/raw/m534-official-realistic-source-integrity-audit-v1.json"
        ).read_text(encoding="utf-8")
    )
    recorded = payload.pop("receipt_self_sha256")
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == digest
    assert {item["id"] for item in payload["sources"]} == {
        "androidworld",
        "browsergym",
        "mcpmark",
        "osworld",
        "osworld_v2",
        "toolsandbox",
        "enterpriseopsgym",
    }
    assert "does not report model scores" in payload["claim_boundary"]
