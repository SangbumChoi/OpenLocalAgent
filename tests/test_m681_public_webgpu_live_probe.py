import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_m681_public_webgpu_probe_is_legacy_and_fail_closed() -> None:
    path = ROOT / "docs/paper/results/raw/m681-public-webgpu-live-probe-v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert claimed == actual
    release = payload["public_release"]
    assert release["space_runtime"] == "RUNNING"
    assert release["current_checkpoint_match"] is False
    assert release["legacy_public_release"] is True
    probe = payload["browser_probe"]
    assert probe["model_ready"] is True
    assert probe["provider"] == "WEBGPU"
    assert probe["tool_route_correct"] is False
    assert probe["external_write_performed"] is False
