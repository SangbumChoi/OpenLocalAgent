import hashlib
import json
from pathlib import Path


def test_m612_projection_receipt_is_self_consistent_and_eval_only() -> None:
    path = Path("docs/paper/results/raw/m612-m585-agentworldbench-text-projection-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["manifest"]["dataset"] == "Qwen/AgentWorldBench"
    assert payload["manifest"]["train_policy"] == "eval_only"
    assert payload["protocol"]["rows"] == 224
    assert payload["protocol"]["domains"] == [
        "android",
        "mcp",
        "os",
        "search",
        "swe",
        "terminal",
        "web",
    ]
    assert payload["metrics"]["overall"]["rows"] == 224
    assert payload["metrics"]["overall"]["assistant_sequence_accuracy"] == 0.0
    assert "not an official AgentWorldBench judge score" in payload["claim_boundary"]
