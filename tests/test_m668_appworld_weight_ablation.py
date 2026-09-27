import hashlib
import json
from pathlib import Path

from localagent.eval.workshop_gate import build_workshop_gate


ROOT = Path(__file__).parents[1]


def test_m668_weight_envelope_is_canonical_and_checkpoint_bound() -> None:
    path = ROOT / "docs/paper/results/raw/m668-appworld-weight-ablation-v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert claimed == actual
    assert payload["ablation"]["matched_rows"] is True
    assert payload["held_out"]["parent_heads"]["teacher_forced"]["rows"] == 6
    assert payload["held_out"]["random"]["teacher_forced"]["rows"] == 6
    report = build_workshop_gate(
        ROOT / "configs/data/realistic-agent-eval.catalog.yaml",
        repo_root=ROOT,
        weight_reports=[path],
    )
    check = next(
        item
        for item in report["checks"]
        if item["requirement"] == "weights:transfer_and_no_transfer_ablation"
    )
    assert check["status"] == "pass"
