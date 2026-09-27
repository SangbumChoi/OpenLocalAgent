import hashlib
import json
from pathlib import Path

from localagent.data.realistic_catalog import load_catalog


def test_agentworldbench_addendum_is_pinned_and_eval_only() -> None:
    path = Path("configs/data/realistic-agent-eval.agentworldbench.yaml")
    catalog, _ = load_catalog(path)
    assert len(catalog["entries"]) == 1
    row = catalog["entries"][0]
    assert row["id"] == "agentworldbench"
    assert row["train_policy"] == "eval_only"
    assert row["source_revision"] == "6b8d28437042434dcdd168434227ca0de408c5ba"


def test_agentworldbench_addendum_audit_is_self_consistent() -> None:
    path = Path("docs/paper/results/raw/m616-agentworldbench-catalog-addendum-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["decision"]["training_admission"] is False
    assert payload["catalog"]["dataset"] == "Qwen/AgentWorldBench"
    assert payload["projection_receipt"]["rows"] == 224
