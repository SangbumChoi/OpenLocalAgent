import hashlib
import json
from pathlib import Path

from localagent.data.realistic_catalog import load_catalog


def test_clawbench_catalog_addendum_is_pinned_and_eval_only() -> None:
    catalog, _ = load_catalog("configs/data/realistic-agent-eval.clawbench.yaml")
    row = catalog["entries"][0]
    assert row["id"] == "clawbench"
    assert row["source_revision"] == "cc146e2128724f47f2a7246f1a3057c643b22f70"
    assert row["train_policy"] == "eval_only"
    assert row["scale"]["v1_tasks"] == 153
    assert row["scale"]["v2_tasks"] == 130


def test_clawbench_source_audit_is_self_consistent() -> None:
    path = Path("docs/paper/results/raw/m621-clawbench-source-audit-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["corpora"]["v1"]["tasks"] == 153
    assert payload["corpora"]["v2"]["tasks"] == 130
    assert payload["corpora"]["v1"]["interception_placeholder_tasks"] == 71
    assert payload["corpora"]["v2"]["interception_placeholder_tasks"] == 0
    assert payload["cross_corpus"]["v1_v2_task_id_disjoint"] is False
    assert payload["evaluation_boundary"]["train_policy"] == "eval_only"
