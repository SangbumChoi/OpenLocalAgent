import hashlib
import json
from pathlib import Path

from localagent.data.realistic_catalog import load_catalog


def test_clawsbench_addendum_is_eval_only_and_pinned() -> None:
    catalog, _ = load_catalog("configs/data/realistic-agent-eval.clawsbench.yaml")
    row = catalog["entries"][0]
    assert row["id"] == "clawsbench"
    assert row["train_policy"] == "eval_only"
    assert row["source_revision"] == "e7c45cc9ff486502176267c1294ac5809cf0700a"
    assert row["scale"]["safety_tasks"] == 24


def test_clawsbench_source_audit_is_self_consistent() -> None:
    path = Path("docs/paper/results/raw/m620-clawsbench-source-audit-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["task_metadata"]["tasks"] == 44
    assert payload["task_metadata"]["safety_tasks"] == 24
    assert payload["evaluation_boundary"]["public_environment_and_verifiers"] is False
    assert payload["evaluation_boundary"]["train_policy"] == "eval_only"
