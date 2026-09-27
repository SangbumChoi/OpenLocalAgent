from pathlib import Path

from localagent.data.realistic_catalog import load_catalog


def test_agentdiff_catalog_addendum_is_pinned_and_fail_closed() -> None:
    path = Path("configs/data/realistic-agent-eval.agentdiff.yaml")
    catalog, _ = load_catalog(path)
    assert len(catalog["entries"]) == 1
    row = catalog["entries"][0]
    assert row["id"] == "agentdiff"
    assert row["source_revision"] == "4a96ea93a8d074daba93ded109f340da7fae2f70"
    assert row["train_policy"] == "eval_only"
    assert row["scale"]["train_tasks"] == 179
    assert row["scale"]["test_tasks"] == 45
