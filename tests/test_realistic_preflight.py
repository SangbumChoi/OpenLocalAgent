from pathlib import Path

from localagent.eval.realistic_preflight import preflight_catalog


CATALOG = Path(__file__).parents[1] / "configs/data/realistic-agent-eval.catalog.yaml"


def test_preflight_is_fail_closed_and_split_aware() -> None:
    report = preflight_catalog(CATALOG)
    assert report["catalog_entries"] == 42
    assert report["counts"] == {
        "runnable": 4,
        "blocked": 38,
        "train_rows": 4,
        "evaluation_or_restricted_rows": 38,
    }
    assert set(report["runnable_ids"]) == {
        "androidcontrol",
        "android_in_the_wild",
        "xlam_function_calling",
        "mind2web_train",
    }
    assert "androidworld" in report["blocked_ids"]
    androidworld = next(row for row in report["rows"] if row["id"] == "androidworld")
    assert androidworld["runnable"] is False
    assert "integration_status:environment_runner_pending" in androidworld["blockers"]


def test_preflight_report_has_source_links_and_dependency_probes() -> None:
    report = preflight_catalog(CATALOG)
    assert all(row["source_url"].startswith("https://") for row in report["rows"])
    assert isinstance(report["dependency_probes"]["module:playwright"], bool)
    assert isinstance(report["dependency_probes"]["command:adb"], bool)
