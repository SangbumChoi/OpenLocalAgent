"""Regression receipt for the action-tail selector/grounding continuation."""

import json
from pathlib import Path


RECEIPT = Path(
    "docs/paper/results/raw/m393-current-stateful-action-tail-lexical-grounding-v1.json"
)


def test_m393_is_current_checkpoint_bound_and_fail_closed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_stateful_runtime_evaluation"
    assert payload["checkpoint"]["sha256"] == (
        "c53c79ad068878966ab897b5bf708d34d832e960f64a303ec059c2e0c8b90bfd"
    )
    assert payload["configuration"]["lexical_weight"] == 0.5
    assert payload["configuration"]["top_m"] == 1
    assert payload["configuration"]["tool_pool_size"] == 63
    assert payload["oracle"]["task_complete_rate"] == 1.0
    assert payload["model"]["task_complete_rate"] == 0.8
    assert payload["model"]["by_family"]["email"]["task_complete_rate"] == 0.0
    assert payload["model"]["by_family"]["browser"]["task_complete_rate"] == 1.0
    assert payload["model"]["by_family"]["notion"]["task_complete_rate"] == 1.0
    assert payload["runtime"]["public_benchmark"] is False
    assert payload["runtime"]["external_accounts_used"] is False


def test_stateful_training_view_adds_only_train_action_paraphrases() -> None:
    from scripts import train_stateful_productivity_probe as probe
    from localagent.data.stateful_productivity import build_tasks

    rows = probe._rows(build_tasks("train"))
    augmented = probe._action_augmented_rows(rows)
    assert len(augmented) > len(rows)
    assert len(probe._rows(build_tasks("eval"))) == 16
    prompts = [row["sample"].prompt for row in augmented]
    assert any("Submit the drafted email." in prompt for prompt in prompts)
    assert any("Click Compose at x=120 y=220" in prompt for prompt in prompts)
