import json
from pathlib import Path

from scripts.normalize_agentworldbench import normalize


def test_agentworldbench_normalizer_preserves_history_and_is_eval_only(tmp_path: Path) -> None:
    source = tmp_path / "web_test.jsonl"
    source.write_text(
        json.dumps(
            {
                "task": "web",
                "id": "trajectory-1",
                "turn_idx": 2,
                "total_turns": 2,
                "prompt": ["Action one", "Action two"],
                "response": [["Observation one"], ["Observation two"]],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows, selection = normalize([source], max_per_domain=32)
    assert len(rows) == 1
    row = rows[0]
    assert [message.content for message in row.messages] == [
        "Action one",
        "Observation one",
        "Action two",
        "Observation two",
    ]
    assert row.meta["dataset"] == "Qwen/AgentWorldBench"
    assert row.meta["train_policy"] == "eval_only"
    assert row.meta["domain"] == "web"
    assert selection["domain_counts"] == {"web": 1}
