import json
from pathlib import Path

from scripts.normalize_agentdiff import normalize


def test_agentdiff_normalizer_canonicalizes_assertions_and_preserves_eval_policy(tmp_path: Path) -> None:
    source = tmp_path / "test.jsonl"
    source.write_text(
        json.dumps(
            {
                "question": "Create a calendar event.",
                "answer": '{"assertions": [{"b": 2, "a": 1}]}',
                "service": "calendar",
                "test_id": "calendar_1",
                "test_name": "Create event",
                "task_horizon": 2,
                "info": "{\"seed_template\":\"calendar_base\"}",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows, selection = normalize(source, split="test")
    assert len(rows) == 1
    assert rows[0].messages[-1].content == '{"assertions":[{"a":1,"b":2}]}'
    assert rows[0].meta["train_policy"] == "eval_only"
    assert rows[0].meta["service"] == "calendar"
    assert selection["services"] == {"calendar": 1}
