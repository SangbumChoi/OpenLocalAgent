import hashlib
import json
from pathlib import Path

import torch

from scripts.train_grounded_mind2web import _pointer_args, _warm_pointer
from localagent.agent.pointer_head import PointerHead


RECEIPT = Path("docs/paper/results/raw/m196-m194-grounded-mind2web-vocab-fix-v1.json")


def test_m196_migrates_expanded_parent_pointer_vocabulary_without_dropping_rows() -> None:
    args = [
        "city", "query", "goal", "term", "song", "topic", "path", "pattern", "command",
        "message", "task", "duration", "title", "recipient", "url", "content", "summary",
        "to", "subject", "body", "app_name", "text", "target",
    ]
    source = PointerHead(8, args=args)
    parent = {"ptr_head": source.state_dict(), "ptr_args": args}
    target_args = _pointer_args(parent)
    migrated = _warm_pointer(parent, 8, target_args)
    assert target_args[-2:] == ["target_id", "value"]
    assert migrated["arg_emb.weight"].shape == (25, 8)
    assert torch.equal(migrated["arg_emb.weight"][args.index("subject")], source.arg_emb.weight[args.index("subject")])


def test_m196_receipt_records_zero_grounding_gain_and_is_not_promoted() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    assert expected == actual
    assert payload["evaluation"]["pointer_span_exact_before"] == 0.0
    assert payload["evaluation"]["pointer_span_exact_after"] == 0.0
    assert payload["weight_movement"]["backbone_reused"] is True
    assert payload["decision"]["adopt_for_webgpu"] is False
    assert "not an official Mind2Web test score" in payload["claim_boundary"]
