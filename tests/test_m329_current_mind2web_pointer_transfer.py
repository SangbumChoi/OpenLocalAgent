import json
from pathlib import Path

import torch

from localagent.agent.pointer_head import PTR_ARGS
from localagent.train.stage_data import canonical_sha256
from scripts.train_mind2web_browser_pointer import _pointer_args


ROOT = Path(__file__).parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m329-current-mind2web-pointer-transfer-v1.json"
WEIGHT = ROOT / "docs/paper/results/raw/m329-current-mind2web-pointer-weight-v1.json"


def test_m329_infers_only_the_known_legacy_pointer_vocab_and_records_negative_transfer() -> None:
    inferred = _pointer_args({"ptr_head": {"arg_emb.weight": torch.zeros(len(PTR_ARGS), 4)}})
    assert inferred[: len(PTR_ARGS)] == PTR_ARGS
    assert inferred[-2:] == ["target", "text"]

    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    assert expected == canonical_sha256(body)
    assert payload["parent"]["sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    assert payload["hyperparameters"]["seed"] == 2049
    assert payload["before"]["exact_span"] == 0.0
    assert payload["after"]["exact_span"] == 0.09523809523809523
    assert payload["decision"]["native_replay_required"] is True

    weight = json.loads(WEIGHT.read_text(encoding="utf-8"))
    assert weight["compatibility"]["config_mismatches"] == {}
    assert weight["compatibility"]["tokenizer_sha256_equal"] is True
    assert weight["groups"]["embedding"]["relative_delta_l2"] < 0.003
