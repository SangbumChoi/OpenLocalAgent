from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from localagent.eval.structured_context import (
    _score_rows,
    canonical_sha256,
    evaluate_decisions,
    materialize_context_ids,
    materialize_context_view,
)
from localagent.model.tokenizer import ASSISTANT, USER, ByteTokenizer
from localagent.train.stage_data import ProbeDecision


class _RouteHead(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, :5]


class _SelectorModel(torch.nn.Module):
    def forward(self, features: torch.Tensor, tool_embeddings: torch.Tensor) -> torch.Tensor:
        return features[:, 5:] @ tool_embeddings.T


class _Selector:
    def __init__(self) -> None:
        self.names = ["click", "web_search"]
        self.model = _SelectorModel()
        self.embs = torch.eye(2)


class _ContextModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(d_model=7, max_seq_len=256)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.zeros((*inputs.shape, self.cfg.d_model), device=inputs.device)
        hidden[..., 0] = 1.0
        logits = torch.zeros((*inputs.shape, 2), device=inputs.device)
        return logits, hidden


def test_materialize_context_ids_matches_browser_padding_contract() -> None:
    tokenizer = ByteTokenizer()
    text = f"{USER}Select the button.{ASSISTANT}"
    natural = tokenizer.encode(text)
    assistant_ids = tokenizer.encode(ASSISTANT)
    target = len(natural) + 7

    padded = materialize_context_ids(
        tokenizer,
        text,
        target_input_tokens=target,
        max_seq_len=256,
    )

    assert padded is not None
    assert len(padded) == target
    assert padded[-len(assistant_ids) :] == assistant_ids
    assert padded[
        len(natural) - len(assistant_ids) : -len(assistant_ids)
    ] == tokenizer.encode(" ") * 7
    assert (
        materialize_context_ids(
            tokenizer,
            text,
            target_input_tokens=len(natural) - 1,
            max_seq_len=256,
        )
        is None
    )

    trailing = materialize_context_view(
        tokenizer,
        text,
        target_input_tokens=target,
        max_seq_len=256,
        materialization="trailing_compute",
    )
    assert trailing is not None
    trailing_ids, feature_index = trailing
    assert trailing_ids[: len(natural)] == natural
    assert trailing_ids[len(natural) :] == tokenizer.encode(" ") * 7
    assert feature_index == len(natural) - 1


def test_evaluate_decisions_optionally_returns_configured_row_records() -> None:
    result = evaluate_decisions(
        model=_ContextModel(),
        tokenizer=ByteTokenizer(),
        route_head=_RouteHead(),
        selector=_Selector(),
        decisions=[
            ProbeDecision(prompt="Choose it", kind="tool", ref_name="web_search"),
            ProbeDecision(prompt="Answer directly", kind="text"),
        ],
        target_input_tokens=None,
        batch_size=2,
        device="cpu",
        include_records=True,
    )

    assert [record["configured_index"] for record in result["records"]] == [0, 1]
    assert all("predicted_route" in record for record in result["records"])

    without_records = evaluate_decisions(
        model=_ContextModel(),
        tokenizer=ByteTokenizer(),
        route_head=_RouteHead(),
        selector=_Selector(),
        decisions=[ProbeDecision(prompt="Choose it", kind="tool", ref_name="web_search")],
        target_input_tokens=None,
        batch_size=1,
        device="cpu",
    )
    assert "records" not in without_records


def test_score_rows_separates_route_selector_and_dispatched_accuracy() -> None:
    features = torch.tensor(
        [
            [0.0, 5.0, 0.0, 0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 0.0, 5.0, 3.0, 0.0],
        ]
    )

    metrics, records = _score_rows(
        features=features,
        gold_routes=["computer_use", "web_search", "text"],
        gold_tools=["click", "web_search", None],
        route_head=_RouteHead(),
        selector=_Selector(),
    )

    assert metrics == {
        "eligible_rows": 3,
        "route_correct": 2,
        "route_accuracy": 2 / 3,
        "route_text_predictions": 2,
        "tool_rows": 2,
        "selector_top1_correct": 2,
        "selector_top1_accuracy": 1.0,
        "dispatched_tool_correct": 1,
        "dispatched_tool_accuracy": 0.5,
    }
    assert records[1]["selector_top1_correct"] is True
    assert records[1]["dispatched_tool_correct"] is False


def test_tracked_seed2027_context_audit_is_self_consistent() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "docs"
        / "paper"
        / "results"
        / "sft-structured-context-robustness-seed2027.summary.json"
    )
    payload = json.loads(path.read_text())
    expected_hash = canonical_sha256(
        {key: value for key, value in payload.items() if key != "summary_sha256"}
    )
    assert payload["summary_sha256"] == expected_hash
    assert payload["checkpoint"]["sha256"] == (
        "79387105de75d332413262e8d8ddb847b6cc13bc03f5e4df3c81663d9897aef1"
    )

    conditions = {row["condition"]: row for row in payload["conditions"]}
    natural = conditions["natural"]
    failed = conditions["fixed_pre_assistant_512"]
    corrected = conditions["fixed_trailing_compute_512"]

    assert natural["agent_eval"]["route_correct"] == 83
    assert natural["agent_eval"]["selector_top1_correct"] == 72
    assert failed["agent_eval"]["route_text_predictions"] == 98
    assert failed["action_suite"]["dispatched_tool_correct"] == 0
    assert corrected["agent_eval"]["route_correct"] == 83
    assert corrected["agent_eval"]["selector_top1_correct"] == 72
    assert corrected["action_suite"]["dispatched_tool_correct"] == 17
