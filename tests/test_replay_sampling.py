from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from openlocalagent.data.corpus.decision_quota_order import order_assistant_decisions
from openlocalagent.data.synth.format_bootstrap import FORMAT_BOOTSTRAP_PHASES
from openlocalagent.data.schema import Conversation, Message, Role, ToolCall, ToolSpec
from openlocalagent.train.replay_sampling import (
    CENTERED_UPDATE_QUANTILES_CONTRACT,
    GENERAL_COVERAGE_SPREAD_CONTRACT,
    MIXED_REPLAY_ORDERING_CONTRACT,
    MIXED_REPLAY_SAMPLING_MODE,
    PARENT_ANCHORED_FORMAT_PULSE_ORDERING_CONTRACT,
    PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE,
    PHASE_ROUND_ROBIN_CONTRACT,
    _spread_quota_coverage,
    mixed_replay_sampling_window,
    parent_anchored_format_pulse_sampling_window,
)


def _tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="lookup",
            description="Look up a query.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="route",
            description="Route an item.",
            parameters={
                "type": "object",
                "properties": {
                    "destination": {"type": "string"},
                    "item": {"type": "string"},
                },
                "required": ["destination", "item"],
                "additionalProperties": False,
            },
        ),
    ]


def _conversation(
    prompt: str,
    *,
    calls: list[ToolCall] | None = None,
    text: str = "",
    category: str = "fixture",
) -> Conversation:
    return Conversation(
        messages=[
            Message(role=Role.user, content=prompt),
            Message(role=Role.assistant, content=text, tool_calls=calls or []),
        ],
        tools=_tools(),
        meta={"category": category, "group": "test", "kind": "tool"},
    )


def _format_rows() -> list[Conversation]:
    return [
        _conversation(
            "lookup format core",
            calls=[ToolCall(name="lookup", arguments={"query": "format core"})],
        ),
        _conversation(
            "route multiple arguments",
            calls=[
                ToolCall(
                    name="route",
                    arguments={"destination": "lab", "item": "sample"},
                )
            ],
        ),
        _conversation(
            "lookup and route in parallel",
            calls=[
                ToolCall(name="lookup", arguments={"query": "parallel"}),
                ToolCall(
                    name="route",
                    arguments={"destination": "lab", "item": "parallel"},
                ),
            ],
        ),
        _conversation("say hello", text="Hello.", category="text"),
    ]


def _sampling_config() -> dict:
    return {
        "mode": MIXED_REPLAY_SAMPLING_MODE,
        "general_source_index": 0,
        "format_source_index": 1,
        "exclude_format_semantic_overlap": True,
        "cycle": [
            "general",
            "general",
            "general",
            "format_core",
            "general",
            "general",
            "general",
            "multi_argument",
            "general",
            "general",
            "general",
            "parallel",
            "general",
            "general",
            "general",
            "text",
        ],
    }


def _ordered_key_sha256(keys: list[tuple[int, int]]) -> str:
    payload = json.dumps(
        [[conversation_index, message_index] for conversation_index, message_index in keys],
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ordered_block_sha256(blocks: list[list[tuple[int, int]]]) -> str:
    payload = json.dumps(
        [
            [[conversation_index, message_index] for conversation_index, message_index in block]
            for block in blocks
        ],
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parent_anchored_fixture() -> tuple[
    list[Conversation],
    list[Conversation],
    dict,
]:
    format_rows = _format_rows() * 2
    general_rows = list(format_rows)
    ordering = order_assistant_decisions(general_rows)
    return (
        general_rows,
        format_rows,
        {
            "mode": PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE,
            "general_source_index": 0,
            "format_source_index": 1,
            "parent_prefix_decisions": 8,
            "update_decisions": 4,
            "expected_parent_order_sha256": ordering.audit.order_sha256,
            "expected_parent_prefix_sha256": _ordered_key_sha256(list(ordering.keys)),
            "format_pulses": {
                "count": 2,
                "rows_per_phase": 1,
                "phase_order": list(FORMAT_BOOTSTRAP_PHASES),
                "within_pulse_order": PHASE_ROUND_ROBIN_CONTRACT,
                "position_contract": CENTERED_UPDATE_QUANTILES_CONTRACT,
            },
        },
    )


def test_mixed_replay_builds_exact_non_overlapping_per_update_cycle() -> None:
    format_rows = _format_rows()
    broad_rows = [
        _conversation(
            f"general query {index}",
            calls=[ToolCall(name="lookup", arguments={"query": f"general {index}"})],
        )
        for index in range(12)
    ]
    general_rows = [*format_rows, *broad_rows]

    order, contract = mixed_replay_sampling_window(
        [general_rows, format_rows],
        selected_decisions=16,
        sampling_config=_sampling_config(),
    )

    assert len(order) == 20
    assert len(set(order)) == 20
    assert contract["contract"] == MIXED_REPLAY_ORDERING_CONTRACT
    assert contract["selected_decisions"] == 16
    assert contract["selected_unique_decisions"] == 16
    assert contract["selected_general_format_semantic_overlap"] == 0
    assert contract["general"]["excluded_format_semantic_conversations"] == 4
    assert contract["general"]["selected_decisions"] == 12
    assert contract["general"]["coverage_spread"] == {
        "contract": GENERAL_COVERAGE_SPREAD_CONTRACT,
        "coverage_decisions": 1,
        "coverage_position_contract": (
            "floor((2*i+1)*selected_decisions/(2*coverage_decisions)), zero-based i"
        ),
        "coverage_position_index_base": 1,
        "coverage_positions_sha256_encoding": (
            "sha256(canonical compact JSON array of one-based integer positions)"
        ),
        "coverage_positions_sha256": (
            "589ffd3acf522ae81192579f297589e93b0c41f748b224d1b4bb5c857a2f70cb"
        ),
        "first_coverage_position": 7,
        "last_coverage_position": 7,
        "max_coverage_decisions_per_cycle": 1,
        "min_coverage_decisions_per_cycle": 1,
        "selected_decisions": 12,
        "selected_order_sha256": contract["general"]["coverage_spread"]["selected_order_sha256"],
    }
    assert contract["format"]["selected_by_phase"] == {
        "format_core": 1,
        "multi_argument": 1,
        "parallel": 1,
        "text": 1,
    }

    # The second source begins after all 16 general-source Conversations.
    assert order[3] == (16, 1)
    assert order[7] == (17, 1)
    assert order[11] == (18, 1)
    assert order[15] == (19, 1)
    assert all(key[0] >= 4 for index, key in enumerate(order[:16]) if index % 4 != 3)
    emitted_general = [key for index, key in enumerate(order[:16]) if index % 4 != 3]
    assert contract["general"]["coverage_spread"]["selected_order_sha256"] == (
        _ordered_key_sha256(emitted_general)
    )
    repeated_order, repeated_contract = mixed_replay_sampling_window(
        [general_rows, format_rows],
        selected_decisions=16,
        sampling_config=_sampling_config(),
    )
    assert repeated_order == order
    assert repeated_contract == contract


def test_mixed_replay_spreads_required_strata_across_every_cycle() -> None:
    format_rows = _format_rows()
    general_rows = [
        _conversation(
            f"general query {index}",
            calls=[ToolCall(name="lookup", arguments={"query": f"general {index}"})],
            category=f"fixture-{index % 8}",
        )
        for index in range(24)
    ]

    order, contract = mixed_replay_sampling_window(
        [general_rows, format_rows * 2],
        selected_decisions=32,
        sampling_config=_sampling_config(),
    )

    spread = contract["general"]["coverage_spread"]
    assert spread["coverage_decisions"] == 8
    assert spread["selected_decisions"] == 24
    assert spread["first_coverage_position"] == 2
    assert spread["last_coverage_position"] == 23
    assert spread["min_coverage_decisions_per_cycle"] == 4
    assert spread["max_coverage_decisions_per_cycle"] == 4
    assert len(order) == 32


def test_real_horizon_coverage_spread_has_six_or_seven_rows_per_cycle() -> None:
    selected = tuple((index, 1) for index in range(3_072))
    spread, contract = _spread_quota_coverage(
        selected,
        coverage_decisions=1_691,
        general_per_cycle=12,
    )

    coverage = set(selected[:1_691])
    per_cycle = [
        sum(key in coverage for key in spread[start : start + 12])
        for start in range(0, len(spread), 12)
    ]
    assert per_cycle.count(7) == 155
    assert per_cycle.count(6) == 101
    assert set(per_cycle) == {6, 7}
    assert contract["min_coverage_decisions_per_cycle"] == 6
    assert contract["max_coverage_decisions_per_cycle"] == 7
    assert contract["first_coverage_position"] == 1
    assert contract["last_coverage_position"] == 3_072
    assert contract["coverage_positions_sha256"] == (
        "281d3ba6fc7f2875b5032aae5d8fab4ef167664f2c8355666c5b13fb08438296"
    )
    assert [key for key in spread if key not in coverage] == list(selected[1_691:])
    assert set(spread) == set(selected)
    assert len(set(spread)) == len(selected)


def test_coverage_spread_handles_full_coverage_and_rejects_duplicate_keys() -> None:
    selected = ((0, 1), (1, 1), (2, 1))
    spread, contract = _spread_quota_coverage(
        selected,
        coverage_decisions=len(selected),
        general_per_cycle=len(selected),
    )
    assert spread == selected
    assert contract["min_coverage_decisions_per_cycle"] == len(selected)
    assert contract["max_coverage_decisions_per_cycle"] == len(selected)

    with pytest.raises(RuntimeError, match="changed the selected general decision set"):
        _spread_quota_coverage(
            ((0, 1), (0, 1)),
            coverage_decisions=1,
            general_per_cycle=2,
        )


def test_mixed_replay_rejects_partial_cycles_and_incomplete_phase_cycles() -> None:
    format_rows = _format_rows()
    general_rows = [
        _conversation(
            f"general query {index}",
            calls=[ToolCall(name="lookup", arguments={"query": f"general {index}"})],
        )
        for index in range(20)
    ]
    with pytest.raises(ValueError, match="integral number"):
        mixed_replay_sampling_window(
            [general_rows, format_rows],
            selected_decisions=15,
            sampling_config=_sampling_config(),
        )

    invalid = _sampling_config()
    invalid["cycle"] = invalid["cycle"][:-1]
    with pytest.raises(ValueError, match="cycle labels mismatch"):
        mixed_replay_sampling_window(
            [general_rows, format_rows],
            selected_decisions=15,
            sampling_config=invalid,
        )


def test_parent_anchored_format_pulses_preserve_parent_blocks_and_allow_overlap() -> None:
    general_rows, format_rows, config = _parent_anchored_fixture()

    order, contract = parent_anchored_format_pulse_sampling_window(
        [general_rows, format_rows],
        selected_decisions=16,
        sampling_config=config,
    )

    assert len(order) == 16
    assert len(set(order)) == 16
    assert contract["contract"] == PARENT_ANCHORED_FORMAT_PULSE_ORDERING_CONTRACT
    assert contract["mode"] == PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE
    assert contract["selected_unique_decisions"] == 16
    assert contract["complete_decisions"] == 16
    assert contract["update_layout"]["pulse_positions_zero_based"] == [1, 3]
    assert contract["update_layout"]["pulse_positions_one_based"] == [2, 4]
    assert contract["update_layout"]["pulse_positions_sha256"] == (
        "f25ed8cd5464710aff6e5d655a575ac273cb9c19d696e53fa13de05db97454ca"
    )

    parent_ordering = order_assistant_decisions(general_rows)
    expected_parent_blocks = [
        list(parent_ordering.keys[:4]),
        list(parent_ordering.keys[4:8]),
    ]
    expected_pulse_blocks = [
        [(8, 1), (9, 1), (10, 1), (11, 1)],
        [(12, 1), (13, 1), (14, 1), (15, 1)],
    ]
    expected_updates = [
        expected_parent_blocks[0],
        expected_pulse_blocks[0],
        expected_parent_blocks[1],
        expected_pulse_blocks[1],
    ]
    assert list(order[:4]) == expected_parent_blocks[0]
    assert list(order[4:8]) == expected_pulse_blocks[0]
    assert list(order[8:12]) == expected_parent_blocks[1]
    assert list(order[12:16]) == expected_pulse_blocks[1]
    assert contract["parent"]["update_blocks_sha256"] == _ordered_block_sha256(
        expected_parent_blocks
    )
    assert contract["format_pulses"]["pulse_blocks_sha256"] == _ordered_block_sha256(
        expected_pulse_blocks
    )
    assert contract["update_layout"]["interleaved_update_blocks_sha256"] == (
        _ordered_block_sha256(expected_updates)
    )
    assert contract["selected_parent_format_semantic_overlap"] == {
        "allowed": True,
        "counting_unit": (
            "selected format rows whose Conversation semantic SHA-256 occurs "
            "in the selected parent prefix"
        ),
        "by_phase": {
            "format_core": 2,
            "multi_argument": 2,
            "parallel": 2,
            "text": 2,
        },
        "total": 8,
    }

    for selected_row_index, key in enumerate(order):
        update_index = selected_row_index // 4
        if update_index in {1, 3}:
            assert key[0] >= 8
            expected_phase = FORMAT_BOOTSTRAP_PHASES[selected_row_index % 4]
            assert (key[0] - 8) % 4 == FORMAT_BOOTSTRAP_PHASES.index(expected_phase)
        else:
            assert key[0] < 8

    repeated_order, repeated_contract = parent_anchored_format_pulse_sampling_window(
        [general_rows, format_rows],
        selected_decisions=16,
        sampling_config=config,
    )
    assert repeated_order == order
    assert repeated_contract == contract


@pytest.mark.parametrize(
    ("path", "value", "error", "match"),
    [
        (
            ("extra",),
            True,
            ValueError,
            "data.sampling keys mismatch",
        ),
        (
            ("parent_prefix_decisions",),
            True,
            TypeError,
            "parent_prefix_decisions must be an integer",
        ),
        (
            ("format_pulses", "extra"),
            True,
            ValueError,
            "data.sampling.format_pulses keys mismatch",
        ),
        (
            ("format_pulses", "phase_order"),
            [
                "multi_argument",
                "format_core",
                "parallel",
                "text",
            ],
            ValueError,
            "phase_order must be exactly",
        ),
        (
            ("format_pulses", "within_pulse_order"),
            "unknown",
            ValueError,
            "within_pulse_order must be",
        ),
        (
            ("format_pulses", "position_contract"),
            "unknown",
            ValueError,
            "position_contract must be",
        ),
        (
            ("expected_parent_order_sha256",),
            "A" * 64,
            ValueError,
            "lowercase hexadecimal SHA-256",
        ),
    ],
)
def test_parent_anchored_format_pulses_reject_noncanonical_config(
    path: tuple[str, ...],
    value: object,
    error: type[Exception],
    match: str,
) -> None:
    general_rows, format_rows, config = _parent_anchored_fixture()
    invalid = deepcopy(config)
    target = invalid
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(error, match=match):
        parent_anchored_format_pulse_sampling_window(
            [general_rows, format_rows],
            selected_decisions=16,
            sampling_config=invalid,
        )


def test_parent_anchored_format_pulses_reject_inconsistent_horizons_and_hashes() -> None:
    general_rows, format_rows, config = _parent_anchored_fixture()

    with pytest.raises(ValueError, match="parent prefix plus complete format pulses"):
        parent_anchored_format_pulse_sampling_window(
            [general_rows, format_rows],
            selected_decisions=15,
            sampling_config=config,
        )

    invalid = deepcopy(config)
    invalid["parent_prefix_decisions"] = 7
    with pytest.raises(ValueError, match="complete parent updates"):
        parent_anchored_format_pulse_sampling_window(
            [general_rows, format_rows],
            selected_decisions=15,
            sampling_config=invalid,
        )

    invalid = deepcopy(config)
    invalid["update_decisions"] = 8
    with pytest.raises(ValueError, match="rows must sum to one complete update"):
        parent_anchored_format_pulse_sampling_window(
            [general_rows, format_rows],
            selected_decisions=24,
            sampling_config=invalid,
        )

    invalid = deepcopy(config)
    invalid["expected_parent_order_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="parent quota order SHA-256 mismatch"):
        parent_anchored_format_pulse_sampling_window(
            [general_rows, format_rows],
            selected_decisions=16,
            sampling_config=invalid,
        )

    invalid = deepcopy(config)
    invalid["expected_parent_prefix_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="parent quota prefix SHA-256 mismatch"):
        parent_anchored_format_pulse_sampling_window(
            [general_rows, format_rows],
            selected_decisions=16,
            sampling_config=invalid,
        )


def test_parent_anchored_format_pulses_reject_capacity_and_source_aliasing() -> None:
    general_rows, format_rows, config = _parent_anchored_fixture()
    with pytest.raises(ValueError, match="format phase is too small"):
        parent_anchored_format_pulse_sampling_window(
            [general_rows, format_rows[:4]],
            selected_decisions=16,
            sampling_config=config,
        )

    invalid = deepcopy(config)
    invalid["format_source_index"] = 0
    with pytest.raises(ValueError, match="sources must be distinct"):
        parent_anchored_format_pulse_sampling_window(
            [general_rows, format_rows],
            selected_decisions=16,
            sampling_config=invalid,
        )


_PAPER_GENERAL = Path("data/synth/agent_sft_paper_train_v2.jsonl")
_PAPER_FORMAT = Path("data/synth/agent_sft_format_bootstrap_v1.jsonl")


@pytest.mark.skipif(
    not (_PAPER_GENERAL.is_file() and _PAPER_FORMAT.is_file()),
    reason="sealed production SFT artifacts are not present",
)
def test_parent_anchored_production_window_pins_every_reference_hash() -> None:
    def load(path: Path) -> list[Conversation]:
        with path.open(encoding="utf-8") as handle:
            return [Conversation.from_json(line) for line in handle]

    general_rows = load(_PAPER_GENERAL)
    format_rows = load(_PAPER_FORMAT)
    pulse_positions_zero_based = [
        7,
        23,
        38,
        54,
        69,
        85,
        100,
        116,
        131,
        147,
        162,
        178,
        193,
        209,
        224,
        240,
        255,
        271,
        286,
        302,
        317,
        333,
        348,
        364,
    ]
    config = {
        "mode": PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE,
        "general_source_index": 0,
        "format_source_index": 1,
        "parent_prefix_decisions": 5_568,
        "update_decisions": 16,
        "expected_parent_order_sha256": (
            "b1f78dc4b3a08647fa91f1643c40c625804ee63c38728f198dc7fa013aad5c9d"
        ),
        "expected_parent_prefix_sha256": (
            "d7cf1bfcc61cf6b8882cfe149f78f7c2536855536f1190685d77569c5a3adedb"
        ),
        "format_pulses": {
            "count": 24,
            "rows_per_phase": 4,
            "phase_order": list(FORMAT_BOOTSTRAP_PHASES),
            "within_pulse_order": PHASE_ROUND_ROBIN_CONTRACT,
            "position_contract": CENTERED_UPDATE_QUANTILES_CONTRACT,
        },
    }

    order, contract = parent_anchored_format_pulse_sampling_window(
        [general_rows, format_rows],
        selected_decisions=5_952,
        sampling_config=config,
    )

    assert len(order) == 101_696
    assert len(set(order)) == 101_696
    assert contract["selected_decisions"] == 5_952
    assert contract["complete_decisions"] == 101_696
    assert contract["parent"]["source_conversations"] == 50_000
    assert contract["parent"]["source_decisions"] == 93_504
    assert contract["parent"]["order_sha256"] == config["expected_parent_order_sha256"]
    assert contract["parent"]["prefix_sha256"] == config["expected_parent_prefix_sha256"]
    assert contract["parent"]["update_blocks"] == 348
    assert contract["parent"]["update_blocks_sha256"] == (
        "3c1845c09146a8175a12d00dd0e349039a746f86dc23f17370c68eabd4924662"
    )
    assert contract["format_pulses"]["source_conversations"] == 8_192
    assert contract["format_pulses"]["available_by_phase"] == {
        "format_core": 4_096,
        "multi_argument": 1_536,
        "parallel": 1_024,
        "text": 1_536,
    }
    assert contract["format_pulses"]["selected_by_phase"] == {
        phase: 96 for phase in FORMAT_BOOTSTRAP_PHASES
    }
    assert contract["format_pulses"]["selected_phase_local_order_sha256"] == {
        "format_core": "656df421b53f0fd91efde071678f7192371f0ad3ac5ee3fc456412f236fe7814",
        "multi_argument": "3b06576552bb75dee4bc80d7cc7e77aaf36f226eb06cba2b4a591b45d6f3d53f",
        "parallel": "92040e57c520f7d9fceaf1fa125a4b2c06db931ec5e8fd74046b44e469021660",
        "text": "f74dbd5099bcc7a8958d01e45c709e41593063043bc6cd0d6465514208e8897f",
    }
    assert contract["format_pulses"]["selected_phase_global_order_sha256"] == {
        "format_core": "a9f5cd7716f8d6a7065f892b849427573d9dd1e59af88b49f871c42cebde7b26",
        "multi_argument": "75b39f052e53108cea9f8116a50b6e10fbcfa95340c503586686fcd8f7138c7b",
        "parallel": "d80e46c6e2f277849caf9e06e4b471e2f7767d7b1169e56b3262fe8b7a8fe28d",
        "text": "b3793d6c650f6f7a8a069333208c4496a6061852440f3e682dd17bcf898146bf",
    }
    assert contract["update_layout"]["pulse_positions_zero_based"] == (pulse_positions_zero_based)
    assert contract["update_layout"]["pulse_positions_one_based"] == [
        position + 1 for position in pulse_positions_zero_based
    ]
    assert contract["update_layout"]["pulse_positions_sha256"] == (
        "7624691cb56c7d3d7d8973b637b9626c2cc245a4985e6e1f3e3d0834bdb947db"
    )
    assert contract["format_pulses"]["pulse_blocks_sha256"] == (
        "f34e693c02537d7f9a98f650410312df96dcd108a367ee1e38de5a41c2883de0"
    )
    assert contract["update_layout"]["interleaved_update_blocks_sha256"] == (
        "c21853082b7a8b9214da6c76122bb4711b2d2afdbb0b41313e0d5364d6fbb1f5"
    )
    assert contract["selected_order_sha256"] == (
        "f0998ee1eefdb9cc82ecfd2866185db1bc00a8fe938da9fe873123e26b397329"
    )
    assert contract["complete_order_sha256"] == (
        "6ed3da1e15fcf390836c20cf3206c3c384bff7c76b18b6268b02d0246e1210a8"
    )
    assert contract["selected_parent_format_semantic_overlap"]["by_phase"] == {
        "format_core": 5,
        "multi_argument": 3,
        "parallel": 41,
        "text": 1,
    }
    assert contract["selected_parent_format_semantic_overlap"]["total"] == 50
