"""Regression checks for the native ToolSandbox stress-smoke receipts."""

import json
from pathlib import Path


def test_toolsandbox_native_stress_receipts_are_fail_closed_and_matched() -> None:
    root = Path(__file__).parents[1] / "docs/paper/results/raw"
    stateful = json.loads((root / "m90-toolsandbox-native-stateful-m75-v1.json").read_text())
    control = json.loads((root / "m90-toolsandbox-native-public-m70-v1.json").read_text())
    for receipt in (stateful, control):
        assert receipt["benchmark_id"] == "toolsandbox"
        assert receipt["environment_executed"] is True
        assert receipt["official_split_verified"] is False
        assert receipt["verifier_executed"] is True
        assert receipt["task_count"] == 5
        assert receipt["success_count"] == 1
        assert receipt["success_rate"] == 0.2
        assert "multi-tool and multi-user-turn" in receipt["claim_boundary"]
        assert receipt["scripted_user_policy"].startswith("terminate_after_first_agent_response")
    assert stateful["checkpoint"]["sha256"] != control["checkpoint"]["sha256"]
    assert [row["scenario"] for row in stateful["scenarios"]] == [
        "send_message_with_contact_content_cellular_off",
        "turn_on_wifi_low_battery_mode",
        "find_current_city_insufficient_information",
        "remove_contact_by_phone_ambiguous",
        "search_message_with_recency_latest_multiple_user_turn",
    ]


def test_toolsandbox_interactive_receipts_bind_multi_step_protocol_and_fix_boundary() -> None:
    root = Path(__file__).parents[1] / "docs/paper/results/raw"
    stateful = json.loads(
        (root / "m91-toolsandbox-native-interactive-stateful-m75-v1.json").read_text()
    )
    control = json.loads(
        (root / "m91-toolsandbox-native-interactive-public-m70-v1.json").read_text()
    )
    projection = json.loads(
        (root / "m91-toolsandbox-native-interactive-public-projection-m59-v1.json").read_text()
    )
    for receipt in (stateful, control, projection):
        assert receipt["protocol"] == "bounded_multi_step_scripted_user"
        assert receipt["max_agent_turns"] == 8
        assert receipt["scripted_user_policy"].startswith("terminate_after_agent_final_text")
        assert "multi-step" not in receipt["claim_boundary"]
        assert receipt["task_count"] == 5
        assert receipt["success_count"] == 1
        assert all(row["exception"] is None for row in receipt["scenarios"])
    assert len({receipt["checkpoint"]["sha256"] for receipt in (stateful, control, projection)}) == 3


def test_toolsandbox_native_hardening_receipts_bind_matched_partial_progress() -> None:
    root = Path(__file__).parents[1] / "docs/paper/results/raw"
    receipts = [
        json.loads((root / name).read_text())
        for name in (
            "m92-toolsandbox-native-interactive-stateful-m75-v1.json",
            "m92-toolsandbox-native-interactive-public-m70-v1.json",
            "m92-toolsandbox-native-interactive-public-projection-m59-v1.json",
        )
    ]
    assert len({receipt["runner"]["sha256"] for receipt in receipts}) == 1
    for receipt in receipts:
        assert receipt["success_count"] == 1
        assert receipt["success_rate"] == 0.2
        assert [row["similarity"] for row in receipt["scenarios"]] == [
            0.25,
            1 / 3,
            1.0,
            0.0,
            0.5,
        ]
        assert all(row["exception"] is None for row in receipt["scenarios"])


def test_toolsandbox_native_mcp_transfer_triad_is_matched_and_negative() -> None:
    root = Path(__file__).parents[1] / "docs/paper/results/raw"
    receipts = [
        json.loads((root / name).read_text())
        for name in (
            "m97-toolsandbox-native-interactive-mcp-lowrate-v1.json",
            "m98-toolsandbox-native-interactive-mcp-frozen-v1.json",
            "m99-toolsandbox-native-interactive-mcp-random-v1.json",
        )
    ]
    assert len({receipt["source_revision"] for receipt in receipts}) == 1
    assert len({receipt["runner"]["sha256"] for receipt in receipts}) == 1
    assert len({tuple(row["scenario"] for row in receipt["scenarios"]) for receipt in receipts}) == 1
    assert all(receipt["official_split_verified"] is False for receipt in receipts)
    assert all(receipt["user_simulator_executed"] is False for receipt in receipts)
    assert all(receipt["verifier_executed"] is True for receipt in receipts)
    assert [receipt["success_count"] for receipt in receipts] == [1, 1, 1]
    assert [receipt["success_rate"] for receipt in receipts] == [0.2, 0.2, 0.2]
