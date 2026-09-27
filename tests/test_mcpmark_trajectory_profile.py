import hashlib
import json
from pathlib import Path

from scripts.download_mcpmark_trajectory import SURFACES, select_paths
from scripts.profile_mcpmark_trajectory import profile


def test_m117_receipt_is_metadata_only_and_pairs_public_trace_events() -> None:
    receipt = json.loads(
        Path(
            "docs/paper/results/raw/m117-mcpmark-trajectory-metadata-v1.json"
        ).read_text(encoding="utf-8")
    )
    source = receipt["source"]
    assert receipt["dataset"]["license"] == "MIT"
    assert receipt["dataset"]["revision"] == "e50578f0ab904d8e6a7c576c387c1e76ae482c89"
    assert source["event_rows"] == 45
    assert source["tool_calls"] == source["paired_tool_outputs"] == 21
    assert source["metadata_only"] is True
    assert source["raw_text_retained"] is False
    assert source["training_used"] is False
    assert source["tools_replayed"] is False
    assert set(source["unique_tools"]) == {
        "list_allowed_directories",
        "list_directory",
        "move_file",
        "read_multiple_files",
        "read_text_file",
    }
    assert "content audit" in receipt["claim_boundary"]


def test_profile_accepts_legacy_user_event_and_rejects_unpaired_calls(tmp_path: Path) -> None:
    valid = tmp_path / "valid.json"
    valid.write_text(
        json.dumps(
            [
                {"role": "user", "content": "Do it."},
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "list_directory",
                    "arguments": '{"path":"/tmp"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": '{"ok":true}',
                },
            ]
        ),
        encoding="utf-8",
    )
    report = profile(valid, revision="test-revision")
    assert report["source"]["event_types"] == {
        "function_call": 1,
        "function_call_output": 1,
        "message": 1,
    }

    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        json.dumps(
            [
                {"role": "user", "content": "Do it."},
                {
                    "type": "function_call",
                    "call_id": "c2",
                    "name": "list_directory",
                    "arguments": {},
                },
            ]
        ),
        encoding="utf-8",
    )
    try:
        profile(invalid)
    except ValueError as error:
        assert "pairing" in str(error)
    else:  # pragma: no cover - the profiler must fail closed
        raise AssertionError("unpaired MCPMark call was accepted")


def test_select_paths_is_sorted_surface_balanced_and_source_disjoint() -> None:
    files = [
        f"root/model__{surface}/run-1/task-{index}/messages.json"
        for surface in SURFACES
        for index in range(4)
    ]
    files += ["root/model__filesystem/run-1/task-0/meta.json", "README.md"]
    selected = select_paths(files, train_per_surface=2, eval_per_surface=1)
    assert len(selected["train"]) == 2 * len(SURFACES)
    assert len(selected["eval"]) == len(SURFACES)
    assert set(selected["train"]).isdisjoint(selected["eval"])
    for surface in SURFACES:
        train = [path for path in selected["train"] if f"__{surface}/" in path]
        evaluation = [path for path in selected["eval"] if f"__{surface}/" in path]
        assert train == sorted(train)
        assert evaluation == [sorted(train + evaluation)[2]]


def test_m532_acquisition_manifest_is_pinned_and_surface_balanced() -> None:
    receipt = json.loads(
        Path("docs/paper/results/raw/m532-mcpmark-trajectory-acquisition-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["dataset"] == "Jakumetsu/mcpmark-trajectory-log"
    assert receipt["revision"] == "e50578f0ab904d8e6a7c576c387c1e76ae482c89"
    assert len(receipt["entries"]) == 15
    assert receipt["selection"]["train_per_surface"] == 2
    assert receipt["selection"]["eval_per_surface"] == 1
    assert receipt["content_policy"].startswith("Acquisition only")


def test_m533_reproduces_normalized_output_hashes() -> None:
    receipt = json.loads(
        Path(
            "docs/paper/results/raw/m533-mcpmark-acquisition-normalization-reproducibility-v1.json"
        ).read_text(encoding="utf-8")
    )
    body = dict(receipt)
    recorded = body.pop("receipt_self_sha256")
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == digest
    assert receipt["normalized_outputs"]["train"]["sha256"] == (
        "21322d7218dd1f5906c4ae22162f1ef3e9ecb988d5633ca6a8f6d2943b8314d6"
    )
    assert receipt["normalized_outputs"]["eval"]["sha256"] == (
        "34c8e24ec58994f97372d9702fca54709a62140a2fef4a4d5e3099f6f4dcca89"
    )
