from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

import pytest

import openlocalagent.data.adapters.weblinx_prompts as weblinx_prompts
from openlocalagent.data.adapters.weblinx_prompts import (
    PRODUCTION_WEBLINX_REVISION,
    PRODUCTION_WEBLINX_SPLIT,
    WEBLINX_PROMPT_ADAPTER_VERSION,
    WebLINXSource,
    detect_weblinx_privacy_reasons,
    export_weblinx_prompt_rows,
    parse_weblinx_action_history,
)
from prompt_freezer_helpers import freeze_production_adapter_output

REVISION = "b" * 40


def _canonical_line(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _identity(path: Path) -> WebLINXSource:
    payload = path.read_bytes()
    return WebLINXSource(
        path=path,
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _row(
    demo: str,
    turn: int,
    *,
    action: str,
    history: str | None = None,
    utterances: str | None = None,
    html: str | None = "<button uid=\"safe\">Safe</button>",
) -> dict[str, object]:
    return {
        "demo": demo,
        "turn": turn,
        "action": action,
        "action_history": history,
        "utterances": utterances or f"Request for {demo}",
        "candidates": "uid=safe button",
        "clean_html": html,
        "viewport": "1280x720",
    }


def _write_chat(
    path: Path,
    rows: list[dict[str, object]],
    *,
    compressed: bool = True,
) -> WebLINXSource:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows, ensure_ascii=False).encode("utf-8")
    path.write_bytes(gzip.compress(payload, mtime=0) if compressed else payload)
    return _identity(path)


def _write_splits(path: Path, splits: dict[str, list[str]]) -> WebLINXSource:
    path.write_text(json.dumps(splits, sort_keys=True), encoding="utf-8")
    return _identity(path)


def _read_canonical_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    for raw_line in path.read_bytes().splitlines(keepends=True):
        assert raw_line.endswith(b"\n")
        row = json.loads(raw_line)
        assert set(row) == {"source_case_id", "prompt"}
        assert (
            json.dumps(
                row,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
            == raw_line
        )
        rows.append(row)
    return rows


def test_weblinx_history_parser_is_restricted_and_canonical() -> None:
    history = (
        'say(speaker="instructor", utterance="Open the page")'
        "</s><s>[INST] "
        'click(uid="node-1"); scroll(y=300, x=-10)'
    )
    assert parse_weblinx_action_history(history) == (
        'say(speaker="instructor", utterance="Open the page")',
        'click(uid="node-1")',
        "scroll(x=-10, y=300)",
    )

    with pytest.raises(ValueError, match="unsupported action"):
        parse_weblinx_action_history('__import__(name="os")')
    with pytest.raises(ValueError, match="literal str"):
        parse_weblinx_action_history('click(uid=get_secret())')
    with pytest.raises(ValueError, match="function calls"):
        parse_weblinx_action_history('os.system(command="id")')


def test_weblinx_export_is_deterministic_canonical_and_current_action_free(
    tmp_path: Path,
) -> None:
    gold_a = 'click(uid="CURRENT_GOLD_A_MUST_NOT_LEAK")'
    gold_b = 'submit(uid="CURRENT_GOLD_B_MUST_NOT_LEAK")'
    rows = [
        _row("demo-b", 3, action=gold_b, utterances="Task B"),
        _row(
            "demo-a",
            4,
            action=gold_a,
            history='click(uid="previous-node")',
            utterances="Task A",
        ),
    ]
    first_chat = _write_chat(tmp_path / "one" / "test_web.json.gz", rows)
    second_chat = _write_chat(
        tmp_path / "two" / "test_web.json.gz",
        list(reversed(rows)),
    )
    splits = _write_splits(
        tmp_path / "splits.json",
        {"test_web": ["demo-b", "demo-a"]},
    )
    first_output = tmp_path / "first.jsonl"
    second_output = tmp_path / "second.jsonl"

    audit = export_weblinx_prompt_rows(
        first_chat,
        splits,
        first_output,
        revision=REVISION,
        split="test_web",
    )
    export_weblinx_prompt_rows(
        second_chat,
        splits,
        second_output,
        revision=REVISION,
        split="test_web",
    )

    assert first_output.read_bytes() == second_output.read_bytes()
    exported = _read_canonical_rows(first_output)
    assert len(exported) == 2
    assert "Task A" in exported[0]["prompt"]
    assert "Task B" in exported[1]["prompt"]
    assert WEBLINX_PROMPT_ADAPTER_VERSION in exported[0]["prompt"]
    output_text = first_output.read_text(encoding="utf-8")
    assert "CURRENT_GOLD_A_MUST_NOT_LEAK" not in output_text
    assert "CURRENT_GOLD_B_MUST_NOT_LEAK" not in output_text
    assert 'click(uid="previous-node")' in exported[0]["prompt"]
    assert set(exported[0]) == {"source_case_id", "prompt"}
    assert audit["adapter_version"] == "weblinx-private-prompt-rows-v1"
    assert audit["label_isolation"]["current_action_emitted"] is False


def test_weblinx_export_normalizes_prefixed_demo_ids_and_keeps_history_inert(
    tmp_path: Path,
) -> None:
    malformed_as_python = (
        'say(speaker="navigator", utterance="Open the "quoted" account")'
    )
    chat = _write_chat(
        tmp_path / "test_web.json.gz",
        [
            _row(
                "annotator_created_demo-a",
                2,
                action='click(uid="current-gold")',
                history=malformed_as_python,
                utterances="Open the public documentation",
            )
        ],
    )
    splits = _write_splits(
        tmp_path / "splits.json",
        {"test_web": ["demo-a"]},
    )
    output = tmp_path / "output.jsonl"

    audit = export_weblinx_prompt_rows(
        chat,
        splits,
        output,
        revision=REVISION,
        split="test_web",
    )

    exported = _read_canonical_rows(output)
    assert len(exported) == 1
    assert malformed_as_python in exported[0]["prompt"]
    assert "annotator_created_demo-a" not in output.read_text(encoding="utf-8")
    assert audit["split_demos"] == 1


def test_weblinx_production_audit_freezes_through_generic_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed_as_python = (
        'say(speaker="navigator", utterance="Open the "quoted" account")'
    )
    chat = _write_chat(
        tmp_path / "test_web.json.gz",
        [
            _row(
                "annotator_created_demo-a",
                2,
                action='click(uid="CURRENT_GOLD_MUST_NOT_LEAK")',
                history=malformed_as_python,
                utterances="Open the public documentation",
            )
        ],
    )
    splits = _write_splits(
        tmp_path / "splits.json",
        {PRODUCTION_WEBLINX_SPLIT: ["demo-a"]},
    )
    output = tmp_path / "weblinx-production-prompts.jsonl"
    audit_path = tmp_path / "weblinx-production-audit.json"

    with pytest.raises(ValueError, match="production WebLINX chat identity mismatch"):
        export_weblinx_prompt_rows(
            chat,
            splits,
            output,
            revision=PRODUCTION_WEBLINX_REVISION,
            split=PRODUCTION_WEBLINX_SPLIT,
            audit_path=audit_path,
        )
    monkeypatch.setattr(weblinx_prompts, "PRODUCTION_WEBLINX_CHAT_BYTES", chat.bytes)
    monkeypatch.setattr(
        weblinx_prompts,
        "PRODUCTION_WEBLINX_CHAT_SHA256",
        chat.sha256,
    )
    monkeypatch.setattr(
        weblinx_prompts,
        "PRODUCTION_WEBLINX_SPLITS_BYTES",
        splits.bytes,
    )
    monkeypatch.setattr(
        weblinx_prompts,
        "PRODUCTION_WEBLINX_SPLITS_SHA256",
        splits.sha256,
    )
    monkeypatch.setattr(weblinx_prompts, "PRODUCTION_WEBLINX_SPLIT_DEMOS", 1)
    audit = export_weblinx_prompt_rows(
        chat,
        splits,
        output,
        revision=PRODUCTION_WEBLINX_REVISION,
        split=PRODUCTION_WEBLINX_SPLIT,
        audit_path=audit_path,
    )
    assert audit["mode"] == "production"
    assert "CURRENT_GOLD_MUST_NOT_LEAK" not in output.read_text(encoding="utf-8")

    frozen = freeze_production_adapter_output(
        tmp_path,
        suite_name="weblinx",
        prompt_path=output,
        audit_path=audit_path,
    )
    assert frozen["suite"]["name"] == "weblinx"
    assert frozen["sources"][0]["records"] == audit["output"]["rows"]
    assert {artifact["role"] for artifact in frozen["raw_artifacts"]} == {
        "weblinx_chat_source",
        "weblinx_splits_source",
    }

    forged_dir = tmp_path / "forged-freeze"
    forged_dir.mkdir()
    forged_output = forged_dir / output.name
    forged_rows = [
        {
            **row,
            "prompt": f"Caller-authored replacement WebLINX prompt {index}.",
        }
        for index, row in enumerate(_read_canonical_rows(output))
    ]
    forged_payload = b"".join(_canonical_line(row) for row in forged_rows)
    forged_output.write_bytes(forged_payload)
    forged_output_sha256 = hashlib.sha256(forged_payload).hexdigest()
    forged_audit_path = forged_dir / audit_path.name
    forged_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    forged_audit["output"].update(
        {
            "bytes": len(forged_payload),
            "path": forged_output.name,
            "rows": len(forged_rows),
            "sha256": forged_output_sha256,
        }
    )
    forged_audit["freeze_binding"]["output"].update(
        {
            "bytes": len(forged_payload),
            "records": len(forged_rows),
            "sha256": forged_output_sha256,
        }
    )
    forged_audit.pop("audit_self_sha256")
    forged_audit["audit_self_sha256"] = hashlib.sha256(
        _canonical_line(forged_audit)[:-1]
    ).hexdigest()
    forged_audit_path.write_bytes(_canonical_line(forged_audit))
    with pytest.raises(ValueError, match="raw-source reexport"):
        freeze_production_adapter_output(
            forged_dir,
            suite_name="weblinx",
            prompt_path=forged_output,
            audit_path=forged_audit_path,
            plan_attestation_path=audit_path,
        )
    assert not (forged_dir / "weblinx-frozen-prompts.jsonl").exists()
    assert not (
        forged_dir / "weblinx-frozen-prompts.provenance.json"
    ).exists()

    tampered_dir = tmp_path / "tampered-freeze"
    tampered_dir.mkdir()
    tampered_output = tampered_dir / output.name
    tampered_output.write_bytes(output.read_bytes())
    tampered_audit_path = tampered_dir / audit_path.name
    tampered_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    tampered_audit["label_isolation"]["current_action_emitted"] = True
    without_hash = dict(tampered_audit)
    without_hash.pop("audit_self_sha256")
    tampered_audit["audit_self_sha256"] = hashlib.sha256(
        _canonical_line(without_hash)[:-1]
    ).hexdigest()
    tampered_audit_path.write_bytes(_canonical_line(tampered_audit))
    with pytest.raises(ValueError, match="label_isolation must contain only false"):
        freeze_production_adapter_output(
            tampered_dir,
            suite_name="weblinx",
            prompt_path=tampered_output,
            audit_path=tampered_audit_path,
            plan_attestation_path=audit_path,
        )


def test_weblinx_whole_demo_privacy_exclusion_is_audited(tmp_path: Path) -> None:
    sensitive_demo = "private-demo"
    rows = [
        _row(
            sensitive_demo,
            2,
            action='say(speaker="navigator", utterance="ok")',
            utterances=(
                "Log in as user@example.com with Password: SuperSecret123!"
            ),
        ),
        _row(
            sensitive_demo,
            3,
            action='click(uid="safe")',
            utterances="This otherwise-clean later step must also be removed",
        ),
        _row(
            "safe-demo",
            2,
            action='click(uid="CURRENT_GOLD")',
            utterances="Open the public documentation",
        ),
    ]
    chat = _write_chat(tmp_path / "test_web.json.gz", rows)
    splits = _write_splits(
        tmp_path / "splits.json",
        {"test_web": [sensitive_demo, "safe-demo"]},
    )
    output = tmp_path / "output.jsonl"
    audit_path = tmp_path / "audit.json"

    audit = export_weblinx_prompt_rows(
        chat,
        splits,
        output,
        revision=REVISION,
        split="test_web",
        audit_path=audit_path,
    )

    exported = _read_canonical_rows(output)
    assert len(exported) == 1
    assert "public documentation" in exported[0]["prompt"]
    assert sensitive_demo not in output.read_text(encoding="utf-8")
    privacy = audit["privacy"]
    assert privacy["excluded_demos"] == 1
    assert privacy["excluded_rows"] == 2
    assert privacy["reason_counts"] == {"email": 1, "labeled_secret": 1}
    assert len(privacy["excluded_demo_id_sha256"]) == 1
    assert sensitive_demo not in json.dumps(audit)
    assert json.loads(audit_path.read_text(encoding="utf-8")) == audit
    assert detect_weblinx_privacy_reasons(
        {"text": "email me at person@example.org"}
    ) == frozenset({"email"})


def test_weblinx_export_rejects_schema_drift_and_split_drift(tmp_path: Path) -> None:
    row = _row("demo-a", 2, action='click(uid="gold")')
    row["expected_calls"] = []
    drifted_chat = _write_chat(tmp_path / "drifted.json.gz", [row])
    splits = _write_splits(tmp_path / "splits.json", {"test_web": ["demo-a"]})

    with pytest.raises(ValueError, match="schema drift"):
        export_weblinx_prompt_rows(
            drifted_chat,
            splits,
            tmp_path / "drifted-output.jsonl",
            revision=REVISION,
            split="test_web",
        )

    clean_chat = _write_chat(
        tmp_path / "clean.json.gz",
        [_row("demo-a", 2, action='click(uid="gold")')],
    )
    incomplete_splits = _write_splits(
        tmp_path / "incomplete-splits.json",
        {"test_web": ["demo-a", "missing-demo"]},
    )
    with pytest.raises(ValueError, match="does not exactly cover"):
        export_weblinx_prompt_rows(
            clean_chat,
            incomplete_splits,
            tmp_path / "split-output.jsonl",
            revision=REVISION,
            split="test_web",
        )


def test_weblinx_export_fails_closed_on_decompression_and_prompt_caps(
    tmp_path: Path,
) -> None:
    chat = _write_chat(
        tmp_path / "test_web.json.gz",
        [
            _row(
                "demo-a",
                2,
                action='click(uid="gold")',
                html="<p>" + "x" * 500 + "</p>",
            )
        ],
    )
    splits = _write_splits(tmp_path / "splits.json", {"test_web": ["demo-a"]})

    with pytest.raises(ValueError, match="max_decompressed_bytes"):
        export_weblinx_prompt_rows(
            chat,
            splits,
            tmp_path / "decompression-cap.jsonl",
            revision=REVISION,
            split="test_web",
            max_decompressed_bytes=32,
        )
    assert not (tmp_path / "decompression-cap.jsonl").exists()

    with pytest.raises(ValueError, match="max_prompt_bytes"):
        export_weblinx_prompt_rows(
            chat,
            splits,
            tmp_path / "prompt-cap.jsonl",
            revision=REVISION,
            split="test_web",
            max_prompt_bytes=128,
        )
    assert not (tmp_path / "prompt-cap.jsonl").exists()


def test_weblinx_export_parses_verified_snapshot_after_same_stat_source_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = _write_chat(
        tmp_path / "test_web.json.gz",
        [_row("demo-a", 2, action='click(uid="gold")')],
    )
    splits = _write_splits(
        tmp_path / "splits.json",
        {"test_web": ["demo-a"]},
    )
    splits_stat = splits.path.stat()
    mutated_payload = json.dumps(
        {"test_web": ["demo-z"]},
        sort_keys=True,
    ).encode("utf-8")
    assert len(mutated_payload) == splits.bytes
    snapshot_source = weblinx_prompts._snapshot_verified_source

    def race_after_snapshot(
        source_to_copy: WebLINXSource,
        snapshot_path: Path,
        *,
        max_source_bytes: int,
        label: str,
    ) -> tuple[Path, dict[str, int | str]]:
        verified = snapshot_source(
            source_to_copy,
            snapshot_path,
            max_source_bytes=max_source_bytes,
            label=label,
        )
        if label == "splits source":
            source_to_copy.path.write_bytes(mutated_payload)
            os.utime(
                source_to_copy.path,
                ns=(source_to_copy.path.stat().st_atime_ns, splits_stat.st_mtime_ns),
            )
        return verified

    monkeypatch.setattr(
        weblinx_prompts,
        "_snapshot_verified_source",
        race_after_snapshot,
    )
    output = tmp_path / "snapshot-output.jsonl"
    audit = export_weblinx_prompt_rows(
        chat,
        splits,
        output,
        revision=REVISION,
        split="test_web",
    )

    raced_stat = splits.path.stat()
    assert (raced_stat.st_ino, raced_stat.st_size, raced_stat.st_mtime_ns) == (
        splits_stat.st_ino,
        splits_stat.st_size,
        splits_stat.st_mtime_ns,
    )
    assert len(_read_canonical_rows(output)) == 1
    assert audit["sources"]["splits"]["sha256"] == splits.sha256


def test_weblinx_export_rejects_raced_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = _write_chat(
        tmp_path / "test_web.json.gz",
        [_row("demo-a", 2, action='click(uid="gold")')],
    )
    splits = _write_splits(
        tmp_path / "splits.json",
        {"test_web": ["demo-a"]},
    )
    output = tmp_path / "output.jsonl"

    def race_destination(_source: str | Path, destination: str | Path) -> None:
        Path(destination).write_text("raced bytes\n", encoding="utf-8")
        raise FileExistsError

    monkeypatch.setattr(weblinx_prompts.os, "link", race_destination)
    with pytest.raises(RuntimeError, match="concurrently published.*does not match"):
        export_weblinx_prompt_rows(
            chat,
            splits,
            output,
            revision=REVISION,
            split="test_web",
        )
    assert output.read_text(encoding="utf-8") == "raced bytes\n"
