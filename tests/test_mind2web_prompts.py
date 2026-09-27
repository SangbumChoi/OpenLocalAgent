from __future__ import annotations

import hashlib
import json
import os
import warnings
import zipfile
from pathlib import Path

import pytest

import localagent.data.mind2web_prompts as mind2web_prompts
from localagent.data.mind2web_dom_ranker import (
    MIND2WEB_DOM_RANKER_VERSION,
    MIND2WEB_RANKED_PROMPT_ADAPTER_VERSION,
    load_mind2web_dom_ranker_config,
    rank_mind2web_dom,
)
from localagent.data.mind2web_prompts import (
    MIND2WEB_PROMPT_ADAPTER_VERSION,
    PRODUCTION_MIND2WEB_MEMBERS,
    PRODUCTION_MIND2WEB_REVISION,
    PRODUCTION_MIND2WEB_SPLIT,
    PRODUCTION_MIND2WEB_TASK_COUNTS,
    Mind2WebArchive,
    Mind2WebRankerInput,
    Mind2WebSource,
    _render_ranked_mind2web_step_prompt,
    export_mind2web_prompt_rows,
    project_mind2web_ranker_input,
)
from prompt_freezer_helpers import freeze_production_adapter_output

REVISION = "a" * 40
RANKER_CONFIG_PATH = (
    Path(__file__).parents[1] / "configs/data/mind2web-dom-lexical-v1.json"
)
RANKER_CONFIG_SHA256 = (
    "cf9c6e75c465827a97121601b937e77ce992cda25316620dbae73b32b90a3f46"
)


def _identity(path: Path, *, archive_member: str | None = None) -> Mind2WebSource:
    payload = path.read_bytes()
    return Mind2WebSource(
        path=path,
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        archive_member=archive_member,
    )


def _action(uid: str, *, context: str, gold: str) -> dict[str, object]:
    return {
        "action_uid": uid,
        "raw_html": f"<raw>{gold}</raw>",
        "cleaned_html": context,
        "operation": {
            "op": "CLICK",
            "original_op": "CLICK",
            "value": gold,
        },
        "pos_candidates": [{"backend_node_id": gold}],
        "neg_candidates": [{"backend_node_id": f"negative-{gold}"}],
    }


def _task(
    annotation_id: str,
    *,
    task: str,
    actions: list[dict[str, object]],
    gold: str,
    action_reprs: list[str] | None = None,
) -> dict[str, object]:
    return {
        "website": "example",
        "domain": "example.com",
        "subdomain": "www",
        "annotation_id": annotation_id,
        "confirmed_task": task,
        "action_reprs": (
            action_reprs if action_reprs is not None else [gold for _ in actions]
        ),
        "actions": actions,
    }


def _write_source(path: Path, tasks: list[dict[str, object]]) -> Mind2WebSource:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
    return _identity(path)


def test_large_json_record_decode_retries_grow_exponentially(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload_value = "x" * (2 * 1024 * 1024)
    payload = json.dumps([{"payload": payload_value}], separators=(",", ":")).encode()
    source = tmp_path / "large-record.json"
    source.write_bytes(payload)
    original_decoder = mind2web_prompts.json.JSONDecoder
    attempts: list[int] = []

    class RecordingDecoder:
        def __init__(self, *args, **kwargs) -> None:
            self._inner = original_decoder(*args, **kwargs)

        def raw_decode(self, value: str, index: int = 0):
            attempts.append(len(value) - index)
            return self._inner.raw_decode(value, index)

    monkeypatch.setattr(mind2web_prompts.json, "JSONDecoder", RecordingDecoder)
    rows = list(
        mind2web_prompts._iter_json_array(
            source,
            max_bytes=len(payload),
            max_record_chars=mind2web_prompts.DEFAULT_MAX_RECORD_CHARS,
        )
    )

    assert rows == [{"payload": payload_value}]
    assert len(attempts) <= 8
    assert attempts == sorted(attempts)
    assert mind2web_prompts.DEFAULT_MAX_RECORD_CHARS == 1024 * 1024 * 1024


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


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _write_self_hashed_audit(path: Path, audit: dict[str, object]) -> None:
    without_hash = dict(audit)
    without_hash.pop("audit_self_sha256", None)
    with_hash = {
        **without_hash,
        "audit_self_sha256": hashlib.sha256(
            _canonical_json_bytes(without_hash)
        ).hexdigest(),
    }
    path.write_bytes(_canonical_json_bytes(with_hash) + b"\n")


_TEST_PRODUCTION_MEMBERS = {
    "cross_domain": ("test_domain/test_domain_0.json",),
    "cross_task": ("test_task/test_task_0.json",),
    "cross_website": ("test_website/test_website_0.json",),
}
_TEST_PRODUCTION_TASK_COUNTS = {
    "cross_domain": 1,
    "cross_task": 1,
    "cross_website": 1,
}


def _production_payloads(*, domain_tasks: int = 1) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for split_name, members in _TEST_PRODUCTION_MEMBERS.items():
        tasks = [
            _task(
                f"{split_name}-task-{index}",
                task=f"Open the visible {split_name} account page {index}",
                actions=[
                    _action(
                        f"{split_name}-step-{index}",
                        context=f"<button>Visible {split_name} {index}</button>",
                        gold=f"CURRENT_GOLD_{split_name}_{index}_MUST_NOT_LEAK",
                    )
                ],
                gold=f"CURRENT_GOLD_{split_name}_{index}_MUST_NOT_LEAK",
            )
            for index in range(domain_tasks if split_name == "cross_domain" else 1)
        ]
        payloads[members[0]] = json.dumps(tasks, ensure_ascii=False).encode("utf-8")
    return payloads


def _write_member_sources(
    directory: Path,
    payloads: dict[str, bytes],
) -> list[Mind2WebSource]:
    sources = []
    for member in sorted(_TEST_PRODUCTION_MEMBERS.values()):
        member_name = member[0]
        path = directory / member_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[member_name])
        sources.append(_identity(path, archive_member=member_name))
    return sources


def _write_archive(
    path: Path,
    members: list[tuple[str, bytes]],
) -> Mind2WebArchive:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for member, payload in members:
                archive.writestr(member, payload)
    payload = path.read_bytes()
    return Mind2WebArchive(
        path=path,
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _patch_production_pins(
    monkeypatch: pytest.MonkeyPatch,
    archive: Mind2WebArchive,
    *,
    total_uncompressed_bytes: int,
    encrypted: bool = False,
    task_counts: dict[str, int] | None = None,
) -> None:
    monkeypatch.setattr(
        mind2web_prompts,
        "PRODUCTION_MIND2WEB_ARCHIVE_BYTES",
        archive.bytes,
    )
    monkeypatch.setattr(
        mind2web_prompts,
        "PRODUCTION_MIND2WEB_ARCHIVE_SHA256",
        archive.sha256,
    )
    monkeypatch.setattr(
        mind2web_prompts,
        "PRODUCTION_MIND2WEB_ARCHIVE_ENCRYPTED",
        encrypted,
    )
    monkeypatch.setattr(
        mind2web_prompts,
        "PRODUCTION_MIND2WEB_TOTAL_UNCOMPRESSED_BYTES",
        total_uncompressed_bytes,
    )
    monkeypatch.setattr(
        mind2web_prompts,
        "PRODUCTION_MIND2WEB_MEMBERS",
        _TEST_PRODUCTION_MEMBERS,
    )
    monkeypatch.setattr(
        mind2web_prompts,
        "PRODUCTION_MIND2WEB_TASK_COUNTS",
        task_counts or _TEST_PRODUCTION_TASK_COUNTS,
    )


def _production_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    domain_tasks: int = 1,
) -> tuple[list[Mind2WebSource], Mind2WebArchive]:
    payloads = _production_payloads(domain_tasks=domain_tasks)
    sources = _write_member_sources(tmp_path / "extracted", payloads)
    archive = _write_archive(
        tmp_path / "test.zip",
        sorted(payloads.items()),
    )
    _patch_production_pins(
        monkeypatch,
        archive,
        total_uncompressed_bytes=sum(len(payload) for payload in payloads.values()),
    )
    return sources, archive


def test_mind2web_export_is_deterministic_canonical_and_label_free(
    tmp_path: Path,
) -> None:
    gold_a = "CURRENT_GOLD_A_MUST_NOT_LEAK"
    gold_b = "CURRENT_GOLD_B_MUST_NOT_LEAK"
    tasks = [
        _task(
            "task-b",
            task="Task B",
            actions=[_action("b-0", context="<button>Visible B</button>", gold=gold_b)],
            gold=gold_b,
        ),
        _task(
            "task-a",
            task="Task A",
            actions=[_action("a-0", context="<button>Visible A</button>", gold=gold_a)],
            gold=gold_a,
        ),
    ]
    first_source = _write_source(tmp_path / "one" / "train_0.json", tasks)
    second_source = _write_source(
        tmp_path / "two" / "train_0.json",
        list(reversed(tasks)),
    )
    first_output = tmp_path / "first.jsonl"
    second_output = tmp_path / "second.jsonl"
    first_audit_path = tmp_path / "first.audit.json"

    audit = export_mind2web_prompt_rows(
        [first_source],
        first_output,
        revision=REVISION,
        audit_path=first_audit_path,
    )
    export_mind2web_prompt_rows(
        [second_source],
        second_output,
        revision=REVISION,
    )

    assert first_output.read_bytes() == second_output.read_bytes()
    rows = _read_canonical_rows(first_output)
    assert len(rows) == 2
    assert "Task A" in rows[0]["prompt"]
    assert "Task B" in rows[1]["prompt"]
    assert MIND2WEB_PROMPT_ADAPTER_VERSION in rows[0]["prompt"]
    output_text = first_output.read_text(encoding="utf-8")
    assert gold_a not in output_text
    assert gold_b not in output_text
    assert "pos_candidates" not in output_text
    assert "neg_candidates" not in output_text
    assert "operation" not in output_text
    assert audit["adapter_version"] == "mind2web-private-prompt-rows-v1"
    assert audit["output"]["rows"] == 2
    assert audit["label_isolation"]["current_action_emitted"] is False
    assert json.loads(first_audit_path.read_text(encoding="utf-8")) == audit


def test_mind2web_production_v2_audit_binds_ranker_and_framed_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources, archive = _production_fixture(tmp_path, monkeypatch)
    output = tmp_path / "mind2web-production-prompts.jsonl"
    audit_path = tmp_path / "mind2web-production-audit.json"

    audit = export_mind2web_prompt_rows(
        sources,
        output,
        revision=PRODUCTION_MIND2WEB_REVISION,
        split=PRODUCTION_MIND2WEB_SPLIT,
        archive=archive,
        audit_path=audit_path,
        ranker_config_path=RANKER_CONFIG_PATH,
    )
    assert audit["mode"] == "production"
    assert audit["adapter_version"] == MIND2WEB_RANKED_PROMPT_ADAPTER_VERSION
    assert audit["schema_version"] == 3
    assert "CURRENT_GOLD_" not in output.read_text(encoding="utf-8")
    assert "<prior_actions>" not in output.read_text(encoding="utf-8")
    rows = _read_canonical_rows(output)
    assert all(
        len(("<|user|>" + row["prompt"] + "<|assistant|>").encode("utf-8"))
        <= 1792
        for row in rows
    )
    attestation = audit["source_attestation"]
    assert attestation["archive"]["sha256"] == archive.sha256
    assert attestation["archive_format"] == {
        "compression": "deflate",
        "encryption": "fixture_unencrypted",
        "members": 3,
    }
    assert attestation["tasks_by_split"] == _TEST_PRODUCTION_TASK_COUNTS
    assert len(attestation["members"]) == 3
    assert len(attestation["members_sha256"]) == 64
    ranking = audit["ranking"]
    assert ranking["ranker"]["ranker_version"] == MIND2WEB_DOM_RANKER_VERSION
    assert ranking["adapter_implementation"] == audit["freeze_binding"][
        "adapter_implementation"
    ]
    assert ranking["adapter_implementation"]["module"] == (
        "localagent.data.mind2web_prompts"
    )
    assert ranking["adapter_implementation"]["path"] == (
        "src/localagent/data/mind2web_prompts.py"
    )
    assert ranking["adapter_implementation"]["sha256"] == hashlib.sha256(
        Path(mind2web_prompts.__file__).read_bytes()
    ).hexdigest()
    assert ranking["totals"]["rows"] == audit["output"]["rows"] == 3
    assert ranking["totals"]["max_framed_prompt_bytes"] <= 1792
    assert ranking["dependencies"] == {
        "action_representations_used_by_ranker": False,
        "action_uids_used_by_ranker": False,
        "model_used": False,
        "negative_candidates_used_by_ranker": False,
        "operations_used_by_ranker": False,
        "positive_candidates_used_by_ranker": False,
        "raw_html_used_by_ranker": False,
        "tokenizer_used": False,
    }
    assert audit["label_isolation"]["prior_action_representations_emitted"] is False


def test_mind2web_production_rejects_adapter_implementation_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources, archive = _production_fixture(tmp_path, monkeypatch)
    output = tmp_path / "raced-adapter.jsonl"
    observed_identity = mind2web_prompts._adapter_implementation_identity()
    calls = 0

    def raced_identity() -> dict[str, int | str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return observed_identity
        return {**observed_identity, "sha256": "0" * 64}

    monkeypatch.setattr(
        mind2web_prompts,
        "_adapter_implementation_identity",
        raced_identity,
    )

    with pytest.raises(ValueError, match="implementation.*changed during export"):
        export_mind2web_prompt_rows(
            sources,
            output,
            revision=PRODUCTION_MIND2WEB_REVISION,
            split=PRODUCTION_MIND2WEB_SPLIT,
            archive=archive,
            ranker_config_path=RANKER_CONFIG_PATH,
        )
    assert calls == 2
    assert not output.exists()


def test_mind2web_ranker_config_is_canonical_self_hashed_and_code_bound() -> None:
    config = load_mind2web_dom_ranker_config(RANKER_CONFIG_PATH)

    assert config.sha256 == RANKER_CONFIG_SHA256
    assert config.config_self_sha256 == (
        "ea258f0eee464f69f18baa97e0631a259c06051d2a9b2ae8454ce34b3244b8f3"
    )
    assert config.max_framed_prompt_bytes == 1792
    assert config.max_unframed_prompt_bytes == 1771
    assert config.generation_reserve_tokens_including_eos == 256
    assert config.audit_identity()["ranker_version"] == MIND2WEB_DOM_RANKER_VERSION


def test_mind2web_ranker_rejects_config_self_hash_and_code_drift(
    tmp_path: Path,
) -> None:
    original = json.loads(RANKER_CONFIG_PATH.read_text(encoding="utf-8"))
    raw = json.loads(json.dumps(original))
    raw["budget"]["minimum_dom_bytes"] -= 1
    drifted = tmp_path / "drifted.json"
    drifted.write_bytes(_canonical_json_bytes(raw) + b"\n")
    with pytest.raises(ValueError, match="self-hash mismatch"):
        load_mind2web_dom_ranker_config(drifted)

    raw = json.loads(json.dumps(original))
    raw["budget"]["user_marker"] = "u"
    raw["budget"]["assistant_marker"] = "a"
    raw["budget"]["max_unframed_prompt_bytes"] = 1790
    without_hash = dict(raw)
    without_hash.pop("config_self_sha256")
    raw["config_self_sha256"] = hashlib.sha256(
        _canonical_json_bytes(without_hash)
    ).hexdigest()
    drifted.write_bytes(_canonical_json_bytes(raw) + b"\n")
    with pytest.raises(ValueError, match="framing/model/generation budget"):
        load_mind2web_dom_ranker_config(drifted)

    raw = json.loads(json.dumps(original))
    raw["implementation"]["sha256"] = "f" * 64
    without_hash = dict(raw)
    without_hash.pop("config_self_sha256")
    raw["config_self_sha256"] = hashlib.sha256(
        _canonical_json_bytes(without_hash)
    ).hexdigest()
    drifted.write_bytes(_canonical_json_bytes(raw) + b"\n")
    with pytest.raises(ValueError, match="implementation SHA-256"):
        load_mind2web_dom_ranker_config(drifted)


def test_mind2web_ranker_is_deterministic_and_uses_stable_preorder_ties() -> None:
    config = load_mind2web_dom_ranker_config(RANKER_CONFIG_PATH)
    buttons = [
        (
            "<button "
            f'id="button-{index}" title="{"save account" if index == 19 else "other control"}" '
            f'aria-label="control {index}">Control {index}</button>'
        )
        for index in range(20)
    ]
    first_html = (
        '<div id="root" role="main">'
        + "".join(buttons)
        + "<script>PRIVATE_SCRIPT_CANARY save account</script></div>"
    )
    second_html = first_html.replace(
        'id="root" role="main"',
        'role="main" id="root"',
    ).replace(
        'id="button-19" title="save account" aria-label="control 19"',
        'aria-label="control 19" title="save account" id="button-19"',
    )

    first = rank_mind2web_dom(
        "save account",
        first_html,
        context_byte_budget=768,
        config=config,
    )
    second = rank_mind2web_dom(
        "save account",
        second_html,
        context_byte_budget=768,
        config=config,
    )

    assert first.context == second.context
    assert first.ranked_dom_sha256 == second.ranked_dom_sha256
    assert first.selected_bytes == second.selected_bytes
    assert first.selected_nodes == second.selected_nodes
    assert first.selected_nodes < first.parsed_nodes
    assert '"button-19"' in first.context
    assert '"button-0"' in first.context
    assert "PRIVATE_SCRIPT_CANARY" not in first.context
    ordinals = [json.loads(line.removeprefix("node "))[0] for line in first.context.splitlines()]
    assert ordinals == sorted(ordinals)


class _PoisonValue:
    def __getattribute__(self, name: str) -> object:
        if name.startswith("__"):
            return object.__getattribute__(self, name)
        raise AssertionError(f"forbidden value was accessed via {name}")

    def __bool__(self) -> bool:
        raise AssertionError("forbidden value truthiness was inspected")

    def __iter__(self) -> object:
        raise AssertionError("forbidden value was iterated")

    def __str__(self) -> str:
        raise AssertionError("forbidden value was stringified")


def test_mind2web_ranker_projection_never_dereferences_gold_fields() -> None:
    poison = _PoisonValue()
    task = {
        "website": poison,
        "domain": poison,
        "subdomain": poison,
        "annotation_id": poison,
        "confirmed_task": "Open account settings",
        "action_reprs": poison,
        "actions": [
            {
                "action_uid": poison,
                "raw_html": poison,
                "cleaned_html": '<button backend_node_id="42">Account settings</button>',
                "operation": poison,
                "pos_candidates": poison,
                "neg_candidates": poison,
            }
        ],
    }

    projected = project_mind2web_ranker_input(task, 0)

    assert projected == Mind2WebRankerInput(
        confirmed_task="Open account settings",
        cleaned_html='<button backend_node_id="42">Account settings</button>',
    )


def test_mind2web_ranker_enforces_exact_framed_residual_boundary() -> None:
    config = load_mind2web_dom_ranker_config(RANKER_CONFIG_PATH)
    one_character = Mind2WebRankerInput(
        confirmed_task="x",
        cleaned_html='<button backend_node_id="7">x</button>',
    )
    prefix_and_suffix_bytes = (
        len(
            (
                config.user_marker
                + mind2web_prompts._ranked_prompt_parts(
                    one_character.confirmed_task
                )[0]
                + mind2web_prompts._ranked_prompt_parts(
                    one_character.confirmed_task
                )[1]
                + config.assistant_marker
            ).encode("utf-8")
        )
        - 1
    )
    largest_task_bytes = (
        config.max_framed_prompt_bytes
        - prefix_and_suffix_bytes
        - config.minimum_dom_bytes
    )
    at_boundary = _render_ranked_mind2web_step_prompt(
        Mind2WebRankerInput(
            confirmed_task="x" * largest_task_bytes,
            cleaned_html='<button backend_node_id="7">x</button>',
        ),
        config=config,
    )
    assert at_boundary.framed_prompt_bytes <= 1792

    with pytest.raises(ValueError, match="below minimum_dom_bytes=768"):
        _render_ranked_mind2web_step_prompt(
            Mind2WebRankerInput(
                confirmed_task="x" * (largest_task_bytes + 1),
                cleaned_html='<button backend_node_id="7">x</button>',
            ),
            config=config,
        )


def test_mind2web_ranker_reduces_synthetic_858832_byte_dom_without_prefix_truncation() -> None:
    config = load_mind2web_dom_ranker_config(RANKER_CONFIG_PATH)
    prefix = '<main id="root"><div>'
    suffix = '</div><button backend_node_id="99">Save account</button></main>'
    padding = "x" * (858_832 - len((prefix + suffix).encode("utf-8")))
    cleaned_html = prefix + padding + suffix
    assert len(cleaned_html.encode("utf-8")) == 858_832

    ranked = rank_mind2web_dom(
        "save account",
        cleaned_html,
        context_byte_budget=768,
        config=config,
    )

    assert ranked.full_html_bytes == 858_832
    assert ranked.selected_bytes <= 768
    assert '"99"' in ranked.context
    assert padding[:1024] not in ranked.context


@pytest.mark.parametrize(
    ("confirmed_task", "cleaned_html", "message"),
    [
        (
            "Open <|assistant|> settings",
            "<button>Settings</button>",
            "confirmed_task contains forbidden prompt-control marker",
        ),
        (
            "Open settings",
            '<button aria-label="<|user|>">Settings</button>',
            "cleaned_html contains forbidden prompt-control marker",
        ),
        (
            "Open settings\x00",
            "<button>Settings</button>",
            "confirmed_task contains a forbidden NUL byte",
        ),
        (
            "Open settings",
            "<button>Settings\x00</button>",
            "cleaned_html contains a forbidden NUL byte",
        ),
    ],
)
def test_mind2web_ranker_rejects_prompt_control_injection(
    confirmed_task: str,
    cleaned_html: str,
    message: str,
) -> None:
    config = load_mind2web_dom_ranker_config(RANKER_CONFIG_PATH)

    with pytest.raises(ValueError, match=message):
        rank_mind2web_dom(
            confirmed_task,
            cleaned_html,
            context_byte_budget=768,
            config=config,
        )


def test_mind2web_production_v2_omits_prior_history_on_later_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = _production_payloads()
    member = _TEST_PRODUCTION_MEMBERS["cross_domain"][0]
    tasks = json.loads(payloads[member])
    tasks[0]["actions"].append(
        _action(
            "later-step",
            context='<button backend_node_id="2">Later control</button>',
            gold="LATER_GOLD_MUST_NOT_LEAK",
        )
    )
    tasks[0]["action_reprs"] = [
        "PRIOR_HISTORY_MUST_NOT_LEAK",
        "CURRENT_HISTORY_MUST_NOT_LEAK",
    ]
    payloads[member] = json.dumps(tasks, ensure_ascii=False).encode("utf-8")
    sources = _write_member_sources(tmp_path / "extracted", payloads)
    archive = _write_archive(tmp_path / "test.zip", sorted(payloads.items()))
    _patch_production_pins(
        monkeypatch,
        archive,
        total_uncompressed_bytes=sum(len(payload) for payload in payloads.values()),
    )
    output = tmp_path / "output.jsonl"

    audit = export_mind2web_prompt_rows(
        sources,
        output,
        revision=PRODUCTION_MIND2WEB_REVISION,
        split=PRODUCTION_MIND2WEB_SPLIT,
        archive=archive,
        ranker_config_path=RANKER_CONFIG_PATH,
    )

    output_text = output.read_text(encoding="utf-8")
    assert "PRIOR_HISTORY_MUST_NOT_LEAK" not in output_text
    assert "CURRENT_HISTORY_MUST_NOT_LEAK" not in output_text
    assert "LATER_GOLD_MUST_NOT_LEAK" not in output_text
    assert audit["output"]["rows"] == 4
    assert audit["label_isolation"]["prior_action_representations_emitted"] is False


def test_mind2web_production_v2_is_invariant_to_forbidden_field_values_and_source_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_payloads = _production_payloads()
    original_sources = _write_member_sources(tmp_path / "original/extracted", original_payloads)
    original_archive = _write_archive(
        tmp_path / "original/test.zip",
        sorted(original_payloads.items()),
    )
    _patch_production_pins(
        monkeypatch,
        original_archive,
        total_uncompressed_bytes=sum(
            len(payload) for payload in original_payloads.values()
        ),
    )
    original_output = tmp_path / "original/output.jsonl"
    original_audit = export_mind2web_prompt_rows(
        list(reversed(original_sources)),
        original_output,
        revision=PRODUCTION_MIND2WEB_REVISION,
        split=PRODUCTION_MIND2WEB_SPLIT,
        archive=original_archive,
        ranker_config_path=RANKER_CONFIG_PATH,
    )

    mutated_payloads: dict[str, bytes] = {}
    for member, payload in original_payloads.items():
        tasks = json.loads(payload)
        for task in tasks:
            task["website"] = {"ignored": True}
            task["domain"] = None
            task["subdomain"] = 123
            task["annotation_id"] = ["ignored"]
            task["action_reprs"] = {"forbidden": "changed"}
            for action in task["actions"]:
                action["action_uid"] = {"forbidden": "changed"}
                action["raw_html"] = ["forbidden", "changed"]
                action["operation"] = "forbidden-changed"
                action["pos_candidates"] = {"forbidden": "changed"}
                action["neg_candidates"] = None
        mutated_payloads[member] = json.dumps(
            tasks,
            ensure_ascii=False,
        ).encode("utf-8")
    mutated_sources = _write_member_sources(
        tmp_path / "mutated/extracted",
        mutated_payloads,
    )
    mutated_archive = _write_archive(
        tmp_path / "mutated/test.zip",
        sorted(mutated_payloads.items()),
    )
    _patch_production_pins(
        monkeypatch,
        mutated_archive,
        total_uncompressed_bytes=sum(
            len(payload) for payload in mutated_payloads.values()
        ),
    )
    mutated_output = tmp_path / "mutated/output.jsonl"
    mutated_audit = export_mind2web_prompt_rows(
        mutated_sources,
        mutated_output,
        revision=PRODUCTION_MIND2WEB_REVISION,
        split=PRODUCTION_MIND2WEB_SPLIT,
        archive=mutated_archive,
        ranker_config_path=RANKER_CONFIG_PATH,
    )

    assert mutated_output.read_bytes() == original_output.read_bytes()
    assert (
        mutated_audit["ranking"]["ordered_row_receipts_sha256"]
        == original_audit["ranking"]["ordered_row_receipts_sha256"]
    )


def test_mind2web_production_requires_ranker_config_and_fixture_rejects_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources, archive = _production_fixture(tmp_path / "production", monkeypatch)
    with pytest.raises(ValueError, match="requires an explicit ranker_config_path"):
        export_mind2web_prompt_rows(
            sources,
            tmp_path / "missing-config.jsonl",
            revision=PRODUCTION_MIND2WEB_REVISION,
            split=PRODUCTION_MIND2WEB_SPLIT,
            archive=archive,
        )

    fixture_source = _write_source(
        tmp_path / "fixture.json",
        [
            _task(
                "fixture",
                task="Use fixture",
                actions=[_action("step", context="<button>Fixture</button>", gold="gold")],
                gold="gold",
            )
        ],
    )
    with pytest.raises(ValueError, match="fixture.*must not declare a ranker config"):
        export_mind2web_prompt_rows(
            [fixture_source],
            tmp_path / "fixture-output.jsonl",
            revision=REVISION,
            ranker_config_path=RANKER_CONFIG_PATH,
        )


def test_generic_freezer_rejects_legacy_mind2web_audit_without_archive_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources, archive = _production_fixture(tmp_path, monkeypatch)
    output = tmp_path / "mind2web-production-prompts.jsonl"
    audit_path = tmp_path / "mind2web-production-audit.json"
    audit = export_mind2web_prompt_rows(
        sources,
        output,
        revision=PRODUCTION_MIND2WEB_REVISION,
        split=PRODUCTION_MIND2WEB_SPLIT,
        archive=archive,
        audit_path=audit_path,
        ranker_config_path=RANKER_CONFIG_PATH,
    )
    legacy_audit = dict(audit)
    legacy_audit.pop("source_attestation")
    legacy_audit["schema_version"] = 1
    _write_self_hashed_audit(audit_path, legacy_audit)

    with pytest.raises(ValueError, match="Mind2Web v2 archive-bound audit"):
        freeze_production_adapter_output(
            tmp_path,
            suite_name="mind2web",
            prompt_path=output,
            audit_path=audit_path,
        )


@pytest.mark.parametrize(
    ("drift", "error"),
    [
        ("member_name", "member layout disagrees with benchmark plan"),
        ("member_sha256", "sources disagree with protected archive"),
        ("member_rows", "member rows and bound prompt output disagree"),
        ("audit_tasks", "total_tasks mismatch"),
        ("missing_self_hash", "audit_self_sha256 is required"),
    ],
)
def test_generic_freezer_rejects_fabricated_mind2web_v2_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
    error: str,
) -> None:
    sources, archive = _production_fixture(tmp_path, monkeypatch)
    output = tmp_path / "mind2web-production-prompts.jsonl"
    audit_path = tmp_path / "mind2web-production-audit.json"
    export_mind2web_prompt_rows(
        sources,
        output,
        revision=PRODUCTION_MIND2WEB_REVISION,
        split=PRODUCTION_MIND2WEB_SPLIT,
        archive=archive,
        audit_path=audit_path,
        ranker_config_path=RANKER_CONFIG_PATH,
    )
    plan_attestation_path = tmp_path / "original-production-audit.json"
    plan_attestation_path.write_bytes(audit_path.read_bytes())
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    attestation = audit["source_attestation"]
    members = attestation["members"]

    if drift == "member_name":
        members[0]["member"] = "test_domain/renamed.json"
        audit["sources"][0]["archive_member"] = "test_domain/renamed.json"
    elif drift == "member_sha256":
        members[0]["sha256"] = "b" * 64
    elif drift == "member_rows":
        members[0]["rows"] += 1
        audit["sources"][0]["rows"] += 1
    elif drift == "audit_tasks":
        audit["tasks"] += 1
    elif drift != "missing_self_hash":
        raise AssertionError(f"unhandled drift case: {drift}")

    attestation["members_sha256"] = hashlib.sha256(
        _canonical_json_bytes(members)
    ).hexdigest()
    if drift == "missing_self_hash":
        audit.pop("audit_self_sha256")
        audit_path.write_bytes(_canonical_json_bytes(audit) + b"\n")
    else:
        _write_self_hashed_audit(audit_path, audit)

    with pytest.raises(ValueError, match=error):
        freeze_production_adapter_output(
            tmp_path,
            suite_name="mind2web",
            prompt_path=output,
            audit_path=audit_path,
            plan_attestation_path=plan_attestation_path,
        )


def test_generic_freezer_rederives_mind2web_v2_from_raw_archive_and_ranker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources, archive = _production_fixture(tmp_path, monkeypatch)
    output = tmp_path / "mind2web-production-prompts.jsonl"
    audit_path = tmp_path / "mind2web-production-audit.json"
    export_mind2web_prompt_rows(
        sources,
        output,
        revision=PRODUCTION_MIND2WEB_REVISION,
        split=PRODUCTION_MIND2WEB_SPLIT,
        archive=archive,
        audit_path=audit_path,
        ranker_config_path=RANKER_CONFIG_PATH,
    )

    frozen = freeze_production_adapter_output(
        tmp_path,
        suite_name="mind2web",
        prompt_path=output,
        audit_path=audit_path,
    )

    assert frozen["suite"]["adapter"]["version"] == (
        MIND2WEB_RANKED_PROMPT_ADAPTER_VERSION
    )
    assert {artifact["role"] for artifact in frozen["raw_artifacts"]} == {
        "mind2web_archive_source",
        "mind2web_ranker_config",
    }
    assert frozen["adapter_provenance"][0]["audit_schema_version"] == 3


def test_generic_freezer_rejects_recomputed_mind2web_prompt_forgery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources, archive = _production_fixture(tmp_path, monkeypatch)
    output = tmp_path / "mind2web-production-prompts.jsonl"
    audit_path = tmp_path / "mind2web-production-audit.json"
    export_mind2web_prompt_rows(
        sources,
        output,
        revision=PRODUCTION_MIND2WEB_REVISION,
        split=PRODUCTION_MIND2WEB_SPLIT,
        archive=archive,
        audit_path=audit_path,
        ranker_config_path=RANKER_CONFIG_PATH,
    )
    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["prompt"] = "Caller-authored replacement prompt."
    output.write_bytes(
        b"".join(_canonical_json_bytes(row) + b"\n" for row in rows)
    )
    forged_payload = output.read_bytes()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    forged_identity = {
        "bytes": len(forged_payload),
        "sha256": hashlib.sha256(forged_payload).hexdigest(),
    }
    audit["output"].update(forged_identity)
    audit["freeze_binding"]["output"].update(forged_identity)
    _write_self_hashed_audit(audit_path, audit)

    with pytest.raises(ValueError, match="differs from the raw-source reexport"):
        freeze_production_adapter_output(
            tmp_path,
            suite_name="mind2web",
            prompt_path=output,
            audit_path=audit_path,
        )


def test_generic_freezer_rejects_drifted_mind2web_ranker_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources, archive = _production_fixture(tmp_path, monkeypatch)
    output = tmp_path / "mind2web-production-prompts.jsonl"
    audit_path = tmp_path / "mind2web-production-audit.json"
    export_mind2web_prompt_rows(
        sources,
        output,
        revision=PRODUCTION_MIND2WEB_REVISION,
        split=PRODUCTION_MIND2WEB_SPLIT,
        archive=archive,
        audit_path=audit_path,
        ranker_config_path=RANKER_CONFIG_PATH,
    )
    drifted_config = tmp_path / RANKER_CONFIG_PATH.name
    payload = RANKER_CONFIG_PATH.read_bytes()
    assert b'"minimum_dom_bytes":768' in payload
    drifted_config.write_bytes(
        payload.replace(
            b'"minimum_dom_bytes":768',
            b'"minimum_dom_bytes":769',
        )
    )

    with pytest.raises(ValueError, match="self-hash mismatch"):
        freeze_production_adapter_output(
            tmp_path,
            suite_name="mind2web",
            prompt_path=output,
            audit_path=audit_path,
            ranker_config_path=drifted_config,
        )


def test_mind2web_production_policy_pins_exact_official_members_and_counts() -> None:
    assert dict(PRODUCTION_MIND2WEB_TASK_COUNTS) == {
        "cross_domain": 912,
        "cross_task": 252,
        "cross_website": 177,
    }
    assert PRODUCTION_MIND2WEB_MEMBERS["cross_domain"] == tuple(
        f"test_domain/test_domain_{index}.json" for index in range(10)
    )
    assert PRODUCTION_MIND2WEB_MEMBERS["cross_task"] == tuple(
        f"test_task/test_task_{index}.json" for index in range(3)
    )
    assert PRODUCTION_MIND2WEB_MEMBERS["cross_website"] == tuple(
        f"test_website/test_website_{index}.json" for index in range(2)
    )
    assert sum(len(members) for members in PRODUCTION_MIND2WEB_MEMBERS.values()) == 15


def test_mind2web_production_rejects_arbitrary_caller_shard_without_archive(
    tmp_path: Path,
) -> None:
    source = _write_source(
        tmp_path / "cross_domain.json",
        [
            _task(
                "fabricated",
                task="Fabricated task",
                actions=[_action("step", context="<button>Fake</button>", gold="gold")],
                gold="gold",
            )
        ],
    )
    with pytest.raises(ValueError, match="requires the protected test.zip archive"):
        export_mind2web_prompt_rows(
            [source],
            tmp_path / "must-not-exist.jsonl",
            revision=PRODUCTION_MIND2WEB_REVISION,
            split=PRODUCTION_MIND2WEB_SPLIT,
            ranker_config_path=RANKER_CONFIG_PATH,
        )
    assert not (tmp_path / "must-not-exist.jsonl").exists()


def test_mind2web_production_rejects_archive_identity_and_plaintext_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources, archive = _production_fixture(tmp_path, monkeypatch)
    wrong_archive = Mind2WebArchive(
        path=archive.path,
        bytes=archive.bytes,
        sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="archive identity mismatch"):
        export_mind2web_prompt_rows(
            sources,
            tmp_path / "wrong-archive.jsonl",
            revision=PRODUCTION_MIND2WEB_REVISION,
            split=PRODUCTION_MIND2WEB_SPLIT,
            archive=wrong_archive,
            ranker_config_path=RANKER_CONFIG_PATH,
        )

    drifted_path = sources[0].path
    drifted_payload = drifted_path.read_bytes().replace(b"Visible", b"Mutated")
    assert len(drifted_payload) == sources[0].bytes
    drifted_path.write_bytes(drifted_payload)
    drifted_source = _identity(
        drifted_path,
        archive_member=sources[0].archive_member,
    )
    with pytest.raises(ValueError, match="plaintext does not match extracted source"):
        export_mind2web_prompt_rows(
            [drifted_source, *sources[1:]],
            tmp_path / "wrong-plaintext.jsonl",
            revision=PRODUCTION_MIND2WEB_REVISION,
            split=PRODUCTION_MIND2WEB_SPLIT,
            archive=archive,
            ranker_config_path=RANKER_CONFIG_PATH,
        )


@pytest.mark.parametrize(
    ("variant", "message"),
    [
        ("missing", "archive member set mismatch"),
        ("extra", "archive member set mismatch"),
        ("renamed", "archive member set mismatch"),
        ("traversal", "unsafe Mind2Web archive member path"),
        ("duplicate", "duplicate member names"),
    ],
)
def test_mind2web_production_rejects_archive_member_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    message: str,
) -> None:
    payloads = _production_payloads()
    sources = _write_member_sources(tmp_path / "extracted", payloads)
    members = sorted(payloads.items())
    if variant == "missing":
        members.pop()
    elif variant == "extra":
        members.append(("unexpected/extra.json", next(iter(payloads.values()))))
    elif variant == "renamed":
        _, payload = members[0]
        members[0] = ("test_domain/test_domain_9.json", payload)
    elif variant == "traversal":
        members.append(("../escape.json", next(iter(payloads.values()))))
    elif variant == "duplicate":
        members.append(members[0])
    archive = _write_archive(tmp_path / f"{variant}.zip", members)
    _patch_production_pins(
        monkeypatch,
        archive,
        total_uncompressed_bytes=sum(len(payload) for payload in payloads.values()),
    )

    with pytest.raises(ValueError, match=message):
        export_mind2web_prompt_rows(
            sources,
            tmp_path / f"{variant}.jsonl",
            revision=PRODUCTION_MIND2WEB_REVISION,
            split=PRODUCTION_MIND2WEB_SPLIT,
            archive=archive,
            ranker_config_path=RANKER_CONFIG_PATH,
        )


def test_mind2web_production_rejects_encryption_and_task_count_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources, archive = _production_fixture(tmp_path, monkeypatch, domain_tasks=2)
    with pytest.raises(ValueError, match="task counts mismatch"):
        export_mind2web_prompt_rows(
            sources,
            tmp_path / "wrong-counts.jsonl",
            revision=PRODUCTION_MIND2WEB_REVISION,
            split=PRODUCTION_MIND2WEB_SPLIT,
            archive=archive,
            ranker_config_path=RANKER_CONFIG_PATH,
        )

    monkeypatch.setattr(
        mind2web_prompts,
        "PRODUCTION_MIND2WEB_ARCHIVE_ENCRYPTED",
        True,
    )
    with pytest.raises(ValueError, match="encryption mismatch"):
        export_mind2web_prompt_rows(
            sources,
            tmp_path / "unencrypted.jsonl",
            revision=PRODUCTION_MIND2WEB_REVISION,
            split=PRODUCTION_MIND2WEB_SPLIT,
            archive=archive,
            ranker_config_path=RANKER_CONFIG_PATH,
        )


def test_mind2web_emits_only_strictly_prior_action_representations(
    tmp_path: Path,
) -> None:
    action_reprs = [
        "PRIOR_ACTION_CANARY",
        "CURRENT_ACTION_CANARY",
        "FUTURE_ACTION_CANARY",
    ]
    task = _task(
        "history-task",
        task="Use the visible controls",
        actions=[
            _action("step-0", context="<button>First</button>", gold="gold-0"),
            _action("step-1", context="<button>Second</button>", gold="gold-1"),
            _action("step-2", context="<button>Third</button>", gold="gold-2"),
        ],
        gold="unused",
        action_reprs=action_reprs,
    )
    source = _write_source(tmp_path / "history.json", [task])
    output = tmp_path / "history-prompts.jsonl"

    audit = export_mind2web_prompt_rows(
        [source],
        output,
        revision=REVISION,
    )
    prompts = [row["prompt"] for row in _read_canonical_rows(output)]

    assert all(canary not in prompts[0] for canary in action_reprs)
    assert "PRIOR_ACTION_CANARY" in prompts[1]
    assert "CURRENT_ACTION_CANARY" not in prompts[1]
    assert "FUTURE_ACTION_CANARY" not in prompts[1]
    assert "PRIOR_ACTION_CANARY" in prompts[2]
    assert "CURRENT_ACTION_CANARY" in prompts[2]
    assert "FUTURE_ACTION_CANARY" not in prompts[2]
    assert audit["label_isolation"]["prior_action_representations_emitted"] is True


@pytest.mark.parametrize(
    ("action_reprs", "message"),
    [
        ([], "align one-to-one"),
        ([123], r"action_reprs\[0\].*non-empty string"),
    ],
)
def test_mind2web_rejects_misaligned_or_nonstring_action_history(
    tmp_path: Path,
    action_reprs: list[object],
    message: str,
) -> None:
    task = _task(
        "history-task",
        task="Use the visible control",
        actions=[_action("step-0", context="<button>First</button>", gold="gold")],
        gold="gold",
    )
    task["action_reprs"] = action_reprs
    source = _write_source(tmp_path / "history.json", [task])

    with pytest.raises((TypeError, ValueError), match=message):
        export_mind2web_prompt_rows(
            [source],
            tmp_path / "history-prompts.jsonl",
            revision=REVISION,
        )


def test_mind2web_export_rejects_schema_drift_and_identity_drift(
    tmp_path: Path,
) -> None:
    task = _task(
        "task-a",
        task="Task A",
        actions=[_action("a-0", context="<button>A</button>", gold="gold")],
        gold="gold",
    )
    task["expected_calls"] = []
    source = _write_source(tmp_path / "train_0.json", [task])

    with pytest.raises(ValueError, match="schema drift"):
        export_mind2web_prompt_rows(
            [source],
            tmp_path / "output.jsonl",
            revision=REVISION,
        )

    clean_task = dict(task)
    clean_task.pop("expected_calls")
    clean_source = _write_source(tmp_path / "clean.json", [clean_task])
    bad_identity = Mind2WebSource(
        path=clean_source.path,
        bytes=clean_source.bytes,
        sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        export_mind2web_prompt_rows(
            [bad_identity],
            tmp_path / "identity-output.jsonl",
            revision=REVISION,
        )


def test_mind2web_export_fails_closed_on_prompt_and_row_caps(tmp_path: Path) -> None:
    task = _task(
        "task-a",
        task="Task A",
        actions=[
            _action("a-0", context="<p>" + "x" * 200 + "</p>", gold="gold-a"),
            _action("a-1", context="<p>second</p>", gold="gold-b"),
        ],
        gold="gold",
    )
    source = _write_source(tmp_path / "train_0.json", [task])

    with pytest.raises(ValueError, match="max_prompt_bytes"):
        export_mind2web_prompt_rows(
            [source],
            tmp_path / "prompt-cap.jsonl",
            revision=REVISION,
            max_prompt_bytes=64,
        )
    assert not (tmp_path / "prompt-cap.jsonl").exists()

    with pytest.raises(ValueError, match="max_rows=1"):
        export_mind2web_prompt_rows(
            [source],
            tmp_path / "row-cap.jsonl",
            revision=REVISION,
            max_rows=1,
        )
    assert not (tmp_path / "row-cap.jsonl").exists()


def test_mind2web_export_rejects_drifted_existing_output(tmp_path: Path) -> None:
    task = _task(
        "task-a",
        task="Task A",
        actions=[_action("a-0", context="<button>A</button>", gold="gold")],
        gold="gold",
    )
    source = _write_source(tmp_path / "train_0.json", [task])
    output = tmp_path / "output.jsonl"
    output.write_text("user-owned bytes\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        export_mind2web_prompt_rows(
            [source],
            output,
            revision=REVISION,
        )
    assert output.read_text(encoding="utf-8") == "user-owned bytes\n"


def test_mind2web_export_parses_verified_snapshot_after_same_stat_source_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_task = _task(
        "task-a",
        task="Task A",
        actions=[_action("a-0", context="<button>ORIGINAL</button>", gold="gold")],
        gold="gold",
    )
    source = _write_source(tmp_path / "train_0.json", [original_task])
    source_stat = source.path.stat()
    mutated_task = _task(
        "task-a",
        task="Task A",
        actions=[_action("a-0", context="<button>MUTATED!</button>", gold="gold")],
        gold="gold",
    )
    mutated_payload = json.dumps([mutated_task], ensure_ascii=False).encode("utf-8")
    assert len(mutated_payload) == source.bytes
    snapshot_source = mind2web_prompts._snapshot_verified_source

    def race_after_snapshot(
        source_to_copy: Mind2WebSource,
        snapshot_path: Path,
        *,
        max_source_bytes: int,
    ) -> tuple[Path, dict[str, int | str]]:
        verified = snapshot_source(
            source_to_copy,
            snapshot_path,
            max_source_bytes=max_source_bytes,
        )
        source_to_copy.path.write_bytes(mutated_payload)
        os.utime(
            source_to_copy.path,
            ns=(source_to_copy.path.stat().st_atime_ns, source_stat.st_mtime_ns),
        )
        return verified

    monkeypatch.setattr(
        mind2web_prompts,
        "_snapshot_verified_source",
        race_after_snapshot,
    )
    output = tmp_path / "snapshot-output.jsonl"
    audit = export_mind2web_prompt_rows(
        [source],
        output,
        revision=REVISION,
    )

    raced_stat = source.path.stat()
    assert (raced_stat.st_ino, raced_stat.st_size, raced_stat.st_mtime_ns) == (
        source_stat.st_ino,
        source_stat.st_size,
        source_stat.st_mtime_ns,
    )
    output_text = output.read_text(encoding="utf-8")
    assert "ORIGINAL" in output_text
    assert "MUTATED!" not in output_text
    assert audit["sources"][0]["sha256"] == source.sha256


def test_mind2web_export_rejects_raced_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _task(
        "task-a",
        task="Task A",
        actions=[_action("a-0", context="<button>A</button>", gold="gold")],
        gold="gold",
    )
    source = _write_source(tmp_path / "train_0.json", [task])
    output = tmp_path / "output.jsonl"

    def race_destination(_source: str | Path, destination: str | Path) -> None:
        Path(destination).write_text("raced bytes\n", encoding="utf-8")
        raise FileExistsError

    monkeypatch.setattr(mind2web_prompts.os, "link", race_destination)
    with pytest.raises(RuntimeError, match="concurrently published.*does not match"):
        export_mind2web_prompt_rows(
            [source],
            output,
            revision=REVISION,
        )
    assert output.read_text(encoding="utf-8") == "raced bytes\n"


def test_mind2web_export_removes_new_output_when_audit_publication_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _task(
        "task-a",
        task="Task A",
        actions=[_action("a-0", context="<button>A</button>", gold="gold")],
        gold="gold",
    )
    source = _write_source(tmp_path / "train_0.json", [task])
    output = tmp_path / "output.jsonl"
    audit_path = tmp_path / "output.audit.json"
    real_link = mind2web_prompts.os.link

    def race_audit(source_path: str | Path, destination: str | Path) -> None:
        if Path(destination) == audit_path:
            Path(destination).write_text("raced audit bytes\n", encoding="utf-8")
            raise FileExistsError
        real_link(source_path, destination)

    monkeypatch.setattr(mind2web_prompts.os, "link", race_audit)
    with pytest.raises(RuntimeError, match="concurrently published.*audit"):
        export_mind2web_prompt_rows(
            [source],
            output,
            revision=REVISION,
            audit_path=audit_path,
        )

    assert not output.exists()
    assert audit_path.read_text(encoding="utf-8") == "raced audit bytes\n"
