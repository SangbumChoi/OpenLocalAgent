from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


_SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_derived_corpora.py"
_SPEC = importlib.util.spec_from_file_location("prepare_derived_corpora_cli", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_cli_maps_paper_groups_and_requires_complete_parent_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, Any] = {}

    def fake_prepare(**kwargs: Any) -> dict[str, dict[str, Any]]:
        captured.update(kwargs)
        return {
            str(tmp_path / "general"): {
                "generation": "a" * 32,
                "total_documents": 3,
                "splits": {"train": {"documents": 2}, "val": {"documents": 1}},
            }
        }

    monkeypatch.setattr(_MODULE, "prepare_derived_corpora", fake_prepare)
    assert (
        _MODULE.main(
            [
                "--freeze",
                "freeze.json",
                "--freeze-spec",
                "freeze.yaml",
                "--parent-filtered-jsonl",
                "filtered.jsonl",
                "--parent-manifest",
                "parent/manifest.json",
                "--tokenizer",
                "tokenizer.json",
                "--project-root",
                str(tmp_path),
                "--group",
                "general=fineweb_edu_dedup+cosmopedia_v2",
                "--group",
                "code=permissive_python",
                "--group",
                "structured=structured_html",
            ]
        )
        == 0
    )
    assert captured["groups"] == {
        "general": ("cosmopedia_v2", "fineweb_edu_dedup"),
        "code": ("permissive_python",),
        "structured": ("structured_html",),
    }
    assert captured["require_complete_parent"] is True
    assert json.loads(capsys.readouterr().out)[str(tmp_path / "general")]["documents"] == 3


def test_cli_supports_explicit_partial_mode_and_rejects_duplicate_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_prepare(**kwargs: Any) -> dict[str, dict[str, Any]]:
        captured.update(kwargs)
        return {
            str(Path("general").absolute()): {
                "generation": "b" * 32,
                "total_documents": 1,
                "splits": {"train": {"documents": 1}, "val": {"documents": 0}},
            }
        }

    monkeypatch.setattr(_MODULE, "prepare_derived_corpora", fake_prepare)
    _MODULE.main(
        [
            "--freeze",
            "freeze.json",
            "--freeze-spec",
            "freeze.yaml",
            "--parent-filtered-jsonl",
            "filtered.jsonl",
            "--parent-manifest",
            "parent/manifest.json",
            "--tokenizer",
            "tokenizer.json",
            "--group",
            "general=fineweb_edu_dedup",
            "--no-require-complete-parent",
        ]
    )
    assert captured["require_complete_parent"] is False

    with pytest.raises(SystemExit) as error:
        _MODULE.main(
            [
                "--freeze",
                "freeze.json",
                "--freeze-spec",
                "freeze.yaml",
                "--parent-filtered-jsonl",
                "filtered.jsonl",
                "--parent-manifest",
                "parent/manifest.json",
                "--tokenizer",
                "tokenizer.json",
                "--group",
                "general=fineweb_edu_dedup",
                "--group",
                "general=cosmopedia_v2",
            ]
        )
    assert error.value.code == 2
