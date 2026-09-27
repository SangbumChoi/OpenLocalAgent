#!/usr/bin/env python
"""Profile a pinned Computer Agent Arena metadata JSONL snapshot.

The benchmark is multimodal and evaluation-only.  This command reads the public JSONL metadata,
counts trajectory/action/text/image coverage, and writes a small provenance receipt.  It never
opens the referenced screenshots, downloads image archives, or turns benchmark rows into training
data.  Action parsing is deliberately conservative: free-form model code is classified by the
first recognizable computer-use primitive and is never executed.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any


DATASET = "xlangai/computer-agent-arena"
DATASET_URL = "https://huggingface.co/datasets/xlangai/computer-agent-arena"
REVISION = "897b9f45287c516a44f9e79879b14bc3c1bc5b0a"
LICENSE = "CC-BY-4.0"

_ACTION_RE = re.compile(
    r"(?:pyautogui|computer)\.(?P<name>click|doubleClick|rightClick|middleClick|"
    r"tripleClick|moveTo|dragTo|scroll|hscroll|write|typewrite|press|hotkey|keyDown|"
    r"keyUp|sleep|wait|screenshot|terminate)\b",
    re.IGNORECASE,
)
_JSON_ACTION_RE = re.compile(r"[\"']action[\"']\s*:\s*[\"'](?P<name>[A-Za-z_]+)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _action_name(code: object) -> str | None:
    if not isinstance(code, str):
        return None
    match = _ACTION_RE.search(code)
    if match:
        return match.group("name").casefold()
    match = _JSON_ACTION_RE.search(code)
    return match.group("name").casefold() if match else None


def action_family(name: str | None) -> str:
    """Map a raw action primitive to a stable modality-level family."""

    if name in {
        "click",
        "doubleclick",
        "double_click",
        "rightclick",
        "right_click",
        "middleclick",
        "middle_click",
        "tripleclick",
        "triple_click",
        "moveto",
        "move_cursor",
        "dragto",
        "drag",
        "left_click",
        "right_click",
        "middle_click",
        "mouse_move",
        "left_click_drag",
        "drag",
        "move",
        "cursor_position",
    }:
        return "pointer"
    if name in {"scroll", "hscroll"}:
        return "scroll"
    if name in {"write", "typewrite", "type", "type_text"}:
        return "type"
    if name in {
        "press",
        "hotkey",
        "keydown",
        "keyup",
        "key",
        "keypress",
        "key_press",
        "hold_key",
    }:
        return "keyboard"
    if name in {"sleep", "wait"}:
        return "wait"
    if name == "screenshot":
        return "observation"
    if name in {"terminate", "done", "fail", "call_user"}:
        return "termination_or_handoff"
    return "unknown"


def _row_value(step: object) -> dict[str, Any]:
    if not isinstance(step, dict):
        return {}
    value = step.get("value")
    return value if isinstance(value, dict) else step


def profile(path: Path, *, revision: str = REVISION) -> dict[str, Any]:
    """Return a deterministic metadata profile for one public JSONL snapshot."""

    rows = 0
    steps = 0
    image_refs = 0
    thought_steps = 0
    observation_steps = 0
    code_steps = 0
    parseable_actions = 0
    task_ids: list[str] = []
    correctness: collections.Counter[str] = collections.Counter()
    models: collections.Counter[str] = collections.Counter()
    raw_actions: collections.Counter[str] = collections.Counter()
    families: collections.Counter[str] = collections.Counter()
    malformed_steps = 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: row must be an object")
            task_id = row.get("task_id")
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError(f"line {line_number}: task_id must be non-empty text")
            trajectory = row.get("traj")
            if not isinstance(trajectory, list):
                raise ValueError(f"line {line_number}: traj must be a list")
            rows += 1
            task_ids.append(task_id)
            models[str(row.get("model", "<missing>"))] += 1
            correctness[str(row.get("human_eval_correctness", "<missing>"))] += 1
            for step in trajectory:
                steps += 1
                if not isinstance(step, dict):
                    malformed_steps += 1
                    continue
                if step.get("image") or _row_value(step).get("image"):
                    image_refs += 1
                value = _row_value(step)
                thought = value.get("thought")
                observation = value.get("observation")
                if isinstance(thought, str) and thought.strip():
                    thought_steps += 1
                if isinstance(observation, str) and observation.strip():
                    observation_steps += 1
                code = value.get("code")
                if not isinstance(code, str) or not code.strip():
                    code = value.get("action")
                if isinstance(code, str) and code.strip():
                    code_steps += 1
                    name = _action_name(code)
                    if name is not None:
                        parseable_actions += 1
                        raw_actions[name] += 1
                        families[action_family(name)] += 1
                    else:
                        families["unknown"] += 1

    if rows == 0:
        raise ValueError("input JSONL is empty")
    duplicate_ids = len(task_ids) - len(set(task_ids))
    correct = int(correctness.get("1", 0))
    payload: dict[str, Any] = {
        "kind": "localagent_computer_agent_arena_metadata_receipt",
        "schema_version": 1,
        "dataset": DATASET,
        "source_url": DATASET_URL,
        "revision": revision,
        "license": LICENSE,
        "source": {
            "path": path.name,
            "source_file_url": (
                f"{DATASET_URL}/resolve/{revision}/agent_arena_data.jsonl"
            ),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "format": "agent_arena_data.jsonl",
            "image_archives_downloaded": False,
        },
        "coverage": {
            "trajectories": rows,
            "steps": steps,
            "unique_task_ids": len(set(task_ids)),
            "duplicate_task_id_rows": duplicate_ids,
            "image_reference_steps": image_refs,
            "thought_steps": thought_steps,
            "observation_steps": observation_steps,
            "code_steps": code_steps,
            "parseable_action_steps": parseable_actions,
            "malformed_steps": malformed_steps,
        },
        "human_eval_correctness": dict(sorted(correctness.items())),
        "human_eval_correct_rate": correct / rows,
        "models": dict(sorted(models.items(), key=lambda pair: (-pair[1], pair[0]))),
        "raw_action_names": dict(sorted(raw_actions.items(), key=lambda pair: (-pair[1], pair[0]))),
        "action_families": dict(sorted(families.items(), key=lambda pair: (-pair[1], pair[0]))),
        "claim_boundary": (
            "Metadata and action-coverage audit only; screenshots were not read, no image archive "
            "was downloaded, no benchmark trajectory was used for training, and no official "
            "Computer Agent Arena or native desktop score is claimed."
        ),
    }
    payload["profile_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    payload = profile(args.input, revision=args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
