#!/usr/bin/env python3
"""Project a bounded AgentNet desktop trajectory sample into canonical text-only Conversations.

AgentNet is an image-grounded computer-use corpus.  This adapter deliberately drops screenshots
and keeps only the task, the annotator's textual observation, and actions that map to the existing
LocalAgent computer-use vocabulary.  It is intended for a reproducible text-only continuation
experiment, not for claiming AgentNetBench or native desktop success.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from localagent.data.schema import Conversation, Message, Role, ToolCall


_COORDINATE = re.compile(r"(?:x|y)\s*=\s*(-?\d+(?:\.\d+)?)")
def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _literal(value: str) -> Any:
    # AgentNet occasionally records keyword calls such as ``write(message='hello')``.  The
    # literal parser intentionally remains strict after removing the known keyword.
    if "=" in value and not value.lstrip().startswith(("[", "{", "(")):
        _, value = value.split("=", 1)
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"unsupported Python literal in AgentNet action: {value!r}") from error


def _coordinates(code: str) -> tuple[float, float] | None:
    values = [float(value) for value in _COORDINATE.findall(code)]
    if len(values) < 2:
        return None
    return values[0], values[1]


def _point_target(code: str, *, prefix: str = "") -> str | None:
    point = _coordinates(code)
    if point is None:
        return None
    x, y = point
    marker = f"{prefix};" if prefix else ""
    return f"{marker}x={x:.6f};y={y:.6f}"


def parse_action(code: str) -> tuple[str, dict[str, Any]] | None:
    """Map one AgentNet/PyAutoGUI action to a LocalAgent tool call.

    Unsupported actions return ``None`` instead of inventing an equivalent operation.  The
    caller records their names so filtering remains auditable.
    """

    text = code.strip()
    drag = re.search(
        r"pyautogui\.moveTo\([^\n]*\).*?pyautogui\.dragTo\(([^\n]*)\)", text, re.DOTALL
    )
    if drag:
        points = [tuple(map(float, pair)) for pair in re.findall(
            r"x\s*=\s*(-?\d+(?:\.\d+)?).*?y\s*=\s*(-?\d+(?:\.\d+)?)", text
        )]
        if len(points) >= 2:
            source, dest = points[0], points[1]
            return "drag", {
                "source": f"x={source[0]:.6f};y={source[1]:.6f}",
                "dest": f"x={dest[0]:.6f};y={dest[1]:.6f}",
            }
    scroll = re.search(r"pyautogui\.scroll\((?:clicks\s*=\s*)?([-+]?\d+(?:\.\d+)?)", text)
    if scroll:
        amount = float(scroll.group(1))
        return "scroll", {"direction": "up" if amount > 0 else "down"}
    if text.startswith("pyautogui.click"):
        target = _point_target(text, prefix="button=left")
        return ("click", {"target": target}) if target else None
    if text.startswith("pyautogui.doubleClick"):
        target = _point_target(text, prefix="clicks=2")
        return ("double_click", {"target": target}) if target else None
    if text.startswith("pyautogui.tripleClick"):
        target = _point_target(text, prefix="clicks=3")
        return ("double_click", {"target": target}) if target else None
    if text.startswith("pyautogui.rightClick"):
        target = _point_target(text, prefix="button=right")
        return ("click", {"target": target}) if target else None
    if text.startswith("pyautogui.moveTo"):
        target = _point_target(text)
        return ("move_cursor", {"target": target}) if target else None
    if text.startswith("pyautogui.write"):
        match = re.search(r"pyautogui\.write\((.*)\)\s*$", text)
        if not match:
            return None
        value = _literal(match.group(1))
        return ("type_text", {"text": str(value)}) if isinstance(value, str) else None
    if text.startswith("pyautogui.press"):
        match = re.search(r"pyautogui\.press\((.*)\)\s*$", text)
        if not match:
            return None
        value = _literal(match.group(1))
        if isinstance(value, list):
            value = value[0] if value else ""
        if not isinstance(value, str) or not value:
            return None
        aliases = {"return": "Enter", "esc": "Escape", "space": "Space"}
        return "key_press", {"key": aliases.get(value.lower(), value)}
    if text.startswith("pyautogui.hotkey"):
        match = re.search(r"pyautogui\.hotkey\((.*)\)\s*$", text)
        if not match:
            return None
        value = _literal(match.group(1))
        if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
            return None
        return "key_press", {"key": "+".join(value)}
    if text.startswith("computer.wait"):
        match = re.search(r"computer\.wait\(([-+]?\d+(?:\.\d+)?)", text)
        if match:
            return "wait", {"seconds": max(0, int(float(match.group(1))))}
        return "wait", {"seconds": 1}
    return None


def _records(path: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    incomplete = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                incomplete += 1
                break
            if not isinstance(row, dict) or not isinstance(row.get("traj"), list):
                raise ValueError("AgentNet rows must contain a trajectory list")
            records.append(row)
    return records, incomplete


def _conversation(
    row: dict[str, Any],
    step_index: int,
    code: str,
    name: str,
    arguments: dict[str, Any],
    *,
    dataset: str,
    revision: str,
    max_observation_chars: int,
) -> Conversation:
    task_id = str(row.get("task_id", ""))
    instruction = str(row.get("instruction", "")).strip()
    observation = str(row.get("value", {}).get("observation", "")).strip()
    if max_observation_chars < 1:
        raise ValueError("max_observation_chars must be positive")
    if len(observation) > max_observation_chars:
        observation = observation[:max_observation_chars].rstrip() + " …[observation truncated]"
    prompt = f"Task: {instruction}\nObservation: {observation}".strip()
    return Conversation(
        messages=[
            Message(role=Role.user, content=prompt),
            Message(role=Role.assistant, tool_calls=[ToolCall(name=name, arguments=arguments)]),
        ],
        meta={
            "kind": "agentnet_text_projection_v1",
            "parent_record_id": task_id,
            "source_dataset": dataset,
            "source_revision": revision,
            "source_split": "ubuntu_jsonl_bounded_prefix",
            "task_id": task_id,
            "step_index": step_index,
            "action_code": code,
            "images_dropped": True,
            "slot_values": {},
        },
    )


def project(
    path: Path,
    *,
    dataset: str,
    revision: str,
    eval_fraction: float = 0.2,
    seed: int = 2027,
    max_observation_chars: int = 1200,
    max_records: int | None = None,
) -> tuple[list[Conversation], list[Conversation], dict[str, Any]]:
    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be between zero and one")
    records, incomplete = _records(path)
    if max_records is not None:
        if max_records < 2:
            raise ValueError("max_records must be at least two when supplied")
        records = records[:max_records]
    rng = random.Random(seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    eval_count = max(1, round(len(records) * eval_fraction))
    eval_ids = {str(records[index].get("task_id", "")) for index in order[:eval_count]}
    train: list[Conversation] = []
    evaluation: list[Conversation] = []
    action_counts: Counter[str] = Counter()
    dropped_counts: Counter[str] = Counter()
    for row in records:
        task_id = str(row.get("task_id", ""))
        for step_index, step in enumerate(row.get("traj", [])):
            value = step.get("value", {}) if isinstance(step, dict) else {}
            code = str(value.get("code", "")).strip() if isinstance(value, dict) else ""
            parsed = parse_action(code)
            if parsed is None:
                dropped_counts[code.split("(", 1)[0] or "<empty>"] += 1
                continue
            name, arguments = parsed
            action_counts[name] += 1
            conversation = _conversation(
                {**row, "value": value},
                step_index,
                code,
                name,
                arguments,
                dataset=dataset,
                revision=revision,
                max_observation_chars=max_observation_chars,
            )
            (evaluation if task_id in eval_ids else train).append(conversation)
    if not train or not evaluation:
        raise ValueError("projection must produce non-empty train and evaluation rows")
    metadata = {
        "source": file_identity(path),
        "complete_parent_records": len(records),
        "incomplete_trailing_lines": incomplete,
        "train_parent_records": len(set(row.meta["parent_record_id"] for row in train)),
        "eval_parent_records": len(set(row.meta["parent_record_id"] for row in evaluation)),
        "train_rows": len(train),
        "eval_rows": len(evaluation),
        "action_counts": dict(sorted(action_counts.items())),
        "dropped_action_counts": dict(sorted(dropped_counts.items())),
        "seed": seed,
        "eval_fraction": eval_fraction,
        "max_observation_chars": max_observation_chars,
        "max_records": max_records,
        "projection": "task_plus_text_observation_to_existing_computer_tool_v1",
        "claim_boundary": "Offline text-only AgentNet trajectory continuation; images are dropped and this is not AgentNetBench, OSWorld, or native desktop success.",
    }
    return train, evaluation, metadata


def _write(rows: Iterable[Conversation], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.to_json() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--eval-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--dataset", default="xlangai/AgentNet")
    parser.add_argument("--revision", default="d76ee50a63fad81cfdbe576416757d7c2091ed50")
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--max-observation-chars", type=int, default=1200)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    train, evaluation, metadata = project(
        args.input,
        dataset=args.dataset,
        revision=args.revision,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
        max_observation_chars=args.max_observation_chars,
        max_records=args.max_records,
    )
    _write(train, args.train_output)
    _write(evaluation, args.eval_output)
    metadata["train_output"] = file_identity(args.train_output)
    metadata["eval_output"] = file_identity(args.eval_output)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
