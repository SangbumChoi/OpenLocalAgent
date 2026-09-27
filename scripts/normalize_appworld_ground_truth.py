#!/usr/bin/env python3
"""Normalize public AppWorld train/dev solutions into canonical Conversation JSONL.

The adapter is intentionally code-as-tool: AppWorld solutions are executable Python over its
``apis`` collection, while LocalAgent already has a schema-compatible ``run_python`` tool.  Train
and evaluation exports are separate by construction; test splits are rejected because their
ground-truth programs are protected.  Raw AppWorld task text and solution code stay outside Git;
the manifest records only source identities and hashes.

This is a data-preparation step, not a benchmark runner.  Native execution must still use an
isolated AppWorld environment and an independent task verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

from localagent.data.schema import Conversation, Message, Role, ToolCall
from localagent.agent.toolset import STANDARD_TOOLS


def _sha256(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _text_hash(value: str) -> dict[str, int | str]:
    encoded = value.encode("utf-8")
    return {"bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def _run_python_spec():
    for tool in STANDARD_TOOLS:
        if tool.name == "run_python":
            return tool
    raise RuntimeError("STANDARD_TOOLS is missing run_python")


def normalize(
    *, root: Path, split: str, purpose: str, output: Path, manifest: Path, max_tasks: int | None
) -> dict[str, Any]:
    if split not in {"train", "dev"}:
        raise ValueError("only AppWorld train/dev splits may be normalized; test splits are protected")
    if purpose not in {"train", "eval"}:
        raise ValueError("purpose must be train or eval")
    if (purpose, split) not in {("train", "train"), ("eval", "dev")}:
        raise ValueError("training export must use train; evaluation export must use disjoint dev")
    root = root.resolve()
    split_path = root / "data" / "datasets" / f"{split}.txt"
    version_path = root / "data" / "version.txt"
    if not split_path.is_file() or not version_path.is_file():
        raise FileNotFoundError("AppWorld data root must contain data/datasets and data/version.txt")

    try:
        from appworld import AppWorld, update_root
    except ImportError as error:
        raise RuntimeError("AppWorld must be installed in the optional evaluation environment") from error
    if Path(update_root(str(root))).resolve() != root:
        raise RuntimeError("AppWorld root did not resolve to the requested data root")
    try:
        appworld_version = importlib.metadata.version("appworld")
    except importlib.metadata.PackageNotFoundError:
        appworld_version = "unknown"

    task_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_tasks is not None:
        if max_tasks < 1:
            raise ValueError("max_tasks must be positive")
        task_ids = task_ids[:max_tasks]
    if not task_ids:
        raise ValueError(f"AppWorld {split} split is empty")

    tool = _run_python_spec()
    rows: list[str] = []
    task_sources: list[dict[str, Any]] = []
    for task_id in task_ids:
        spec_path = root / "data" / "tasks" / task_id / "specs.json"
        if not spec_path.is_file():
            raise FileNotFoundError(f"missing AppWorld task specs: {spec_path}")
        with AppWorld(
            task_id=task_id,
            experiment_name=f"localagent_normalize_{purpose}",
            load_ground_truth=True,
            ground_truth_mode="full",
        ) as world:
            instruction = str(world.task.instruction)
            ground_truth = world.task.ground_truth
            if ground_truth is None:
                raise ValueError(f"ground truth missing for public {split} task {task_id}")
            code = str(ground_truth.compiled_solution_code).rstrip() + "\nsolution(apis, requester)"
        if "def solution(" not in code or "apis." not in code:
            raise ValueError(f"ground truth for {task_id} is not an AppWorld API solution")
        row = Conversation(
            messages=[
                Message(role=Role.user, content=instruction),
                Message(
                    role=Role.assistant,
                    tool_calls=[ToolCall(name="run_python", arguments={"code": code})],
                ),
            ],
            tools=[tool],
            meta={
                "kind": "appworld_ground_truth_program",
                "parent_record_id": task_id,
                "source_dataset": "appworld",
                "source_split": split,
                "purpose": purpose,
                "data_version": version_path.read_text(encoding="utf-8").strip(),
                "instruction": _text_hash(instruction),
                "solution": _text_hash(code),
            },
        )
        rows.append(row.to_json())
        task_sources.append(
            {
                "task_id": task_id,
                "spec": _sha256(spec_path),
                "instruction": _text_hash(instruction),
                "solution": _text_hash(code),
            }
        )

    if output.exists() or output.is_symlink() or manifest.exists() or manifest.is_symlink():
        raise FileExistsError("refusing to overwrite AppWorld normalized output or manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    body = {
        "kind": "localagent_appworld_conversation_export",
        "schema_version": 1,
        "source": {
            "dataset": "appworld",
            "package_version": appworld_version,
            "data_version": version_path.read_text(encoding="utf-8").strip(),
            "root": str(root),
            "split": split,
            "split_file": _sha256(split_path),
            "purpose": purpose,
        },
        "output": _sha256(output),
        "rows": len(rows),
        "tasks": task_sources,
        "claim_boundary": (
            "Public AppWorld train/dev ground-truth programs normalized as run_python supervision. "
            "No protected test split, AppWorld-UL asset, external account, or native task score is "
            "included; source code remains outside the repository."
        ),
    }
    body["manifest_self_sha256"] = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("APPWORLD_ROOT", ".")))
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--purpose", choices=("train", "eval"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    result = normalize(
        root=args.root,
        split=args.split,
        purpose=args.purpose,
        output=args.output,
        manifest=args.manifest,
        max_tasks=args.max_tasks,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
