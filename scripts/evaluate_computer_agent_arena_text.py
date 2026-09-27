#!/usr/bin/env python
"""Run an instruction-only Computer Agent Arena action-family probe.

This is a deliberately narrow diagnostic for the current text-first WebGPU checkpoint.  It uses
only the published task instruction and the first parseable action from each trajectory; screenshots,
thought traces, later observations, and action arguments are excluded.  The result measures the
model's action prior and route gate, not visual grounding, trajectory completion, or native desktop
success.  The source remains evaluation-only and is never passed to an optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from localagent.agent.constrained import hybrid_decode
from localagent.agent.parser import extract_tool_calls
from localagent.agent.runtime import Agent
from localagent.agent.routes import ROUTES
from localagent.agent.tool_head import _feat
from localagent.agent.toolset import STANDARD_TOOLS
from localagent.agent.tools import ToolRegistry
from scripts.profile_computer_agent_arena import _action_name, action_family


DATASET = "xlangai/computer-agent-arena"
REVISION = "897b9f45287c516a44f9e79879b14bc3c1bc5b0a"
SOURCE_URL = "https://huggingface.co/datasets/xlangai/computer-agent-arena"

_COMPUTER_NAMES = frozenset(
    {
        "screenshot",
        "click",
        "double_click",
        "type_text",
        "key_press",
        "scroll",
        "drag",
        "wait",
        "move_cursor",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_tool(raw_name: str | None) -> str | None:
    """Project public desktop primitives into the deployed text-grounded tool vocabulary."""

    name = (raw_name or "").casefold()
    if name in {"doubleclick", "double_click"}:
        return "double_click"
    if name in {"dragto", "drag", "left_click_drag"}:
        return "drag"
    if name in {"moveto", "mouse_move", "move", "cursor_position"}:
        return "move_cursor"
    if name in {
        "click",
        "left_click",
        "rightclick",
        "right_click",
        "middleclick",
        "middle_click",
        "tripleclick",
        "triple_click",
    }:
        return "click"
    if name in {"write", "typewrite", "type", "type_text"}:
        return "type_text"
    if name in {"press", "hotkey", "keydown", "keyup", "key", "keypress", "key_press", "hold_key"}:
        return "key_press"
    if name in {"scroll", "hscroll"}:
        return "scroll"
    if name in {"sleep", "wait"}:
        return "wait"
    if name == "screenshot":
        return "screenshot"
    return None


def prompt_for_instruction(instruction: str) -> str:
    """Construct the explicit no-vision prompt contract used by this probe."""

    return (
        f"Desktop task: {instruction.strip()}\n\n"
        "Screenshot and desktop accessibility tree are intentionally omitted in this text-only "
        "probe.\nChoose exactly one computer action or abstain."
    )


def load_cases(path: Path, *, limit: int, seed: int) -> list[dict[str, Any]]:
    """Load one first-action case per unique trajectory, with no thought/image fields."""

    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: row must be an object")
            task_id = row.get("task_id")
            instruction = row.get("instruction")
            trajectory = row.get("traj")
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError(f"line {line_number}: task_id must be non-empty text")
            if task_id in seen:
                raise ValueError(f"duplicate task_id: {task_id}")
            seen.add(task_id)
            if not isinstance(instruction, str) or not instruction.strip():
                continue
            if not isinstance(trajectory, list):
                raise ValueError(f"line {line_number}: traj must be a list")
            for step in trajectory:
                value = step.get("value", step) if isinstance(step, dict) else {}
                if not isinstance(value, dict):
                    continue
                code = value.get("code")
                if not isinstance(code, str) or not code.strip():
                    code = value.get("action")
                raw_name = _action_name(code)
                tool = canonical_tool(raw_name)
                if tool is None:
                    continue
                cases.append(
                    {
                        "task_id": task_id,
                        "instruction": instruction.strip(),
                        "gold_tool": tool,
                        "gold_family": action_family(raw_name),
                        "model": str(row.get("model", "<missing>")),
                        "human_eval_correctness": row.get("human_eval_correctness"),
                    }
                )
                break
    if not cases:
        raise ValueError("no parseable first-action cases")
    rng = random.Random(seed)
    rng.shuffle(cases)
    return cases[: min(limit, len(cases))]


def _computer_tools() -> list[Any]:
    return [spec for spec in STANDARD_TOOLS if spec.name in _COMPUTER_NAMES]


def _load_agent(checkpoint: Path, tokenizer: Path | None) -> tuple[Agent, list[Any]]:
    tools = _computer_tools()
    registry = ToolRegistry()
    for spec in tools:
        registry.register(spec, lambda **kwargs: kwargs)
    agent = Agent.from_checkpoint(checkpoint, registry, tokenizer_path=tokenizer)
    if agent.model is None or agent.tokenizer is None or agent.selector is None:
        raise ValueError("checkpoint must contain model, tokenizer, and dense selector state")
    return agent, tools


@torch.no_grad()
def evaluate(
    source: Path,
    checkpoint: Path,
    *,
    tokenizer: Path | None,
    limit: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    if device != "cpu":
        raise ValueError("the reproducible bounded probe currently supports CPU only")
    cases = load_cases(source, limit=limit, seed=seed)
    agent, tools = _load_agent(checkpoint, tokenizer)
    model = agent.model.to(device)
    model.eval()
    selector = agent.selector
    route_head = agent.route_head
    if route_head is None:
        raise ValueError("checkpoint must contain route_head state")

    records: list[dict[str, Any]] = []
    exact = family = abstained = route_correct = 0
    by_family: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_human: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for case in cases:
        prompt = prompt_for_instruction(case["instruction"])
        output = hybrid_decode(
            model,
            agent.tokenizer,
            prompt,
            tools,
            device=device,
            selector=selector,
            route_head=route_head,
            ptr_head=agent.ptr_head,
            top_m=1,
        )
        calls = extract_tool_calls(output)
        predicted = calls[0].name if calls else None
        predicted_family = action_family(predicted)
        tool_ok = predicted == case["gold_tool"]
        family_ok = predicted_family == case["gold_family"]
        feature = _feat(model, agent.tokenizer, prompt, device)
        route = ROUTES[int(route_head(feature).argmax(-1))]
        route_ok = route == "computer_use"
        exact += int(tool_ok)
        family += int(family_ok)
        abstained += int(predicted is None)
        route_correct += int(route_ok)
        bucket = by_family[case["gold_family"]]
        bucket["rows"] += 1
        bucket["tool_exact"] += int(tool_ok)
        bucket["family_exact"] += int(family_ok)
        human_key = str(case["human_eval_correctness"])
        human_bucket = by_human[human_key]
        human_bucket["rows"] += 1
        human_bucket["tool_exact"] += int(tool_ok)
        human_bucket["family_exact"] += int(family_ok)
        records.append(
            {
                "task_id": case["task_id"],
                "gold_tool": case["gold_tool"],
                "gold_family": case["gold_family"],
                "predicted_tool": predicted,
                "predicted_family": predicted_family,
                "route": route,
                "tool_exact": tool_ok,
                "family_exact": family_ok,
                "route_correct": route_ok,
            }
        )

    def ratios(values: dict[str, int]) -> dict[str, Any]:
        rows = values["rows"]
        return {
            "rows": rows,
            "tool_exact_rate": values["tool_exact"] / rows,
            "family_exact_rate": values["family_exact"] / rows,
        }

    checkpoint_bytes = checkpoint.stat().st_size
    payload: dict[str, Any] = {
        "kind": "localagent_computer_agent_arena_instruction_only_probe",
        "schema_version": 1,
        "dataset": DATASET,
        "source_url": SOURCE_URL,
        "source_revision": REVISION,
        "source": {
            "path": source.name,
            "bytes": source.stat().st_size,
            "sha256": _sha256(source),
            "split_policy": "evaluation_only",
        },
        "checkpoint": {
            "path": checkpoint.name,
            "bytes": checkpoint_bytes,
            "sha256": _sha256(checkpoint),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "prompt_contract": {
            "kind": "instruction_only_no_vision_v1",
            "uses_instruction": True,
            "uses_thought": False,
            "uses_screenshot": False,
            "uses_later_trajectory_steps": False,
            "arguments_scored": False,
            "tool_projection": "computer_arena_primitive_to_local_computer_family_v1",
        },
        "selection": {
            "seed": seed,
            "requested_limit": limit,
            "evaluated_rows": len(cases),
            "unique_parent_tasks": len({case["task_id"] for case in cases}),
        },
        "overall": {
            "tool_exact_rate": exact / len(cases),
            "family_exact_rate": family / len(cases),
            "route_accuracy": route_correct / len(cases),
            "abstention_rate": abstained / len(cases),
        },
        "by_gold_family": {key: ratios(dict(value)) for key, value in sorted(by_family.items())},
        "by_human_eval_correctness": {
            key: ratios(dict(value)) for key, value in sorted(by_human.items())
        },
        "records": records,
        "claim_boundary": (
            "Instruction-only action-family diagnostic. It omits screenshots, accessibility trees, "
            "thought traces, later trajectory state, and argument scoring; it is not a Computer "
            "Agent Arena score, AgentNetBench score, visual-grounding result, native desktop run, "
            "or training artifact."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.limit <= 1024:
        raise SystemExit("--limit must be in [1, 1024]")
    if args.output.exists():
        raise SystemExit("refusing to overwrite probe output")
    payload = evaluate(
        args.input,
        args.checkpoint,
        tokenizer=args.tokenizer,
        limit=args.limit,
        seed=args.seed,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
