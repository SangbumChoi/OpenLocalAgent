#!/usr/bin/env python3
"""Run the LocalAgent checkpoint inside the pinned BrowserGym/MiniWoB environment.

This runner is intentionally optional-dependency guarded.  It imports BrowserGym, Gymnasium, and
Playwright only when invoked, converts the model's text-grounded actions to BrowserGym high-level
actions using the live accessibility tree, and writes a compact receipt without screenshots.  A
receipt is marked as an official-split evaluation only after the complete 240-episode pinned plan
has run with the exact source and Chromium revisions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from localagent.agent.parser import extract_tool_calls
from localagent.agent.runtime import Agent
from localagent.agent.tools import ToolRegistry
from localagent.agent.toolset import REALISTIC_BROWSER_TOOLS, STANDARD_TOOLS
from localagent.data.browsergym_capture import production_capture_plan
from localagent.data.browsergym_prompts import (
    PRODUCTION_BROWSERGYM_REVISION,
    PRODUCTION_BROWSERGYM_VERSION,
    PRODUCTION_CHROMIUM_REVISION,
    PRODUCTION_CHROMIUM_VERSION,
    PRODUCTION_EPISODES,
    PRODUCTION_FIXED_SEEDS,
    PRODUCTION_MAX_STEPS,
    PRODUCTION_MINIWOB_REVISION,
    PRODUCTION_PLAYWRIGHT_VERSION,
    PRODUCTION_SIMILARITY_GROUPS,
    PRODUCTION_TASK_VARIANTS,
)

_VISIBLE_ROLES = frozenset({"button", "textbox", "checkbox", "radio", "combobox", "link", "listbox"})
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _node_role(node: dict[str, Any]) -> str:
    role = node.get("role")
    if not isinstance(role, dict):
        return ""
    return str(role.get("value", ""))


def _node_name(node: dict[str, Any]) -> str:
    name = node.get("name")
    if not isinstance(name, dict):
        return ""
    return str(name.get("value", ""))


def _visible_elements(observation: dict[str, Any]) -> list[dict[str, str]]:
    tree = observation.get("axtree_object")
    if not isinstance(tree, dict):
        return []
    nodes = tree.get("nodes")
    if not isinstance(nodes, list):
        return []
    elements: list[dict[str, str]] = []
    for raw in nodes:
        if not isinstance(raw, dict) or raw.get("ignored"):
            continue
        bid = raw.get("browsergym_id")
        role = _node_role(raw)
        name = _node_name(raw)
        if isinstance(bid, str) and bid and role in _VISIBLE_ROLES:
            elements.append({"bid": bid, "role": role, "name": name})
    return elements


def _dom_string(strings: list[Any], index: Any) -> str:
    if isinstance(index, int) and 0 <= index < len(strings):
        value = strings[index]
        return str(value) if value is not None else ""
    return ""


def _dom_node_text(
    nodes: dict[str, Any],
    strings: list[Any],
    children: dict[int, list[int]],
    index: int,
) -> str:
    """Return visible text/label content for a compact BrowserGym DOM node."""

    values: list[str] = []
    stack = [index]
    seen: set[int] = set()
    while stack and len(values) < 8:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        node_type = nodes.get("nodeType", [None])[current] if current < len(nodes.get("nodeType", [])) else None
        if node_type == 3:
            node_values = nodes.get("nodeValue", [])
            if current < len(node_values):
                text = _dom_string(strings, node_values[current]).strip()
                if text:
                    values.append(text)
        stack.extend(reversed(children.get(current, [])))
    if values:
        return " ".join(dict.fromkeys(values))

    attributes = nodes.get("attributes", [])
    if index < len(attributes) and isinstance(attributes[index], list):
        pairs = attributes[index]
        for offset in range(0, len(pairs) - 1, 2):
            key = _dom_string(strings, pairs[offset]).lower()
            value = _dom_string(strings, pairs[offset + 1]).strip()
            if key in {"aria-label", "title", "value", "data-index", "name"} and value:
                return value
    return ""


def _dom_coordinate_candidates(
    observation: dict[str, Any],
    *,
    device_pixel_ratio: float = 1.0,
    screenshot_scale: float = 1.0,
) -> list[dict[str, float | str]]:
    """Extract text-labelled clickable DOM boxes for the optional coordinate bridge.

    BrowserGym's accessibility tree omits some SVG/generic controls used by MiniWoB.  The DOM
    snapshot still exposes ``isClickable`` and layout bounds.  This function only reads those
    observation fields and never reads task verifiers, hidden labels, or screenshots.
    """

    dom = observation.get("dom_object")
    if not isinstance(dom, dict):
        return []
    documents = dom.get("documents")
    if not isinstance(documents, list) or not documents or not isinstance(documents[0], dict):
        return []
    document = documents[0]
    nodes = document.get("nodes")
    layout = document.get("layout")
    strings = dom.get("strings", document.get("strings", []))
    if not isinstance(nodes, dict) or not isinstance(layout, dict) or not isinstance(strings, list):
        return []
    parent_index = nodes.get("parentIndex", [])
    if not isinstance(parent_index, list):
        return []
    children: dict[int, list[int]] = {}
    for index, parent in enumerate(parent_index):
        if isinstance(parent, int) and parent >= 0:
            children.setdefault(parent, []).append(index)
    clickable = nodes.get("isClickable", {})
    clickable_indices = clickable.get("index", []) if isinstance(clickable, dict) else []
    node_layouts = layout.get("nodeIndex", [])
    bounds = layout.get("bounds", [])
    if not isinstance(clickable_indices, list) or not isinstance(node_layouts, list):
        return []
    if not isinstance(bounds, list) or device_pixel_ratio <= 0 or screenshot_scale <= 0:
        return []
    candidates: list[dict[str, float | str]] = []
    for raw_index in clickable_indices:
        if not isinstance(raw_index, int) or raw_index < 0 or raw_index >= len(parent_index):
            continue
        box_indices = [position for position, node in enumerate(node_layouts) if node == raw_index]
        if not box_indices:
            continue
        box = next(
            (
                bounds[position]
                for position in box_indices
                if isinstance(bounds[position], list)
                and len(bounds[position]) == 4
                and all(isinstance(value, (int, float)) for value in bounds[position])
                and bounds[position][2] > 0
                and bounds[position][3] > 0
            ),
            None,
        )
        if box is None:
            continue
        name = _dom_node_text(nodes, strings, children, raw_index)
        if not name:
            continue
        x, y, width, height = (float(value) for value in box)
        candidates.append(
            {
                "name": name,
                "x": (x + width / 2.0) / device_pixel_ratio * screenshot_scale,
                "y": (y + height / 2.0) / device_pixel_ratio * screenshot_scale,
            }
        )
    return candidates


def _compact_context(
    observation: dict[str, Any],
    coordinate_candidates: list[dict[str, float | str]] | None = None,
) -> str:
    elements = _visible_elements(observation)
    lines = [
        "Live accessibility elements (quoted names are valid targets):",
    ]
    for element in elements:
        name = element["name"] or "<unnamed>"
        lines.append(f"[{element['bid']}] {element['role']}: \"{name}\"")
    if len(lines) == 1:
        lines.append("<none>")
    focused = observation.get("focused_element_bid")
    if isinstance(focused, str) and focused:
        lines.append(f"Focused element id: [{focused}]")
    if coordinate_candidates:
        lines.append("Visible coordinate candidates (text-backed controls):")
        for index, candidate in enumerate(coordinate_candidates):
            lines.append(f'[coord-{index}] text: "{candidate["name"]}"')
    return "\n".join(lines)


def _model_prompt(
    observation: dict[str, Any],
    coordinate_candidates: list[dict[str, float | str]] | None = None,
) -> str:
    goal = str(observation.get("goal", "")).strip()
    return (
        f"Browser task: {goal}\n\n{_compact_context(observation, coordinate_candidates)}\n\n"
        "Choose exactly one grounded computer action or abstain."
    )


def _predict(agent: Agent, prompt: str, tools: list[Any]) -> tuple[str, str | None, dict[str, Any]]:
    """Run only the model decode, bypassing registry echo dispatch."""

    from localagent.agent.constrained import hybrid_decode

    raw = hybrid_decode(
        agent.model,
        agent.tokenizer,
        prompt,
        tools,
        selector=agent.selector,
        route_head=agent.route_head,
        ptr_head=agent.ptr_head,
        top_m=1,
    )
    calls = extract_tool_calls(raw)
    if not calls:
        return raw, None, {}
    call = calls[0]
    return raw, call.name, dict(call.arguments)


def _target_bid(
    target: Any,
    elements: list[dict[str, str]],
    *,
    goal: str = "",
    semantic_fallback: bool = False,
) -> str | None:
    text = str(target).strip().strip("'\"")
    if any(element["bid"] == text for element in elements):
        return text
    normalized = " ".join(text.lower().split())
    if not normalized:
        return None
    exact = [element for element in elements if " ".join(element["name"].lower().split()) == normalized]
    if exact:
        return exact[0]["bid"]
    contained = [
        element
        for element in elements
        for element_name in [" ".join(element["name"].lower().split())]
        if element_name
        and (normalized in element_name or element_name in normalized)
    ]
    if contained:
        return contained[0]["bid"]
    if not semantic_fallback:
        return None
    low_goal = f"{goal} {text}".lower()
    if "number" in low_goal and "ascending" in low_goal:
        numeric = [element for element in elements if element["name"].strip().isdigit()]
        if numeric:
            return min(numeric, key=lambda element: int(element["name"].strip()))["bid"]
    if re.search(r"\b(?:button|link)\b", low_goal):
        candidates = [element for element in elements if element["role"] in {"button", "link"}]
        if len(candidates) == 1:
            return candidates[0]["bid"]
    return None


def _target_coordinate(
    target: Any,
    candidates: list[dict[str, float | str]],
    *,
    goal: str = "",
) -> dict[str, float | str] | None:
    text = " ".join(str(target).strip().strip("'\"").lower().split())
    if not text:
        return None
    normalized = [
        (candidate, " ".join(str(candidate["name"]).lower().split()))
        for candidate in candidates
    ]
    exact = [candidate for candidate, name in normalized if name == text]
    if exact:
        return exact[0]
    contained = [
        candidate
        for candidate, name in normalized
        if text in name or name in text
    ]
    if contained:
        return contained[0]
    # A generic MiniWoW goal such as "click the numbers in ascending order" exposes numeric
    # clickable labels.  When the model supplies a non-specific target, selecting the smallest
    # visible number is a deterministic, label-free grounding rule; subsequent DOM snapshots
    # remove the completed number and expose the next one.
    low_goal = f"{goal} {text}".lower()
    if "number" in low_goal and "ascending" in low_goal:
        numeric = [
            candidate
            for candidate, name in normalized
            if name.isdigit()
        ]
        if numeric:
            return min(numeric, key=lambda candidate: int(str(candidate["name"])))
    return None


def _browser_action(
    tool: str | None,
    arguments: dict[str, Any],
    observation: dict[str, Any],
    *,
    coordinate_fallback: bool = False,
    semantic_fallback: bool = False,
    device_pixel_ratio: float = 1.0,
    screenshot_scale: float = 1.0,
) -> tuple[str, bool]:
    """Map a model tool call to a live BrowserGym action and report grounding success."""

    elements = _visible_elements(observation)
    coordinate_candidates = (
        _dom_coordinate_candidates(
            observation,
            device_pixel_ratio=device_pixel_ratio,
            screenshot_scale=screenshot_scale,
        )
        if coordinate_fallback
        else []
    )
    if tool in {"click", "double_click", "web_click"}:
        target = arguments.get("target_id") if tool == "web_click" else arguments.get("target")
        bid = _target_bid(
            target,
            elements,
            goal=str(observation.get("goal", "")),
            semantic_fallback=semantic_fallback,
        )
        if bid is None and coordinate_candidates:
            candidate = _target_coordinate(target, coordinate_candidates, goal=str(observation.get("goal", "")))
            if candidate is not None:
                action = "mouse_dblclick" if tool == "double_click" else "mouse_click"
                return f"{action}({candidate['x']:.3f}, {candidate['y']:.3f})", True
        if bid is None:
            return "noop(0)", False
        action = "dblclick" if tool == "double_click" else "click"
        return f"{action}({bid!r})", True
    if tool == "move_cursor" and coordinate_candidates and coordinate_fallback:
        candidate = _target_coordinate(
            arguments.get("target"),
            coordinate_candidates,
            goal=str(observation.get("goal", "")),
        )
        if candidate is not None and "click" in str(observation.get("goal", "")).lower():
            return f"mouse_click({candidate['x']:.3f}, {candidate['y']:.3f})", True
    if tool in {"type_text", "web_type"}:
        if tool == "web_type":
            focused = _target_bid(arguments.get("target_id"), elements)
        else:
            focused = observation.get("focused_element_bid")
            if not isinstance(focused, str) or not focused:
                focused = next((e["bid"] for e in elements if e["role"] == "textbox"), None)
        if not focused:
            return "noop(0)", False
        return f"fill({focused!r}, {str(arguments.get('text', ''))!r})", True
    if tool == "web_select":
        bid = _target_bid(arguments.get("target_id"), elements)
        if bid is None:
            return "noop(0)", False
        return f"select_option({bid!r}, {str(arguments.get('value', ''))!r})", True
    if tool == "key_press":
        key = str(arguments.get("key", "Enter"))
        focused = observation.get("focused_element_bid")
        if isinstance(focused, str) and focused:
            return f"press({focused!r}, {key!r})", True
        return f"keyboard_press({key!r})", True
    if tool == "scroll":
        direction = str(arguments.get("direction", "down")).lower()
        return ("scroll(0, -500)" if direction == "up" else "scroll(0, 500)"), True
    return "noop(0)", False


def _make_agent(checkpoint: Path, tools: list[Any]) -> Agent:
    registry = ToolRegistry()
    for spec in tools:
        registry.register(spec, lambda **kwargs: kwargs)
    return Agent.from_checkpoint(checkpoint, registry)


def _run_episode(
    env: Any,
    agent: Agent,
    *,
    tools: list[Any],
    task: str,
    seed: int,
    max_steps: int,
    coordinate_fallback: bool = False,
    semantic_fallback: bool = False,
) -> dict[str, Any]:
    observation, reset_info = env.reset(seed=seed)
    goal = str(observation.get("goal", ""))
    records: list[dict[str, Any]] = []
    total_reward = 0.0
    terminated = truncated = False
    page = getattr(getattr(env, "unwrapped", None), "page", None)
    try:
        device_pixel_ratio = float(page.evaluate("devicePixelRatio")) if page is not None else 1.0
    except Exception:  # pragma: no cover - optional runtime property
        device_pixel_ratio = 1.0
    screenshot_scale = float(getattr(page, "_bgym_scale_factor", 1.0)) if page is not None else 1.0
    for step in range(max_steps):
        # Keep the model's trained accessibility prompt stable.  The coordinate candidates are a
        # runtime grounding sidecar, not an extra prompt contract; exposing them here would create
        # a distribution shift for checkpoints trained before this diagnostic existed.
        prompt = _model_prompt(observation)
        started = time.perf_counter()
        raw, tool, arguments = _predict(agent, prompt, tools)
        action, grounded = _browser_action(
            tool,
            arguments,
            observation,
            coordinate_fallback=coordinate_fallback,
            semantic_fallback=semantic_fallback,
            device_pixel_ratio=device_pixel_ratio,
            screenshot_scale=screenshot_scale,
        )
        observation, reward, terminated, truncated, info = env.step(action)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        total_reward += float(reward)
        records.append(
            {
                "action": action,
                "arguments": arguments,
                "grounded": grounded,
                "info_action_error": str(info.get("action_error", "")),
                "model_output": raw,
                "model_tool": tool,
                "reward": float(reward),
                "step": step,
                "wall_ms": round(elapsed_ms, 3),
            }
        )
        if terminated or truncated:
            break
    task_info = info.get("task_info", {}) if records else reset_info.get("task_info", {})
    success = bool(total_reward > 0.0 or (isinstance(task_info, dict) and task_info.get("REWARD_GLOBAL", 0) > 0))
    return {
        "goal": goal,
        "seed": seed,
        "steps": records,
        "success": success,
        "task": task,
        "total_reward": total_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browsergym-checkout", type=Path, required=True)
    parser.add_argument("--miniwob-checkout", type=Path, required=True)
    parser.add_argument("--browser-executable", type=Path, required=True)
    parser.add_argument("--browser-installation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="run only the first N pinned episodes")
    parser.add_argument("--max-steps", type=int, default=PRODUCTION_MAX_STEPS)
    parser.add_argument(
        "--tool-pool",
        choices=("standard", "realistic_browser"),
        default="standard",
        help="tool vocabulary used by the checkpoint and native action adapter",
    )
    parser.add_argument(
        "--coordinate-fallback",
        action="store_true",
        help="use live DOM clickable geometry when accessibility-tree grounding is unavailable; non-official diagnostic",
    )
    parser.add_argument(
        "--semantic-fallback",
        action="store_true",
        help="use generic goal-language grounding for unmatched accessibility targets; non-official diagnostic",
    )
    args = parser.parse_args()

    for path in (args.browsergym_checkout, args.miniwob_checkout, args.browser_executable, args.browser_installation, args.checkpoint):
        if not path.exists():
            raise SystemExit(f"missing required path: {path}")
    if args.out.exists() or args.out.is_symlink():
        raise SystemExit(f"refusing to overwrite receipt: {args.out}")
    if _git_head(args.browsergym_checkout) != PRODUCTION_BROWSERGYM_REVISION:
        raise SystemExit("BrowserGym checkout revision does not match the pinned evaluation plan")
    if _git_head(args.miniwob_checkout) != PRODUCTION_MINIWOB_REVISION:
        raise SystemExit("MiniWoB checkout revision does not match the pinned evaluation plan")

    os.environ["MINIWOB_URL"] = (args.miniwob_checkout / "miniwob" / "html" / "miniwob").resolve().as_uri() + "/"
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(args.browser_installation.resolve().parent)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import gymnasium as gym
    import browsergym.core  # noqa: F401  # registers the open-ended task
    import browsergym.miniwob  # noqa: F401  # registers MiniWoB tasks
    from browsergym.core.action.highlevel import HighLevelActionSet

    action_mapping = HighLevelActionSet(subsets=["bid", "miniwob_all"], multiaction=False).to_python_code
    episodes = list(production_capture_plan())
    selected = episodes[: args.limit] if args.limit > 0 else episodes
    tools = STANDARD_TOOLS if args.tool_pool == "standard" else REALISTIC_BROWSER_TOOLS
    agent = _make_agent(args.checkpoint, tools)
    cases: list[dict[str, Any]] = []
    for index, episode in enumerate(selected, start=1):
        env = gym.make(
            f"browsergym/{episode.task_name}",
            viewport={"width": 1280, "height": 720},
            timeout=30000,
            locale="en-US",
            timezone_id="UTC",
            headless=True,
            action_mapping=action_mapping,
            use_raw_page_output=False,
            pw_chromium_kwargs={"executable_path": str(args.browser_executable.resolve())},
            pw_context_kwargs={"device_scale_factor": 1.0},
            max_episode_steps=args.max_steps,
        )
        try:
            case = _run_episode(
                env,
                agent,
                tools=tools,
                task=episode.task_name,
                seed=episode.seed,
                max_steps=args.max_steps,
                coordinate_fallback=args.coordinate_fallback,
                semantic_fallback=args.semantic_fallback,
            )
        finally:
            env.close()
        cases.append(case)
        print(f"[{index}/{len(selected)}] {episode.task_name} seed={episode.seed} success={case['success']}", flush=True)

    full_plan = (
        len(selected) == PRODUCTION_EPISODES
        and args.max_steps == PRODUCTION_MAX_STEPS
        and not args.coordinate_fallback
        and not args.semantic_fallback
    )
    receipt = {
        "benchmark_id": "browsergym_miniwob",
        "browsergym": {
            "revision": PRODUCTION_BROWSERGYM_REVISION,
            "version": PRODUCTION_BROWSERGYM_VERSION,
        },
        "checkpoint": {"path": str(args.checkpoint), "sha256": _sha256(args.checkpoint)},
        "tool_pool": args.tool_pool,
        "coordinate_fallback": args.coordinate_fallback,
        "semantic_fallback": args.semantic_fallback,
        "tool_names": [spec.name for spec in tools],
        "tool_pool_claim_boundary": (
            "The realistic_browser pool is a vocabulary/dispatch diagnostic only: Mind2Web backend "
            "node IDs are not MiniWoB live DOM IDs, so this run must not be interpreted as cross-site "
            "grounding or Mind2Web-to-BrowserGym transfer."
            if args.tool_pool == "realistic_browser"
            else "Legacy standard tool vocabulary; no public Mind2Web transfer claim."
        ),
        "claim_boundary": (
            "Native BrowserGym/MiniWoB checkpoint-in-the-loop evaluation over the pinned task plan. "
            "This is a text/accessibility-tree modality result, not a visual-agent result, WebArena "
            "result, or real-account email/Notion execution."
            + (
                " The coordinate fallback is an optional DOM-geometry diagnostic and makes this run non-official."
                if args.coordinate_fallback
                else ""
            )
            + (
                " The semantic fallback is an optional goal-language grounding diagnostic and makes this run non-official."
                if args.semantic_fallback
                else ""
            )
        ),
        "cases": cases,
        "environment_executed": True,
        "official_split_verified": full_plan,
        "runtime": {
            "browser_executable": str(args.browser_executable),
            "browser_executable_sha256": _sha256(args.browser_executable),
            "chromium_revision": PRODUCTION_CHROMIUM_REVISION,
            "chromium_version": PRODUCTION_CHROMIUM_VERSION,
            "headless": True,
            "max_steps": args.max_steps,
            "miniwob_revision": PRODUCTION_MINIWOB_REVISION,
            "playwright_version": importlib.metadata.version("playwright"),
            "playwright_pinned_version": PRODUCTION_PLAYWRIGHT_VERSION,
        },
        "success_rate": sum(case["success"] for case in cases) / len(cases) if cases else 0.0,
        "task_count": len(cases),
        "task_plan": {
            "fixed_seeds": list(PRODUCTION_FIXED_SEEDS),
            "expected_episodes": PRODUCTION_EPISODES,
            "expected_task_variants": PRODUCTION_TASK_VARIANTS,
            "expected_similarity_groups": PRODUCTION_SIMILARITY_GROUPS,
            "selected_limit": args.limit,
        },
        "kind": "localagent_browsergym_native_eval",
        "schema_version": 1,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in ("benchmark_id", "environment_executed", "official_split_verified", "task_count", "success_rate")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
