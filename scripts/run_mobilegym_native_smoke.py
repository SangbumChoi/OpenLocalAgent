#!/usr/bin/env python3
"""Run a bounded MobileGym browser/runtime smoke without invoking a model or task side effect.

The smoke verifies the pinned source registry, the public browser entrypoint, and MobileGym's
documented ``window.__SIM__`` state/reset bridge.  It records only hashes, shapes, counts, and
volatile-field paths; state values, screenshots, task prompts, and benchmark content are not
written to the receipt.  This is a runtime preflight, not a MobileGym-Bench score.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError, sync_playwright


MOBILEGYM_REVISION = "093a3292d13fc4186e279af4ef1b005ac8e4d2b7"
MOBILEGYM_URL = "https://github.com/Purewhiter/mobilegym"


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": size, "sha256": digest.hexdigest()}


def _state_digest(state: Any) -> dict[str, Any]:
    encoded = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "bytes": len(encoded.encode("utf-8")),
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "top_level_keys": sorted(state) if isinstance(state, dict) else [],
    }


def _shape(value: Any) -> str:
    if isinstance(value, list):
        return f"array:{len(value)}"
    if isinstance(value, dict):
        return f"object:{len(value)}"
    return type(value).__name__


def _diff_paths(first: Any, second: Any, *, limit: int = 20) -> list[dict[str, str]]:
    diffs: list[dict[str, str]] = []

    def walk(left: Any, right: Any, path: str) -> None:
        if len(diffs) >= limit:
            return
        if type(left) is not type(right) or isinstance(left, list) != isinstance(right, list):
            diffs.append({"path": path, "kind": "type", "left": _shape(left), "right": _shape(right)})
            return
        if isinstance(left, dict):
            for key in sorted(set(left) | set(right)):
                walk(left.get(key), right.get(key), f"{path}/{key}")
        elif isinstance(left, list):
            if len(left) != len(right):
                diffs.append({"path": path, "kind": "length", "left": str(len(left)), "right": str(len(right))})
            for index, (item_left, item_right) in enumerate(zip(left, right)):
                walk(item_left, item_right, f"{path}/{index}")
        elif left != right:
            diffs.append({"path": path, "kind": "value", "left": _shape(left), "right": _shape(right)})

    walk(first, second, "")
    return diffs


def _reset_and_snapshot(page: Page) -> Any:
    try:
        with page.expect_navigation(wait_until="networkidle", timeout=10_000):
            page.evaluate("window.__SIM__.reset()")
    except PlaywrightTimeoutError:
        # Some simulator revisions reset stores without a full navigation.
        pass
    page.wait_for_timeout(500)
    return page.evaluate("window.__SIM__.getState()")


def _registry_profile(source_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(source_root))
    try:
        registry = importlib.import_module("bench_env.task")
        splits = importlib.import_module("bench_env.splits")
        tasks = registry.load_tasks()
        task_ids = {task.id for task in tasks}
        train = splits.resolve_split("train")
        test = splits.resolve_split("test")
    finally:
        sys.path.pop(0)
    return {
        "loaded_tasks": len(tasks),
        "loaded_task_ids_unique": len(task_ids) == len(tasks),
        "train_tasks": len(train),
        "test_tasks": len(test),
        "train_missing": sorted(train - task_ids),
        "test_missing": sorted(test - task_ids),
        "train_test_overlap": sorted(train & test),
    }


def run_smoke(
    *,
    env_url: str,
    source_root: Path,
    archive: Path | None = None,
    revision: str = MOBILEGYM_REVISION,
) -> dict[str, Any]:
    registry = _registry_profile(source_root)
    console_events: list[dict[str, str]] = []
    page_errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 360, "height": 800}, device_scale_factor=3)
        page.on("console", lambda message: console_events.append({"type": message.type, "text": message.text}))
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        response = page.goto(env_url, wait_until="networkidle", timeout=30_000)
        page.wait_for_timeout(1_000)
        initial = page.evaluate("window.__SIM__.getState()")
        first_reset = _reset_and_snapshot(page)
        second_reset = _reset_and_snapshot(page)
        body_text = page.locator("body").inner_text()
        title = page.title()
        url = page.url
        browser.close()

    error_classes = []
    for event in console_events:
        if event["type"] != "error":
            continue
        text = event["text"]
        if "404" in text:
            error_classes.append("http_404")
        elif "WMR" in text:
            error_classes.append("wmr_widget")
        else:
            error_classes.append("other_console_error")

    payload: dict[str, Any] = {
        "kind": "localagent_mobilegym_native_runtime_smoke",
        "schema_version": 1,
        "source": {
            "repository": MOBILEGYM_URL,
            "revision": revision,
            "source_root": str(source_root.resolve()),
            "archive": _identity(archive) if archive is not None else None,
        },
        "registry": registry,
        "runtime": {
            "env_url": env_url,
            "http_status": response.status if response is not None else None,
            "page_title": title,
            "final_url": url,
            "sim_bridge": True,
            "state_initial": _state_digest(initial),
            "state_after_reset": _state_digest(first_reset),
            "state_after_second_reset": _state_digest(second_reset),
            "reset_hash_equal": _state_digest(first_reset)["sha256"] == _state_digest(second_reset)["sha256"],
            "reset_diff_paths": _diff_paths(first_reset, second_reset),
            "body_bytes": len(body_text.encode("utf-8")),
            "body_sha256": hashlib.sha256(body_text.encode("utf-8")).hexdigest(),
            "visible_app_markers": sum(marker in body_text for marker in ("计算器", "笔记", "微信", "设置", "地图")),
            "console_error_classes": sorted(set(error_classes)),
            "console_error_count": len(error_classes),
            "page_error_count": len(page_errors),
        },
        "localagent_adaptation": {
            "model_invocations": 0,
            "task_episodes": 0,
            "training_rows_added": 0,
            "official_score": None,
            "claim_boundary": "Local browser/runtime preflight only; no model action, task judge, screenshot score, or benchmark result was produced.",
        },
    }
    payload["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-url", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--revision", default=MOBILEGYM_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    receipt = run_smoke(
        env_url=args.env_url,
        source_root=args.source_root,
        archive=args.archive,
        revision=args.revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
