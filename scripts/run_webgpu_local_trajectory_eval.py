#!/usr/bin/env python3
"""Run the local browser/email/Notion WebGPU trajectory fixture and seal its receipt.

The page owns the resettable in-memory state machine and independent transition checks.  This
wrapper only drives a real Chromium page, binds the exact local bundle, and records the result;
it never contacts an external account or claims AndroidWorld, BrowserGym, MCP, or real-productivity
success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": size, "sha256": digest.hexdigest()}


def _self_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def run(
    *,
    url: str,
    bundle_dir: Path,
    timeout_ms: int,
    headless: bool,
    enable_unsafe_webgpu: bool,
) -> dict[str, Any]:
    console_events: list[dict[str, str]] = []
    page_errors: list[str] = []
    with sync_playwright() as playwright:
        launch_args = ["--use-angle=metal"]
        if enable_unsafe_webgpu:
            launch_args.append("--enable-unsafe-webgpu")
        browser = playwright.chromium.launch(headless=headless, args=launch_args)
        page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=1)
        page.on("console", lambda message: console_events.append({"type": message.type, "text": message.text}))
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        response = page.goto(url, wait_until="networkidle", timeout=timeout_ms)
        try:
            page.wait_for_function(
                "window.__localAgentMobileTrajectoryResult !== undefined",
                timeout=timeout_ms,
            )
        except PlaywrightTimeoutError:
            pass
        result = page.evaluate("window.__localAgentMobileTrajectoryResult || null")
        status = page.locator("#mobile-trajectory-progress").inner_text()
        title = page.title()
        final_url = page.url
        user_agent = page.evaluate("navigator.userAgent")
        browser.close()

    manifest = bundle_dir / "bundle-manifest.json"
    payload: dict[str, Any] = {
        "kind": "localagent_webgpu_local_trajectory_receipt",
        "schema_version": 1,
        "backend": result.get("backend") if isinstance(result, dict) else "webgpu",
        "environment_executed": isinstance(result, dict),
        "trajectory_result": result,
        "bundle_identity": {
            "manifest": _identity(manifest),
            "checkpoint_sha256": json.loads(manifest.read_text(encoding="utf-8")).get(
                "checkpoint_sha256"
            ),
        },
        "runner": {
            "url": url,
            "response_status": response.status if response is not None else None,
            "title": title,
            "final_url": final_url,
            "user_agent": user_agent,
            "status_text": status,
            "console_error_count": sum(event["type"] == "error" for event in console_events),
            "page_error_count": len(page_errors),
            "console_error_classes": sorted(
                {event["text"] for event in console_events if event["type"] == "error"}
            ),
        },
        "claim_boundary": (
            "Resettable local in-memory browser/email/Notion trajectory fixture driven by the native "
            "WebGPU bundle. No real account, external side effect, Android emulator, BrowserGym, "
            "MCP server, or official benchmark score is implied."
        ),
    }
    payload["receipt_self_sha256"] = _self_hash(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-ms", type=int, default=180_000)
    parser.add_argument("--headful", action="store_true")
    parser.add_argument("--enable-unsafe-webgpu", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit(f"refusing to overwrite {args.output}")
    receipt = run(
        url=args.url,
        bundle_dir=args.bundle_dir,
        timeout_ms=args.timeout_ms,
        headless=not args.headful,
        enable_unsafe_webgpu=args.enable_unsafe_webgpu,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["environment_executed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
