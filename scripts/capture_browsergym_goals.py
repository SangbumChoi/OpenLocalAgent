#!/usr/bin/env python
"""Run the controlled 240-reset BrowserGym/MiniWoB goal capture.

This command performs real browser initialization for the exact 60-task x four-seed production
plan. It reads only ``observation["goal"]`` from each reset, takes no actions, and emits a
self-hashed receipt. Both source repositories and the Playwright Chromium installation must
already exist locally at the pinned revisions; this command never downloads dependencies.

Example:
  python scripts/capture_browsergym_goals.py \
    --browsergym-checkout /path/to/BrowserGym \
    --miniwob-checkout /path/to/miniwob-plusplus \
    --browser-executable /path/to/ms-playwright/chromium-1117/.../Chromium \
    --browser-installation /path/to/ms-playwright/chromium-1117 \
    --runtime-manifest configs/data/browsergym-capture-runtime-darwin-arm64-py312.json \
    --capture data/private/browsergym-miniwob-reset-goals.jsonl \
    --receipt data/private/browsergym-miniwob-reset-goals.receipt.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localagent.data.browsergym_capture import capture_browsergym_goals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--browsergym-checkout",
        type=Path,
        required=True,
        help=(
            "pristine BrowserGym git top-level at the pinned revision "
            "(no untracked, ignored, or special-index files)"
        ),
    )
    parser.add_argument(
        "--miniwob-checkout",
        type=Path,
        required=True,
        help=(
            "pristine MiniWoB++ git top-level at the pinned revision "
            "(no untracked, ignored, or special-index files)"
        ),
    )
    parser.add_argument(
        "--browser-executable",
        type=Path,
        required=True,
        help="Playwright Chromium executable inside --browser-installation",
    )
    parser.add_argument(
        "--browser-installation",
        type=Path,
        required=True,
        help="Playwright installation directory named chromium-1117",
    )
    parser.add_argument(
        "--runtime-manifest",
        type=Path,
        required=True,
        help="frozen capture-environment manifest matching the active Python environment",
    )
    parser.add_argument(
        "--capture",
        type=Path,
        required=True,
        help="new canonical reset-goal JSONL path; existing files are never overwritten",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        required=True,
        help="new self-hashed producer receipt path; existing files are never overwritten",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        receipt = capture_browsergym_goals(
            args.capture,
            args.receipt,
            browsergym_checkout=args.browsergym_checkout,
            miniwob_checkout=args.miniwob_checkout,
            browser_executable=args.browser_executable,
            browser_installation=args.browser_installation,
            environment_manifest=args.runtime_manifest,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"BrowserGym controlled capture failed: {error}") from error
    print(
        json.dumps(
            {
                "capture": receipt["capture"],
                "receipt_self_sha256": receipt["receipt_self_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
