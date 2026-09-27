#!/usr/bin/env python
"""Verify or stage a generated LocalAgent WebGPU demo bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

from localagent.eval.demo_deploy import (
    sync_demo_bundle,
    verify_demo_deploy,
    write_demo_deploy_receipt,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", default="spaces/localagent-webgpu")
    parser.add_argument("--bundle-dir", help="generated export directory; defaults to --demo-dir")
    parser.add_argument("--sync", action="store_true", help="copy a verified bundle beside the app")
    parser.add_argument("--out", help="write a new JSON receipt (refuses overwrite)")
    parser.add_argument(
        "--checkpoint",
        help="checkpoint whose SHA-256 must be bound by bundle-manifest.json",
    )
    parser.add_argument(
        "--expected-tool-count",
        type=int,
        help="expected number of named tools in meta.json",
    )
    args = parser.parse_args()

    checkpoint_sha256 = None
    if args.checkpoint:
        digest = hashlib.sha256()
        with open(args.checkpoint, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        checkpoint_sha256 = digest.hexdigest()

    if args.sync:
        if not args.bundle_dir:
            parser.error("--sync requires --bundle-dir")
        report = sync_demo_bundle(
            args.bundle_dir,
            args.demo_dir,
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_tool_count=args.expected_tool_count,
        )
    else:
        report = verify_demo_deploy(
            args.demo_dir,
            bundle_dir=args.bundle_dir,
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_tool_count=args.expected_tool_count,
        )
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.out:
        write_demo_deploy_receipt(report, args.out)
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
