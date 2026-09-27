#!/usr/bin/env python3
"""Refresh the current m679 gate with the complete bounded Playwright service subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _self_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def assemble(*, prior: Path, checkpoint: Path, browser: Path, output: Path) -> dict[str, Any]:
    gate = json.loads(prior.read_text())
    receipt = json.loads(browser.read_text())
    current = _identity(checkpoint)
    if gate.get("current_checkpoint", {}).get("sha256") != current["sha256"]:
        raise ValueError("prior gate checkpoint mismatch")
    if receipt.get("checkpoint_sha256") != current["sha256"]:
        raise ValueError("browser receipt checkpoint mismatch")
    blocked = dict(gate.get("blocked_requirements", {}))
    blocked["native:mcpmark_playwright"] = "bounded_subset_and_verifier_zero"
    payload: dict[str, Any] = {
        "kind": "localagent_m703_workshop_gate_current_m679",
        "schema_version": 1,
        "ready": False,
        "current_checkpoint": current,
        "passed_requirements": sorted(set(gate.get("passed_requirements", []))),
        "blocked_requirements": dict(sorted(blocked.items())),
        "evidence": {**gate.get("evidence", {}), "mcpmark_playwright": _identity(browser)},
        "claim_boundary": (
            "Fail-closed gate after all four pinned MCPMark Playwright standard tasks were "
            "exercised through a real MCP browser server. All verifiers remain zero; full official "
            "split parity, visual grounding, user simulation, external accounts, and current "
            "public artifact binding remain incomplete."
        ),
        "source_gate": _identity(prior),
    }
    payload["receipt_self_sha256"] = _self_hash(payload)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--browser", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = assemble(prior=args.prior, checkpoint=args.checkpoint, browser=args.browser, output=args.output)
    print(json.dumps({"output": str(args.output), "ready": payload["ready"], "blockers": len(payload["blocked_requirements"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
