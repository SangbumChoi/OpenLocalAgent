#!/usr/bin/env python3
"""Build and verify the sealed paper confirmatory agent-evaluation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localagent.eval.confirmatory_eval_split import build_confirmatory_eval_split


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path("configs/eval/paper-confirmatory-eval-split-v2.yaml"),
        help="strict confirmatory-split config",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = build_confirmatory_eval_split(args.config)
    summary = {
        "assistant_decisions": receipt["filtered_selection"]["selected"]["assistant_decisions"],
        "assistant_loss_tokens": receipt["token_accounting"]["assistant_loss_tokens"],
        "filtered_selection_audit_sha256": receipt["filtered_selection"]["audit_sha256"],
        "manifest_sha256": receipt["output"]["manifest"]["sha256"],
        "output_sha256": receipt["output"]["jsonl"]["sha256"],
        "receipt_self_sha256": receipt["receipt_self_sha256"],
        "reference_contract_sha256": receipt["reference_contract_sha256"],
        "rows": receipt["output"]["rows"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
