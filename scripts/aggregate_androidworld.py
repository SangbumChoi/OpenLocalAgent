#!/usr/bin/env python
"""Aggregate a real AndroidWorld run_... checkpointer directory.

AndroidWorld writes gzip-compressed pickle task groups. This command only parses existing output;
it never starts an emulator or invokes adb. The default safe loader handles builtin-only metadata
fixtures. For a trusted local AndroidWorld run containing screenshots and other objects, pass
--allow-unsafe-pickle explicitly and keep the resulting receipt private until the source directory
has been audited.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localagent.eval.androidworld import aggregate_androidworld_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="AndroidWorld run_... directory")
    parser.add_argument(
        "--expected-task",
        action="append",
        dest="expected_tasks",
        help="expected task template; repeat once per task",
    )
    parser.add_argument(
        "--n-task-combinations",
        type=int,
        help="expected number of instances for every task template",
    )
    parser.add_argument("--source-revision", help="pinned AndroidWorld source revision")
    parser.add_argument("--agent-name", help="agent identifier recorded by the runner")
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help="acknowledge that trusted upstream pickles can execute code while loading",
    )
    parser.add_argument("--output", required=True, type=Path, help="receipt JSON path")
    args = parser.parse_args()

    receipt = aggregate_androidworld_run(
        args.run_dir,
        expected_tasks=args.expected_tasks,
        n_task_combinations=args.n_task_combinations,
        source_revision=args.source_revision,
        agent_name=args.agent_name,
        allow_unsafe_pickle=args.allow_unsafe_pickle,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "overall": receipt["overall"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
