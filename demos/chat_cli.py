"""Terminal agent demo (Phase 7): a REPL over agent.runtime.Agent.

Shows the live tool-call / tool-response trace and current memory state inline.
Run: python demos/chat_cli.py results/runs/sft/latest.pt
"""

from __future__ import annotations

import sys


def main(checkpoint: str) -> None:
    raise NotImplementedError(
        "TODO(phase-7): load ckpt + tokenizer + default_registry -> Agent -> input() loop"
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/runs/sft/latest.pt")
