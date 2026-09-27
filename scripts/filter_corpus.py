#!/usr/bin/env python
"""Drop training rows whose rendered prompts collide with a held-out conversation file.

The trainer's own overlap assertion is the authority; this applies the same hash test ahead of
time so a corpus passes it by construction. Relabelling changes assistant targets, never prompts,
so filtering the relabelled corpus is equivalent to filtering before relabelling.

  python scripts/filter_corpus.py --corpus data/distill2/train.jsonl \
      --held-out data/merged-v2/eval.jsonl --out data/distill2/train-clean.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from localagent.data.conversation_artifact import (rendered_assistant_prompts,
                                                   rendered_prompt_sha256)
from localagent.data.prompt_contract import OPENAI_FULL_CATALOG_V1
from localagent.train.stage_data import read_conversations


def prompt_hashes(conversation) -> set[str]:
    try:
        return {rendered_prompt_sha256(prompt) for prompt in rendered_assistant_prompts(
            conversation, conversation_prompt_contract=OPENAI_FULL_CATALOG_V1)}
    except (ValueError, KeyError):
        return set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--held-out", action="append", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    forbidden: set[str] = set()
    for path in args.held_out:
        for conversation in read_conversations(Path(path)):
            forbidden |= prompt_hashes(conversation)
    print(f"held-out prompts: {len(forbidden)}", flush=True)

    kept = dropped = 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for conversation in read_conversations(Path(args.corpus)):
            if prompt_hashes(conversation) & forbidden:
                dropped += 1
                continue
            handle.write(json.dumps(asdict(conversation), default=str, sort_keys=True) + "\n")
            kept += 1
    print(f"kept={kept} dropped={dropped}")
    print("FILTER_DONE " + str(out), flush=True)


if __name__ == "__main__":
    main()
