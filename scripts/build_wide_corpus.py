#!/usr/bin/env python
"""Widen the student's catalog exposure by adding xLAM's own training split.

The student's weakness is open-catalog function calling: it has never met most of the 176–197
tools the evaluation catalogs contain, and 16,300 rows drawn from four sources do not cover them.
xLAM's train split is thousands of distinct APIs with gold calls, which is exactly the missing
axis. Rows are checked against the evaluation shards by rendered-prompt hash — the same identity
the trainer audits with — so widening cannot become leaking.

  python scripts/build_wide_corpus.py --out data/wide/train.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

from openlocalagent.data.conversation_artifact import (rendered_assistant_prompts,
                                                   rendered_prompt_sha256)
from openlocalagent.data.prompt_contract import OPENAI_FULL_CATALOG_V1
from openlocalagent.data.render import render_conversation_rows
from openlocalagent.model.tokenizer import load_tokenizer
from openlocalagent.train.stage_data import read_conversations


def fits(conversation, tok, max_seq_len: int) -> bool:
    try:
        render_conversation_rows(conversation, tok, prompt_contract=OPENAI_FULL_CATALOG_V1,
                                 max_seq_len=max_seq_len)
    except (ValueError, KeyError):
        return False
    return True


def prompt_hashes(conversation) -> set[str] | None:
    """None marks a row the contract refuses to render — undedupable and untrainable alike."""
    try:
        return {rendered_prompt_sha256(prompt) for prompt in rendered_assistant_prompts(
            conversation, conversation_prompt_contract=OPENAI_FULL_CATALOG_V1)}
    except (ValueError, KeyError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/merged-v2/train.jsonl")
    ap.add_argument("--add", default="data/public/xlam-train.jsonl")
    ap.add_argument("--guard", action="append",
                    default=["data/public/xlam-test.jsonl", "data/public/toolace-eval.jsonl"])
    ap.add_argument("--cap", type=int, default=18000)
    ap.add_argument("--out", default="data/wide/train.jsonl")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--tokenizer", default="data/tokenizer-h100-16k.json")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    tok = load_tokenizer("bpe", args.tokenizer)
    forbidden: set[str] = set()
    for guard in args.guard:
        path = Path(guard)
        if path.exists():
            for conversation in read_conversations(path):
                forbidden |= prompt_hashes(conversation) or set()
    print(f"guarded prompts: {len(forbidden)}", flush=True)

    base = read_conversations(Path(args.base))
    extra = read_conversations(Path(args.add))
    random.Random(args.seed).shuffle(extra)

    kept, dropped_overlap, dropped_size = [], 0, 0
    for conversation in extra:
        if len(kept) >= args.cap:
            break
        hashes = prompt_hashes(conversation)
        if hashes is None:
            dropped_size += 1
            continue
        if hashes & forbidden:
            dropped_overlap += 1
            continue
        if not fits(conversation, tok, args.max_seq_len):
            dropped_size += 1
            continue
        kept.append(conversation)

    rows = list(base) + kept
    random.Random(args.seed).shuffle(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), default=str, sort_keys=True) + "\n")
    payload = out.read_bytes()
    manifest = {"base": args.base, "base_rows": len(base), "added": args.add,
                "added_rows": len(kept), "dropped_overlapping_prompts": dropped_overlap,
                "dropped_oversize": dropped_size, "total_rows": len(rows),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "guards": args.guard}
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print("WIDE_CORPUS_DONE " + str(out), flush=True)


if __name__ == "__main__":
    main()
