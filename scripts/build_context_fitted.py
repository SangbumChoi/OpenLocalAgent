#!/usr/bin/env python
"""Drop conversations whose rendered rows do not fit a context budget.

`openai_full_catalog_v1` renders the whole tool catalog into the prompt and refuses to truncate,
because a row missing part of its own catalog is a row whose gold answer is unreachable. That is
the right behaviour, and it means a single oversized conversation aborts the training stage:

    ValueError: openai_full_catalog_v1 row exceeds max_seq_len and cannot be truncated:
    assistant_message_index=1, tokens=6502, max_seq_len=2048

Measured over the posttrain corpora, 99.9%+ of rows already fit 2048 tokens, so filtering costs
almost nothing and one filtered set serves every tier. This writes the rows that fit and reports
exactly what it dropped, so the cost is visible rather than assumed.

  python scripts/build_context_fitted.py --in data/public/toolace-train.jsonl \
      --out data/public/toolace-train-fit2048.jsonl --max-tokens 2048
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

from openlocalagent.data.prompt_contract import assistant_training_examples
from openlocalagent.model.tokenizer import load_tokenizer
from openlocalagent.train.stage_data import read_conversations


@dataclass(frozen=True)
class FilterReport:
    """What a filtering pass kept and what it cost."""

    kept: int
    dropped_oversize: int
    dropped_invalid: int
    longest_kept: int
    longest_dropped: int

    @property
    def total(self) -> int:
        return self.kept + self.dropped_oversize + self.dropped_invalid


def filter_file(source: str, target: str, tokenizer_path: str, max_tokens: int) -> FilterReport:
    tokenizer = load_tokenizer("bpe", tokenizer_path)
    kept = oversize = invalid = 0
    longest_kept = longest_dropped = 0

    with open(source, encoding="utf-8") as handle, open(target, "w", encoding="utf-8") as out:
        for raw, conversation in zip(handle, read_conversations(source)):
            try:
                examples = assistant_training_examples(conversation)
            except Exception:
                invalid += 1
                continue
            longest = max(
                (len(tokenizer.encode(e.prompt)) + len(tokenizer.encode(e.body)) for e in examples),
                default=0,
            )
            if longest > max_tokens:
                oversize += 1
                longest_dropped = max(longest_dropped, longest)
                continue
            longest_kept = max(longest_kept, longest)
            out.write(raw if raw.endswith("\n") else raw + "\n")
            kept += 1

    return FilterReport(kept, oversize, invalid, longest_kept, longest_dropped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="source", required=True)
    parser.add_argument("--out", dest="target", required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--tokenizer", default="data/tokenizer-h100-16k.json")
    args = parser.parse_args()

    report = filter_file(args.source, args.target, args.tokenizer, args.max_tokens)
    share = report.kept / max(report.total, 1) * 100
    print(
        f"{args.source}: kept {report.kept:,}/{report.total:,} ({share:.2f}%) "
        f"-> {args.target}\n"
        f"  dropped {report.dropped_oversize:,} over {args.max_tokens} tokens "
        f"(longest {report.longest_dropped:,}), {report.dropped_invalid:,} contract-invalid; "
        f"longest kept {report.longest_kept:,}"
    )
    print("FILTER_JSON " + json.dumps(asdict(report)))


if __name__ == "__main__":
    main()
