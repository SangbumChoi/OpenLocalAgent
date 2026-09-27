#!/usr/bin/env python
"""Train the 65,536-entry ByteLevel BPE, same recipe and special tokens as the 16k original.

The vocabulary is the one architecture knob the comparison keeps honest by retraining from the
same corpus pool the models pretrain on, so compression gains are attributable to size alone.

  python scripts/train_tokenizer64k.py --out data/tokenizer-h100-64k.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIALS = ["<|end|>", "<|user|>", "<|assistant|>", "<|tool|>",
            "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="+",
                    default=["data/pretrain/pool.txt"])
    ap.add_argument("--out", default="data/tokenizer-h100-64k.json")
    ap.add_argument("--vocab-size", type=int, default=65536)
    ap.add_argument("--max-lines", type=int, default=2_500_000)
    args = ap.parse_args()

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True)
    tokenizer.decoder = decoders.ByteLevel(add_prefix_space=True, trim_offsets=True)
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=True)

    def lines():
        budget = args.max_lines
        for corpus in args.corpora:
            with Path(corpus).open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if budget <= 0:
                        return
                    budget -= 1
                    yield line

    tokenizer.train_from_iterator(lines(), trainer=trainer)
    tokenizer.save(args.out)
    sample = "Call get_weather with {\"city\": \"Seoul\"} and report the result."
    print("vocab:", tokenizer.get_vocab_size(), "| sample tokens:",
          len(tokenizer.encode(sample).ids))
    print("TOKENIZER_DONE " + args.out, flush=True)


if __name__ == "__main__":
    main()
