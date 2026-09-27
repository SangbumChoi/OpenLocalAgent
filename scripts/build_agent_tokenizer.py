#!/usr/bin/env python
"""Train a 16k ByteLevel BPE that has seen the agent's own JSON.

The current tokenizer (sha 2f49a644) was trained on pretraining text alone. Measured on the
demo bundle, a get_weather tool call costs 34 tokens against 20 for LFM2's 64k vocabulary and
23 for Needle 2's 8k one: `arguments` splits into three pieces, `city` into two, and the catalog
markers `<|tool_catalog|>` are spelled out as text. Every catalog entry and every emitted call pays
that, which is why 18 tools already render to 1,544 of 2,048 context tokens.

This trains the same recipe (same size, same specials plus the two catalog markers) on the same
pretraining pool with the contract's own renderings mixed in: the function catalog of every
conversation and every assistant tool call, exactly as `prompt_contract` renders them for
training. Nothing else changes, so a ladder tier trained on it measures the tokenizer alone.

A new tokenizer is a new generation: shards must be repacked and conversation files refit with
it, and no existing checkpoint can be chained. The output path is therefore never the shared
data/tokenizer-h100-16k.json.

  python scripts/build_agent_tokenizer.py --raw data/raw/pt-4b \\
      --conversations data/public/fit2048-2f49a644-*.jsonl --out data/tokenizer-agent-16k.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402

from openlocalagent.data.prompt_contract import (  # noqa: E402
    TOOL_CATALOG_CLOSE,
    TOOL_CATALOG_OPEN,
    render_function_catalog,
    render_tool_calls,
)
from openlocalagent.data.schema import Conversation  # noqa: E402

SPECIALS = ["<|end|>", "<|user|>", "<|assistant|>", "<|tool|>",
            "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
            TOOL_CATALOG_OPEN, TOOL_CATALOG_CLOSE]

PROBES = [
    "What is the weather in Tokyo?",
    '<tool_call>{"arguments":{"city":"Tokyo","unit":"c"},"name":"get_weather"}</tool_call>',
    '{"type":"function","function":{"name":"web_search","description":"Search the web.",'
    '"parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}',
    "def f(x):\n    return x * 2",
]


def text_lines(raw: Path, per_source: int):
    """Up to `per_source` document texts from each source prefix under the raw staging dir."""
    by_source: dict[str, list[Path]] = {}
    for path in sorted(raw.glob("*.jsonl")):
        by_source.setdefault(path.name.rsplit("-", 1)[0], []).append(path)
    for source, paths in by_source.items():
        taken = 0
        for path in paths:
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if taken >= per_source:
                        break
                    try:
                        text = json.loads(line).get("text", "")
                    except json.JSONDecodeError:
                        continue
                    if text:
                        taken += 1
                        yield text
            if taken >= per_source:
                break
        print(f"  {source}: {taken} documents", file=sys.stderr)


def agent_lines(patterns: list[str], limit: int):
    """The contract's own renderings: one catalog and every tool call per conversation."""
    seen_catalogs: set[str] = set()
    emitted = 0
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if emitted >= limit:
                        return
                    try:
                        conv = Conversation.from_json(line)
                    except Exception:
                        continue
                    if conv.tools:
                        catalog = render_function_catalog(conv.tools)
                        key = hashlib.sha1(catalog.encode()).hexdigest()
                        if key not in seen_catalogs:
                            seen_catalogs.add(key)
                            emitted += 1
                            yield catalog
                    for message in conv.messages:
                        if message.tool_calls:
                            emitted += 1
                            yield render_tool_calls(message.tool_calls)
    print(f"  agent renderings: {emitted}", file=sys.stderr)


def report(new: Tokenizer, old_path: str | None) -> dict:
    old = Tokenizer.from_file(old_path) if old_path and Path(old_path).is_file() else None
    rows = []
    for probe in PROBES:
        entry = {"probe": probe[:60], "new": len(new.encode(probe).ids)}
        if old is not None:
            entry["old"] = len(old.encode(probe).ids)
        rows.append(entry)
    return {"vocab_size": new.get_vocab_size(), "probes": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw/pt-4b", type=Path)
    ap.add_argument("--conversations", nargs="+", default=["data/public/fit2048-*.jsonl"])
    ap.add_argument("--out", default="data/tokenizer-agent-16k.json")
    ap.add_argument("--compare-to", default="data/tokenizer-h100-16k.json")
    ap.add_argument("--vocab-size", type=int, default=16_384)
    ap.add_argument("--docs-per-source", type=int, default=120_000)
    ap.add_argument("--agent-lines", type=int, default=400_000)
    args = ap.parse_args()
    if Path(args.out).resolve() == Path(args.compare_to).resolve():
        raise SystemExit("refusing to overwrite the shared tokenizer; choose a new --out")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True)
    tokenizer.decoder = decoders.ByteLevel(add_prefix_space=True, trim_offsets=True)
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)

    def corpus():
        yield from text_lines(args.raw, args.docs_per_source)
        yield from agent_lines(args.conversations, args.agent_lines)

    tokenizer.train_from_iterator(corpus(), trainer=trainer)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out))
    summary = report(tokenizer, args.compare_to)
    summary["sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    summary["specials"] = SPECIALS
    summary["docs_per_source"] = args.docs_per_source
    summary["agent_lines"] = args.agent_lines
    Path(str(out) + ".report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
