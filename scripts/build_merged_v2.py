#!/usr/bin/env python
"""Merge the public corpora into one catalog-carrying training set with held-out splits.

Every row leaves here with a tool catalog attached, because the models trained on it are rendered
with the `openai_full_catalog_v1` contract: the catalog is in the prompt, so a model can answer
about tools it never saw in training. Rows that arrive without one (the synthetic episodes) get the
standard toolset; each source is capped and split deterministically so the eval halves are disjoint.

  python scripts/build_merged_v2.py --cap 6000 --out-dir data/merged-v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, replace
from pathlib import Path

from openlocalagent.agent.toolset import STANDARD_TOOLS
from openlocalagent.data.synth.agent_synth import Generator
from openlocalagent.data.conversation_artifact import (
    conversation_semantic_sha256,
    rendered_assistant_prompts,
    rendered_prompt_sha256,
)
from openlocalagent.data.prompt_contract import OPENAI_FULL_CATALOG_V1
from openlocalagent.data.render import render_conversation_rows
from openlocalagent.model.tokenizer import load_tokenizer
from openlocalagent.train.stage_data import read_conversations

PUBLIC = Path("data/public")
SOURCES = {
    "toolace": PUBLIC / "toolace-train.jsonl",
    "mind2web": PUBLIC / "mind2web-train.jsonl",
    "androidcontrol": PUBLIC / "androidcontrol-train.jsonl",
}


def fits(conversation, tok, max_seq_len: int) -> bool:
    """Whether the row renders inside the context under the full-catalog contract.

    The contract refuses to truncate a catalog row, so an oversize row is a hard training error.
    Dropping them here — and counting them — keeps that guarantee instead of silently weakening it.
    """
    try:
        render_conversation_rows(conversation, tok, prompt_contract=OPENAI_FULL_CATALOG_V1,
                                 max_seq_len=max_seq_len)
    except (ValueError, KeyError):
        return False
    return True


def with_catalog(conversation, tools, rng=None, catalog_size: int = 12):
    """Attach a task-scoped catalog: the tools the row actually calls, plus distractors.

    Handing every synthetic row the whole 50-tool standard set pushes the rendered prompt past the
    context window, and it is not how the public rows look either — each carries a small catalog
    for its own task. Distractors keep the selection problem non-trivial.
    """
    if conversation.tools:
        return conversation
    used = {call.name for message in conversation.messages for call in (message.tool_calls or ())}
    chosen = [tool for tool in tools if tool.name in used]
    others = [tool for tool in tools if tool.name not in used]
    (rng or random).shuffle(others)
    chosen += others[: max(0, catalog_size - len(chosen))]
    chosen.sort(key=lambda tool: tool.name)
    return replace(conversation, tools=tuple(chosen))


def prompt_fingerprints(conversation) -> tuple[str, ...]:
    """The trainer's own rendered-prompt hashes for every assistant decision in a row."""
    return tuple(rendered_prompt_sha256(prompt) for prompt in rendered_assistant_prompts(
        conversation, conversation_prompt_contract=OPENAI_FULL_CATALOG_V1))


def split_disjoint(rows, bucket: int):
    """Split rows so no rendered prompt appears on both sides.

    Rows that share any prompt must travel together — a ToolACE prompt with two acceptable calls,
    or two steps of one trajectory that render the same prefix — so they are grouped first and the
    group, not the row, is assigned. This is the same identity the trainer audits with.
    """
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        while parent.setdefault(node, node) != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[left_root] = right_root

    seen_semantic, unique = set(), []
    for row in rows:
        semantic = conversation_semantic_sha256(row)
        if semantic in seen_semantic:
            continue
        seen_semantic.add(semantic)
        prompts = prompt_fingerprints(row) or (semantic,)
        for prompt in prompts[1:]:
            union(prompts[0], prompt)
        unique.append((prompts[0], row))

    train_rows, eval_rows = [], []
    for anchor, row in unique:
        group = find(anchor)
        (eval_rows if int(group[:8], 16) % bucket == 0 else train_rows).append(row)
    return train_rows, eval_rows


def write(path: Path, rows) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), default=str, sort_keys=True) + "\n")
    payload = path.read_bytes()
    return {"path": str(path), "rows": len(rows), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=6000, help="rows kept per public source")
    ap.add_argument("--synthetic-episodes", type=int, default=1200)
    ap.add_argument("--eval-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", default="data/merged-v2")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--tokenizer", default="data/tokenizer-h100-16k.json")
    args = ap.parse_args()

    out = Path(args.out_dir)
    rng = random.Random(args.seed)
    tok = load_tokenizer("bpe", args.tokenizer)
    train, evaluation, manifest_sources = [], [], []
    dropped = {}

    generator = Generator(level=4, seed=args.seed, split="train")
    # Trajectory episodes plus planner-style episodes: the only multi-turn structure in the mix.
    synthetic = [with_catalog(c, list(STANDARD_TOOLS), rng)
                 for c in generator.episodes(args.synthetic_episodes)]
    synthetic += [with_catalog(c, list(STANDARD_TOOLS), rng) for c in
                  Generator(level=4, seed=args.seed + 1, split="train")
                  .plan_episodes(args.synthetic_episodes // 2)]
    bucket = max(2, int(round(1 / args.eval_fraction)))
    before = len(synthetic)
    synthetic = [row for row in synthetic if fits(row, tok, args.max_seq_len)]
    dropped["synthetic"] = before - len(synthetic)
    kept, held = split_disjoint(synthetic, bucket)
    evaluation += held
    train += kept
    manifest_sources.append({"source": "synthetic", "train": len(kept), "eval": len(held)})

    for name, path in SOURCES.items():
        if not path.exists():
            continue
        available = read_conversations(path)
        rows = [with_catalog(c, list(STANDARD_TOOLS), rng) for c in available]
        rng.shuffle(rows)
        candidates = rows[: args.cap * 3]
        rows = [row for row in candidates if fits(row, tok, args.max_seq_len)]
        dropped[name] = len(candidates) - len(rows)
        source_train, source_eval = split_disjoint(rows[: args.cap * 2], bucket)
        source_train, source_eval = source_train[: args.cap], source_eval[: args.cap // 8]
        write(out / f"eval-{name}.jsonl", source_eval)
        evaluation += source_eval
        train += source_train
        manifest_sources.append({"source": name, "train": len(source_train),
                                 "eval": len(source_eval), "available": len(available)})

    rng.shuffle(train)
    rng.shuffle(evaluation)
    manifest = {
        "kind": "localagent_merged_agent_corpus",
        "schema_version": 2,
        "seed": args.seed,
        "cap_per_public_source": args.cap,
        "prompt_contract": "openai_full_catalog_v1",
        "note": ("Every row carries a tool catalog so the trained model reads its action space from "
                 "the prompt instead of memorising it in the weights."),
        "max_seq_len": args.max_seq_len,
        "dropped_oversize_rows": dropped,
        "sources": manifest_sources,
        "train": write(out / "train.jsonl", train),
        "eval": write(out / "eval.jsonl", evaluation),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"train": len(train), "eval": len(evaluation),
                      "sources": manifest_sources}, indent=2))
    print("MERGED_V2_DONE " + str(out), flush=True)


if __name__ == "__main__":
    main()
