#!/usr/bin/env python
"""Materialise the designed union corpus as one Conversation JSONL with a provenance manifest.

A naive concatenation of the public corpora costs multi-turn ability, so the mixture is capped per
source and the synthetic trajectory episodes are kept whole. The manifest records every source
file's bytes and SHA-256, the cap that was applied, and the realised composition.

  python scripts/build_union_dataset.py --cap 1000 --out data/public/openlocalagent-union-v1.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path

from openlocalagent.data.synth.agent_synth import Generator
from openlocalagent.train.stage_data import read_conversations

PUBLIC = Path("data/public")
SOURCES = {
    "toolace": PUBLIC / "toolace-train.jsonl",
    "mind2web": PUBLIC / "mind2web-train.jsonl",
    "androidcontrol": PUBLIC / "androidcontrol-train.jsonl",
}


def digest(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def serialise(conversation) -> str:
    if is_dataclass(conversation):
        return json.dumps(asdict(conversation), default=str, sort_keys=True)
    return json.dumps(conversation, default=str, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=1000, help="rows kept per public source")
    ap.add_argument("--episodes", type=int, default=360, help="synthetic trajectory episodes kept")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="data/public/openlocalagent-union-v1.jsonl")
    ap.add_argument("--manifest", default="data/public/openlocalagent-union-v1.manifest.json")
    args = ap.parse_args()

    rows, composition, provenance = [], {}, []
    episodes = list(Generator(level=3, seed=5000 + args.seed, split="train").episodes(args.episodes))
    rows.extend(episodes)
    composition["synthetic_episodes"] = len(episodes)

    for name, path in SOURCES.items():
        if not path.exists():
            continue
        source_rows = read_conversations(path)
        random.Random(args.seed).shuffle(source_rows)
        kept = source_rows[: args.cap]
        rows.extend(kept)
        composition[name] = len(kept)
        provenance.append({"source": name, "available": len(source_rows), "kept": len(kept),
                           **digest(path)})

    random.Random(args.seed + 1).shuffle(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for conversation in rows:
            handle.write(serialise(conversation) + "\n")

    manifest = {
        "kind": "localagent_designed_union_corpus",
        "schema_version": 1,
        "seed": args.seed,
        "cap_per_public_source": args.cap,
        "rationale": (
            "A naive union drowns the multi-turn trajectory signal: public agent corpora are "
            "overwhelmingly single-step. Public sources are capped and the synthetic episodes are "
            "kept whole so trajectory competence survives the mixture."
        ),
        "composition": composition,
        "total_rows": len(rows),
        "sources": provenance,
        "output": digest(out),
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "composition": composition}, indent=2))
    print("UNION_DATASET_DONE " + str(out), flush=True)


if __name__ == "__main__":
    main()
