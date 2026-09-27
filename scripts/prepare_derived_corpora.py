#!/usr/bin/env python3
"""Prepare parent-freeze-derived shard groups for midtraining.

Publication is coordinated by cooperative lock files, uses no-replace artifact links, and writes
each manifest last. Failed or interrupted runs never delete destination names, but can leave
complete groups or exact manifestless subsets; SIGKILL or power loss can additionally leave stale
locks. After confirming no publisher remains alive, remove stale locks manually and rerun the
identical command for deterministic repair.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

from localagent.data.derived_corpus import (
    parse_group_definition,
    prepare_derived_corpora,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fan one verified parent filtered corpus into disjoint, split-preserving shard groups."
        ),
        epilog=(
            "Publication uses cooperative locks and manifests last. After SIGKILL or power loss, "
            "confirm no publisher is alive, manually remove stale lock files, then rerun the "
            "identical command to repair exact manifestless partial groups."
        ),
    )
    parser.add_argument(
        "--freeze",
        "--parent-freeze",
        dest="freeze",
        required=True,
        help="recorded parent corpus freeze JSON",
    )
    parser.add_argument(
        "--freeze-spec",
        "--parent-freeze-spec",
        dest="freeze_spec",
        required=True,
        help="parent corpus freeze specification",
    )
    parser.add_argument(
        "--parent-filtered-jsonl",
        "--filtered-jsonl",
        dest="parent_filtered_jsonl",
        required=True,
        help="exact filtered JSONL bound by the parent freeze",
    )
    parser.add_argument(
        "--parent-manifest",
        "--frozen-split-manifest",
        dest="parent_manifest",
        required=True,
        help="exact parent packed manifest whose frozen split is inherited",
    )
    parser.add_argument(
        "--tokenizer",
        "--tokenizer-path",
        dest="tokenizer",
        required=True,
        help="exact tokenizer artifact bound by the parent freeze",
    )
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        metavar="OUTPUT_DIR=SOURCE[+SOURCE...]",
        help=(
            "repeatable disjoint output mapping, for example "
            "--group data/shards/paper-general=fineweb_edu_dedup+cosmopedia_v2"
        ),
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="root for relative input and output paths (default: current directory)",
    )
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        default=2048,
        help="packed rows per immutable NumPy shard",
    )
    parser.add_argument(
        "--require-complete-parent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require every parent mixture_source to belong to exactly one group (default: true)",
    )
    return parser


def _group_mapping(values: Sequence[str]) -> OrderedDict[str, tuple[str, ...]]:
    groups: OrderedDict[str, tuple[str, ...]] = OrderedDict()
    for value in values:
        output, sources = parse_group_definition(value)
        if output in groups:
            raise ValueError(f"group output is repeated: {output}")
        groups[output] = sources
    return groups


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        groups = _group_mapping(args.group)
        manifests = prepare_derived_corpora(
            freeze_path=args.freeze,
            spec_path=args.freeze_spec,
            parent_filtered_jsonl=args.parent_filtered_jsonl,
            parent_manifest=args.parent_manifest,
            tokenizer_path=args.tokenizer,
            groups=groups,
            project_root=args.project_root,
            rows_per_shard=args.rows_per_shard,
            require_complete_parent=args.require_complete_parent,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    summary = {
        output: {
            "generation": manifest["generation"],
            "documents": manifest["total_documents"],
            "train_documents": manifest["splits"]["train"]["documents"],
            "val_documents": manifest["splits"]["val"]["documents"],
            "manifest": str(Path(output) / "manifest.json"),
        }
        for output, manifest in sorted(manifests.items())
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
