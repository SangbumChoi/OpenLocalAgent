#!/usr/bin/env python
"""Create or verify a deterministic packed-corpus freeze.

Examples:
  python scripts/freeze_corpus.py configs/data/pretrain-paper-freeze.yaml \
      --out data/shards/paper-all/freeze.json
  python scripts/freeze_corpus.py configs/data/pretrain-paper-freeze.yaml \
      --verify data/shards/paper-all/freeze.json
"""

from __future__ import annotations

import argparse
import json

from localagent.data.corpus_freeze import (
    build_corpus_freeze,
    verify_corpus_freeze,
    write_corpus_freeze,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", help="versioned corpus-freeze YAML specification")
    parser.add_argument(
        "--project-root",
        default=".",
        help="root for project-relative paths in the spec and packed manifest",
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--out", help="atomically write the verified freeze JSON")
    output.add_argument("--verify", help="verify an existing freeze JSON against local artifacts")
    args = parser.parse_args()

    try:
        if args.verify:
            freeze = verify_corpus_freeze(
                args.verify,
                args.spec,
                project_root=args.project_root,
            )
        else:
            freeze = build_corpus_freeze(
                args.spec,
                project_root=args.project_root,
            )
            if args.out:
                write_corpus_freeze(freeze, args.out)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
