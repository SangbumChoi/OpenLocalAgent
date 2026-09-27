#!/usr/bin/env python
"""Pack each pretraining source into its own shard directory, one loader per dataset.

A merged corpus cannot answer the question the mixture exists to ask. `pt-big` was packed as a
single `filtered.jsonl` with no per-row provenance, so once it existed there was no way to say
which of FineWeb-Edu, Cosmopedia or permissive Python the loss was coming from - or to reweight
them without rebuilding everything. The manifest recorded exactly one source: `file:.jsonl`.

Per-source shards make the mixture a runtime decision instead of a packing decision. Each dataset
keeps its own directory, its own manifest and its own token count; the trainer draws a
micro-batch per source, so every batch knows where it came from and per-source loss curves fall
out of the existing accounting. That is how midtrain already works, and why midtrain can attribute
loss to a corpus while pretrain cannot.

Sources are discovered from the staging directory's filename prefixes (`fineweb-00000.jsonl` ->
`fineweb`), so adding a dataset to the fetch script is enough - nothing here needs editing.

  python scripts/build_source_shards.py --raw data/raw/pt-4b --out data/shards/src
  python scripts/build_source_shards.py --raw data/raw/pt-4b --only fineweb,c4
  python scripts/build_source_shards.py --raw data/raw/pt-4b --list
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Source:
    """One dataset and the staged files that make it up."""

    #: `fineweb-00003.jsonl` -> `fineweb`. A file with no index is its own source. This naming
    #: rule is what a Source *is*, so it lives here rather than beside the module's constants.
    PREFIX = re.compile(r"^(?P<name>.+?)(?:-\d+)?\.jsonl$")

    name: str
    files: tuple[Path, ...]

    @property
    def bytes(self) -> int:
        return sum(f.stat().st_size for f in self.files)

    @classmethod
    def discover(cls, raw: Path) -> list[Source]:
        """Group a staging directory's files by filename prefix.

        Adding a dataset to the fetch script is therefore enough - no registry to edit, and no
        chance of a source existing on disk that the packer does not know about.
        """
        grouped: dict[str, list[Path]] = defaultdict(list)
        for path in sorted(raw.glob("*.jsonl")):
            match = cls.PREFIX.match(path.name)
            if match:
                grouped[match.group("name")].append(path)
        return [cls(name, tuple(files)) for name, files in sorted(grouped.items())]

    @classmethod
    def select(cls, sources: list[Source], names: str) -> list[Source]:
        """Narrow to a comma-separated subset, refusing a name that does not exist.

        A typo must not silently pack nothing: this is a job measured in hours, and "it finished
        quickly" is the worst way to learn that `--only fineweb2` matched no source.
        """
        wanted = {name.strip() for name in names.split(",") if name.strip()}
        available = {source.name for source in sources}
        unknown = wanted - available
        if unknown:
            raise KeyError(f"unknown sources {sorted(unknown)}; found {sorted(available)}")
        return [source for source in sources if source.name in wanted]

    def shard_dir(self, out: Path) -> Path:
        return out / self.name

    def is_packed(self, out: Path) -> bool:
        """A source with a manifest is done; re-running must not redo hours of work."""
        return (self.shard_dir(out) / "manifest.json").is_file()

    def pack(self, out: Path, tokenizer: Path, staging: Path, seq_len: int) -> int:
        """Run build_corpus.py over just this source's files. Returns the exit code.

        --reuse-tokenizer is not optional: without it build_corpus retrains the BPE and
        overwrites the shared tokenizer, which has already destroyed one ladder's lineage. Every
        source must also share one tokenizer or the shards cannot be mixed at all.
        """
        target = self.shard_dir(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/build_corpus.py"),
            *[str(f) for f in self.files],
            "--out", str(target),
            "--reuse-tokenizer",
            "--staging-db", str(staging / f"{self.name}.sqlite3"),
            "--seq-len", str(seq_len), "--rows-per-shard", "4096", "--val-fraction", "0.01",
            "--tokenizer", "bpe", "--tokenizer-path", str(tokenizer),
            "--vocab-size", "16384", "--no-near-dedup",
        ]
        return subprocess.run(command, cwd=ROOT).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw/pt-4b", help="staging dir of *.jsonl sources")
    parser.add_argument("--out", default="data/shards/src", help="parent dir for per-source shards")
    parser.add_argument("--tokenizer", default="data/tokenizer-h100-16k.json")
    parser.add_argument("--staging", default="/var/tmp/ola-corpus-staging",
                        help="SQLite staging root; keep this on local disk, never on NFS")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--only", help="comma-separated source names")
    parser.add_argument("--list", action="store_true", help="show discovered sources and exit")
    args = parser.parse_args()

    raw, out = ROOT / args.raw, ROOT / args.out
    sources = Source.discover(raw)
    if not sources:
        print(f"no *.jsonl sources under {raw}", file=sys.stderr)
        return 1
    if args.only:
        try:
            sources = Source.select(sources, args.only)
        except KeyError as unknown:
            print(unknown.args[0], file=sys.stderr)
            return 1

    if args.list:
        for source in sources:
            state = "packed" if source.is_packed(out) else "pending"
            print(f"{source.name:<14} {len(source.files):>2} files  "
                  f"{source.bytes / 1e9:6.2f} GB  {state}")
        return 0

    staging = Path(args.staging)
    staging.mkdir(parents=True, exist_ok=True)
    for source in sources:
        if source.is_packed(out):
            print(f"[{time.strftime('%H:%M:%S')}] have {source.name}", flush=True)
            continue
        print(f"[{time.strftime('%H:%M:%S')}] packing {source.name} "
              f"({len(source.files)} files, {source.bytes / 1e9:.2f} GB)", flush=True)
        code = source.pack(out, ROOT / args.tokenizer, staging, args.seq_len)
        if code != 0:
            print(f"[{time.strftime('%H:%M:%S')}] FAILED {source.name} (exit {code})",
                  file=sys.stderr)
            return code
        print(f"[{time.strftime('%H:%M:%S')}] done {source.name}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] SOURCE_SHARDS_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
