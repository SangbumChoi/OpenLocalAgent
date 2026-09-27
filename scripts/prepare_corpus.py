#!/usr/bin/env python
"""Filter, audit, split, tokenize, and pack a LocalAgent pretraining corpus.

Examples:
  python scripts/prepare_corpus.py --sample --out data/shards/sample --seq-len 128
  python scripts/prepare_corpus.py data/raw/general data/raw/code --out data/shards/mixed
  python scripts/prepare_corpus.py data/raw --tokenizer bpe --vocab-size 16384 \
      --tokenizer-path data/tokenizer-16k.json --out data/shards/bpe
  python scripts/prepare_corpus.py data/shards/base/filtered.jsonl \
      --frozen-split-manifest data/shards/base/manifest.json --out data/shards/subset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml

from localagent.data.evaluation_denylist_manifest import (
    verify_evaluation_denylist_manifest,
)
from localagent.data.hf_corpus import normalize_evaluation_decontamination
from localagent.data.pretrain_corpus import (
    CorpusDocument,
    build_disk_backed_corpus,
    download_sample,
    iter_documents,
    load_frozen_split_assignment_manifest,
    pack_disk_backed_shards,
    read_evaluation_denylist,
)
from localagent.model.tokenizer import ByteTokenizer, load_tokenizer, train_bpe


def _file_artifact(path: str | Path) -> dict[str, object]:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"artifact is missing or is not a file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": digest.hexdigest(),
    }


_SUITE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _named_denylist_path(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, raw_path = value.split("=", 1)
    else:
        raw_path = value
        name = Path(raw_path).stem
    name = name.strip()
    if _SUITE_NAME.fullmatch(name) is None:
        raise ValueError(
            f"evaluation denylist name {name!r} must contain only letters, digits, '.', '_', '-'"
        )
    if not raw_path:
        raise ValueError(f"evaluation denylist {name!r} has an empty path")
    return name, Path(raw_path)


def _verified_denylist_manifest(
    path: str | Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    return verify_evaluation_denylist_manifest(path)


def _source_manifest_inputs(
    paths: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Checksum upstream manifests and collect config-owned decontamination policies."""

    artifacts: list[dict[str, object]] = []
    policies: list[dict[str, object]] = []
    for raw_path in paths:
        path = Path(raw_path)
        artifact = _file_artifact(path)
        artifacts.append(artifact)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        embedded_policy = None
        if "evaluation_decontamination" in payload:
            embedded_policy = normalize_evaluation_decontamination(
                {"evaluation_decontamination": payload["evaluation_decontamination"]}
            )

        config_policy = None
        config_artifact = None
        config_loaded = False
        declared_config = payload.get("config")
        expected_config_sha256 = payload.get("config_sha256")
        if isinstance(declared_config, str) and isinstance(expected_config_sha256, str):
            declared_path = Path(declared_config)
            candidates = (
                [declared_path]
                if declared_path.is_absolute()
                else [declared_path, path.parent / declared_path]
            )
            config_path = next((candidate for candidate in candidates if candidate.is_file()), None)
            if config_path is not None:
                config_loaded = True
                config_artifact = _file_artifact(config_path)
                if config_artifact["sha256"] != expected_config_sha256:
                    raise ValueError(
                        f"{path}: referenced corpus config SHA-256 mismatch: {config_path}"
                    )
                expected_config_bytes = payload.get("config_bytes")
                if (
                    expected_config_bytes is not None
                    and config_artifact["bytes"] != expected_config_bytes
                ):
                    raise ValueError(
                        f"{path}: referenced corpus config byte-size mismatch: {config_path}"
                    )
                config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                if not isinstance(config_payload, dict):
                    raise ValueError(f"{config_path}: corpus config must be a mapping")
                config_policy = normalize_evaluation_decontamination(config_payload)
        if config_loaded and embedded_policy is not None:
            if config_policy != embedded_policy:
                raise ValueError(
                    f"{path}: embedded evaluation policy differs from referenced corpus config"
                )
        policy = config_policy if config_loaded else embedded_policy
        if policy is None:
            continue
        policies.append(
            {
                **policy,
                "source_manifest": {
                    "path": str(path),
                    "bytes": artifact["bytes"],
                    "sha256": artifact["sha256"],
                },
                "policy_config": config_artifact,
            }
        )
    return artifacts, policies


def _merge_required_suites(
    policies: list[dict[str, object]],
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    required: dict[str, dict[str, object]] = {}
    policy_sources: list[dict[str, object]] = []
    for policy in policies:
        source = policy.get("source_manifest")
        if isinstance(source, dict):
            policy_sources.append(source)
        suites = policy.get("required_suites", [])
        if not isinstance(suites, list):
            raise ValueError("evaluation decontamination policy required_suites is invalid")
        for suite in suites:
            if not isinstance(suite, dict) or not isinstance(suite.get("name"), str):
                raise ValueError("evaluation decontamination policy suite is invalid")
            name = str(suite["name"])
            constraint = {
                key: suite[key] for key in ("bytes", "sha256") if key in suite
            }
            previous = required.get(name)
            if previous is not None and previous != constraint:
                raise ValueError(
                    f"conflicting corpus policies for required evaluation suite {name!r}"
                )
            required[name] = constraint
    return required, policy_sources


def _evaluation_denylist_inputs(
    direct_inputs: list[str],
    list_manifests: list[str],
    required_suites: dict[str, dict[str, object]] | None = None,
    policy_sources: list[dict[str, object]] | None = None,
) -> tuple[list[Path], dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    manifest_artifacts: list[dict[str, object]] = []
    for raw_manifest in list_manifests:
        manifest_inputs, manifest_artifact = _verified_denylist_manifest(raw_manifest)
        artifacts.extend(manifest_inputs)
        manifest_artifacts.append(manifest_artifact)
    for raw_input in direct_inputs:
        name, path = _named_denylist_path(raw_input)
        artifacts.append(
            {
                "name": name,
                **_file_artifact(path),
                "source": "cli",
            }
        )

    names = [str(artifact["name"]) for artifact in artifacts]
    if len(names) != len(set(names)):
        duplicates = sorted(name for name in set(names) if names.count(name) > 1)
        raise ValueError(f"duplicate evaluation denylist name(s): {', '.join(duplicates)}")
    paths = [Path(str(artifact["path"])) for artifact in artifacts]
    resolved_paths = [path.resolve() for path in paths]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("the same evaluation denylist file was supplied more than once")
    artifacts_by_name = {str(artifact["name"]): artifact for artifact in artifacts}
    required_suites = required_suites or {}
    missing_required = sorted(set(required_suites) - set(artifacts_by_name))
    if missing_required:
        raise ValueError(
            "corpus policy requires evaluation denylist suite(s) missing from supplied inputs: "
            + ", ".join(missing_required)
        )
    for name, constraint in sorted(required_suites.items()):
        artifact = artifacts_by_name[name]
        if not constraint and artifact.get("source") != "list_manifest":
            raise ValueError(
                f"corpus policy suite {name!r} is not hash-pinned by the config and must be "
                "supplied through --eval-denylist-manifest"
            )
        if "bytes" in constraint and artifact["bytes"] != constraint["bytes"]:
            raise ValueError(f"corpus policy byte-size mismatch for suite {name!r}")
        if "sha256" in constraint and artifact["sha256"] != constraint["sha256"]:
            raise ValueError(f"corpus policy SHA-256 mismatch for suite {name!r}")
    canonical_inputs = sorted(
        (
            {
                "bytes": artifact["bytes"],
                "name": artifact["name"],
                "sha256": artifact["sha256"],
            }
            for artifact in artifacts
        ),
        key=lambda artifact: str(artifact["name"]),
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            canonical_inputs,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return paths, {
        "inputs": sorted(artifacts, key=lambda artifact: str(artifact["name"])),
        "input_count": len(artifacts),
        "inputs_sha256": fingerprint,
        "list_manifests": manifest_artifacts,
        "required_suites": [
            {"name": name, **constraint}
            for name, constraint in sorted(required_suites.items())
        ],
        "required_suite_policy_sources": policy_sources or [],
    }


def _select_mixture_sources(
    documents,
    included_sources: set[str],
):
    """Filter downloader rows by their auditable mixture-source label."""

    if not included_sources:
        yield from documents
        return
    for document in documents:
        if not isinstance(document, CorpusDocument):
            raise TypeError("mixture filtering expects CorpusDocument rows")
        if str(document.meta.get("mixture_source", "")) in included_sources:
            yield document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", help=".txt/.md/code files, JSONL, or directories")
    parser.add_argument("--sample", action="store_true", help="download a public-domain toy corpus")
    parser.add_argument("--out", required=True, help="output shard directory")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--rows-per-shard", type=int, default=2048)
    parser.add_argument(
        "--val-fraction",
        type=float,
        help="validation fraction (default: 0.01, or inherited from --frozen-split-manifest)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="split seed (default: 42, or inherited from --frozen-split-manifest)",
    )
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=2_000_000)
    parser.add_argument(
        "--staging-db",
        help="SQLite staging path (default: OUT/corpus-staging.sqlite3)",
    )
    parser.add_argument(
        "--source-manifest",
        action="append",
        default=[],
        metavar="PATH",
        help="upstream download/license manifest to checksum as provenance (repeatable)",
    )
    parser.add_argument(
        "--eval-denylist",
        action="append",
        default=[],
        metavar="[NAME=]PATH",
        help=(
            "plain-text, JSONL, or versioned JSON-suite evaluation prompts to screen before "
            "splitting; every input is checksummed (repeatable)"
        ),
    )
    parser.add_argument(
        "--eval-denylist-manifest",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "self-hashed provenance-bound list of named prompt-only denylist suites "
            "(repeatable)"
        ),
    )
    parser.add_argument(
        "--decontam-coverage",
        type=float,
        default=0.9,
        help="minimum denylist token-shingle containment for exclusion (default: 0.9)",
    )
    parser.add_argument(
        "--near-dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable bounded SimHash/shingle near-duplicate filtering (default: enabled)",
    )
    parser.add_argument("--near-dedup-shingle-size", type=int, default=5)
    parser.add_argument("--near-dedup-max-shingles", type=int, default=256)
    parser.add_argument("--near-dedup-hamming", type=int, default=3)
    parser.add_argument("--near-dedup-jaccard", type=float, default=0.95)
    parser.add_argument("--tokenizer", choices=["byte", "bpe"], default="byte")
    parser.add_argument("--tokenizer-path", default="data/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=16_384)
    parser.add_argument(
        "--reuse-tokenizer",
        action="store_true",
        help="load the existing BPE artifact instead of fitting a new tokenizer",
    )
    parser.add_argument(
        "--include-mixture-source",
        action="append",
        default=[],
        metavar="NAME",
        help="retain only rows whose meta.mixture_source matches NAME (repeatable)",
    )
    parser.add_argument(
        "--frozen-split-manifest",
        metavar="PATH",
        help=(
            "reuse the content-bound document split artifact referenced by a base packed-corpus "
            "manifest; missing, changed, or unmapped documents fail closed"
        ),
    )
    args = parser.parse_args()

    inputs = list(args.inputs)
    if args.sample:
        inputs.append(str(download_sample("data/raw/sample")))
    if not inputs:
        parser.error("provide at least one input or --sample")
    if args.reuse_tokenizer and args.tokenizer != "bpe":
        parser.error("--reuse-tokenizer requires --tokenizer bpe")

    try:
        frozen_split_assignment = (
            load_frozen_split_assignment_manifest(args.frozen_split_manifest)
            if args.frozen_split_manifest
            else None
        )
        seed = (
            args.seed
            if args.seed is not None
            else frozen_split_assignment.seed
            if frozen_split_assignment is not None
            else 42
        )
        val_fraction = (
            args.val_fraction
            if args.val_fraction is not None
            else frozen_split_assignment.val_fraction
            if frozen_split_assignment is not None
            else 0.01
        )
        source_manifests, decontamination_policies = _source_manifest_inputs(
            args.source_manifest
        )
        required_suites, policy_sources = _merge_required_suites(
            decontamination_policies
        )
        denylist_paths, denylist_provenance = _evaluation_denylist_inputs(
            args.eval_denylist,
            args.eval_denylist_manifest,
            required_suites,
            policy_sources,
        )
        denylist = read_evaluation_denylist(denylist_paths) if denylist_paths else []
        reverified_list_manifests = [
            _verified_denylist_manifest(raw_manifest)[1]
            for raw_manifest in args.eval_denylist_manifest
        ]
        if (
            reverified_list_manifests
            != denylist_provenance["list_manifests"]
        ):
            raise ValueError(
                "evaluation denylist artifacts changed while their prompts were read"
            )
        denylist_provenance["normalized_entries"] = len(denylist)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        staging_path = (
            Path(args.staging_db) if args.staging_db is not None else out / "corpus-staging.sqlite3"
        )
        corpus = build_disk_backed_corpus(
            _select_mixture_sources(
                iter_documents(inputs),
                set(args.include_mixture_source),
            ),
            staging_path,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            denylist=denylist,
            decontam_coverage=args.decontam_coverage,
            near_dedup=args.near_dedup,
            near_dedup_shingle_size=args.near_dedup_shingle_size,
            near_dedup_max_shingles=args.near_dedup_max_shingles,
            near_dedup_hamming=args.near_dedup_hamming,
            near_dedup_jaccard=args.near_dedup_jaccard,
            val_fraction=val_fraction,
            seed=seed,
            frozen_split_assignment=frozen_split_assignment,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    filtered_path = out / "filtered.jsonl"
    filtered_artifact = corpus.write_filtered_jsonl(filtered_path)

    if args.tokenizer == "byte":
        tokenizer = ByteTokenizer()
        tokenizer_training = {
            "kind": "byte",
            "trained": False,
            "split": None,
        }
    elif args.reuse_tokenizer:
        tokenizer = load_tokenizer("bpe", args.tokenizer_path)
        tokenizer_training = {
            "kind": "bpe",
            "trained": False,
            "reused": True,
            "split": None,
            "path": str(args.tokenizer_path),
            "vocab_size": tokenizer.vocab_size,
            "artifact": _file_artifact(args.tokenizer_path),
        }
    else:
        tokenizer = train_bpe(
            (document.text for document in corpus.iter_documents("train")),
            args.tokenizer_path,
            vocab_size=args.vocab_size,
        )
        tokenizer_training = {
            "kind": "bpe",
            "trained": True,
            "split": "train",
            "path": str(args.tokenizer_path),
            "requested_vocab_size": args.vocab_size,
            "vocab_size": tokenizer.vocab_size,
        }
        tokenizer_path = Path(args.tokenizer_path)
        if tokenizer_path.is_file():
            tokenizer_training["artifact"] = _file_artifact(tokenizer_path)

    preparation_provenance = {
        "input_paths": inputs,
        "included_mixture_sources": sorted(set(args.include_mixture_source)),
        "source_manifests": source_manifests,
        "evaluation_denylist_paths": list(args.eval_denylist),
        "evaluation_denylist": denylist_provenance,
        "filtered_jsonl": filtered_artifact,
    }
    if frozen_split_assignment is not None:
        preparation_provenance["frozen_split_manifest"] = dict(
            frozen_split_assignment.source_manifest
        )
    manifest = pack_disk_backed_shards(
        corpus,
        tokenizer,
        args.seq_len,
        str(out),
        rows_per_shard=args.rows_per_shard,
        tokenizer_training=tokenizer_training,
        preparation_provenance=preparation_provenance,
    )
    raw_documents = int(
        manifest["corpus_audit"]["quality_and_exact_deduplication"]["input_documents"]
    )
    accepted_documents = int(manifest["total_documents"])
    print(
        json.dumps(
            {
                "raw_documents": raw_documents,
                "accepted_documents": accepted_documents,
                "rejected_documents": raw_documents - accepted_documents,
                "licenses": manifest["license_counts"],
                "filtered_jsonl": str(filtered_path),
                "staging_database": str(staging_path),
                "corpus_audit": manifest["corpus_audit"],
                "tokenizer_training": manifest["tokenizer_training"],
                "manifest": manifest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
