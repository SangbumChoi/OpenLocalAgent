#!/usr/bin/env python3
"""Run a public-train-only mobile/browser/desktop continuation and weight audit.

Each ``--train-data``/``--eval-data`` argument is ``LABEL=PATH`` and must already be a
normalized ``Conversation`` JSONL.  The script keeps source rows grouped in the receipt,
checks the combined train/eval split, reports per-surface teacher-forced metrics, and records
relative movement of the transferred backbone.  It deliberately does not claim native emulator,
browser, desktop, or external-account success.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

from localagent.data.schema import Conversation
from localagent.model import LocalAgentLM, ModelConfig
from localagent.train.sft import _evaluate_conversations, sft
from localagent.train.stage_data import (
    build_continuation_lineage,
    tokenizer_identity,
)
from scripts.analyze_weight_transfer import analyze as analyze_weight_transfer
from scripts.train_public_agent_continuation import (
    _assert_disjoint,
    _checkpoint_tokenizer,
    _identity,
    _load_rows,
)


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label.strip(), Path(raw_path).expanduser()


def _parse_source_reference(value: str) -> tuple[str, dict[str, str]]:
    label, separator, raw_reference = value.partition("=")
    dataset, url_separator, url = raw_reference.partition("|")
    if not separator or not label.strip() or not dataset.strip() or not url_separator or not url.strip():
        raise argparse.ArgumentTypeError("expected LABEL=DATASET|URL")
    return label.strip(), {"dataset": dataset.strip(), "url": url.strip()}


def _source_profile(
    label: str,
    path: Path,
    rows: list[Conversation],
    reference: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    datasets: set[str] = set()
    revisions: set[str] = set()
    splits: set[str] = set()
    source_families: set[str] = set()
    source_urls: set[str] = set()
    parent_ids: set[str] = set()
    omitted_visuals = 0
    for row in rows:
        meta = row.meta if isinstance(row.meta, Mapping) else {}
        provenance = meta.get("provenance") if isinstance(meta.get("provenance"), Mapping) else {}
        for key, values in (
            ("dataset", (provenance.get("dataset"), meta.get("source_dataset"))),
            ("revision", (provenance.get("revision"), meta.get("source_revision"))),
            ("split", (provenance.get("subset"), meta.get("source_split"), meta.get("split"))),
        ):
            target = datasets if key == "dataset" else revisions if key == "revision" else splits
            target.update(str(value) for value in values if value not in (None, ""))
        source_families.update(
            str(value)
            for value in (meta.get("source_family"), meta.get("source_dataset"))
            if value not in (None, "")
        )
        source_urls.update(
            str(value)
            for value in (provenance.get("url"), meta.get("source_url"))
            if value not in (None, "")
        )
        if meta.get("parent_record_id"):
            parent_ids.add(str(meta["parent_record_id"]))
        omitted_visuals += int(bool(meta.get("visual_input_omitted")))
    return {
        "label": label,
        "input": _identity(path),
        "rows": len(rows),
        "datasets": sorted(datasets),
        "revisions": sorted(revisions),
        "splits": sorted(splits),
        "source_families": sorted(source_families),
        "source_urls": sorted(source_urls),
        "public_reference": dict(reference) if reference else None,
        "unique_parent_records": len(parent_ids),
        "visual_input_omitted_rows": omitted_visuals,
    }


def _evaluate_by_source(
    grouped_rows: list[tuple[str, Path, list[Conversation]]],
    model: LocalAgentLM,
    tokenizer: Any,
    *,
    max_seq_len: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    return {
        label: _evaluate_conversations(
            model,
            rows,
            tokenizer,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            device=device,
        )
        for label, _path, rows in grouped_rows
    }


def _load_labeled_groups(
    values: Iterable[tuple[str, Path]],
    *,
    max_rows: int = 0,
) -> list[tuple[str, Path, list[Conversation]]]:
    if max_rows < 0:
        raise ValueError("max_rows must be nonnegative")
    groups: list[tuple[str, Path, list[Conversation]]] = []
    labels: set[str] = set()
    for label, path in values:
        if label in labels:
            raise ValueError(f"duplicate source label: {label!r}")
        labels.add(label)
        rows = _load_rows([path])
        if max_rows:
            rows = rows[:max_rows]
        groups.append((label, path, rows))
    if not groups:
        raise ValueError("at least one labeled source is required")
    return groups


def _assert_source_disjoint(
    train_groups: list[tuple[str, Path, list[Conversation]]],
    eval_groups: list[tuple[str, Path, list[Conversation]]],
) -> None:
    """Check parent/slot leakage within each named source, not across sources.

    Different public datasets legitimately reuse generic slot names and values (for example,
    ``route=12``).  Treating those values as a global pool falsely rejects a cross-surface
    mixture.  A source-local check still rejects leakage when the same dataset contributes both
    train and evaluation rows.
    """

    train_by_label = {label: rows for label, _path, rows in train_groups}
    eval_by_label = {label: rows for label, _path, rows in eval_groups}
    for label in sorted(set(train_by_label) & set(eval_by_label)):
        _assert_disjoint(train_by_label[label], eval_by_label[label])


def _disable_lazy_torch_dynamo() -> None:
    """Keep eager CPU continuation runs from importing TorchDynamo on optimizer calls.

    Recent PyTorch releases decorate optimizer construction/state methods with a lazy Dynamo
    wrapper.  The wrapper imports the full ``torch._dynamo``/SymPy stack the first time AdamW is
    constructed, which can dominate a bounded CPU experiment and is unnecessary for this eager
    training script.  Unwrap the methods after importing ``torch.optim``; the original functions
    remain numerically identical and this helper is only used behind the explicit CLI flag.
    """

    import torch.optim

    add_param_group = getattr(torch.optim.Optimizer.add_param_group, "__wrapped__", None)
    if add_param_group is not None:
        torch.optim.Optimizer.add_param_group = add_param_group

    # Adam's differentiability wrapper normally toggles no-grad around the in-place parameter
    # update.  Calling its wrapped body directly avoids Dynamo but would make leaf updates illegal,
    # so preserve the eager no-grad behavior explicitly.
    for owner in (torch.optim.Adam, torch.optim.AdamW):
        method = getattr(owner, "step")
        wrapped = getattr(method, "__wrapped__", None)
        if wrapped is not None:
            def eager_step(self, closure=None, _wrapped=wrapped):
                with torch.no_grad():
                    return _wrapped(self, closure)

            setattr(owner, "step", eager_step)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=_parse_labeled_path, action="append", required=True)
    parser.add_argument("--eval-data", type=_parse_labeled_path, action="append", required=True)
    parser.add_argument(
        "--source-reference",
        type=_parse_source_reference,
        action="append",
        default=[],
        help="public source identity as LABEL=DATASET|URL; repeat for every source",
    )
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--backbone-init", choices=("parent", "random"), default="parent")
    parser.add_argument("--random-backbone-seed", type=int, default=2028)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="deterministically cap rows per training source (0 keeps the full source)",
    )
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=0,
        help="deterministically cap rows per evaluation source (0 keeps the full source)",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--disable-dynamo-wrapper",
        action="store_true",
        help=(
            "skip PyTorch's lazy optimizer dynamo wrapper; useful for bounded eager CPU runs "
            "where importing torch._dynamo is disproportionately slow"
        ),
    )
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite continuation outputs")
    if (
        args.steps < 1
        or args.batch_size < 1
        or args.lr <= 0
        or args.max_train_rows < 0
        or args.max_eval_rows < 0
    ):
        raise SystemExit(
            "steps and batch-size must be positive; lr and row caps must be nonnegative/positive"
        )

    train_groups = _load_labeled_groups(args.train_data, max_rows=args.max_train_rows)
    eval_groups = _load_labeled_groups(args.eval_data, max_rows=args.max_eval_rows)
    references = dict(args.source_reference)
    expected_labels = {label for label, _path, _rows in train_groups} | {
        label for label, _path, _rows in eval_groups
    }
    if set(references) != expected_labels:
        missing = sorted(expected_labels - set(references))
        extra = sorted(set(references) - expected_labels)
        raise SystemExit(f"source references must match labels; missing={missing}, extra={extra}")
    train_rows = [row for _label, _path, rows in train_groups for row in rows]
    eval_rows = [row for _label, _path, rows in eval_groups for row in rows]
    _assert_source_disjoint(train_groups, eval_groups)
    parent_identity = _identity(args.init)
    parent = torch.load(args.init, map_location="cpu", weights_only=False)
    config = ModelConfig(**parent["cfg"])
    config.assert_within_budget()
    if args.backbone_init == "random":
        torch.manual_seed(args.random_backbone_seed)
    model = LocalAgentLM(config)
    if args.backbone_init == "parent":
        model.load_state_dict(parent["state_dict"])
    tokenizer = _checkpoint_tokenizer(parent)
    tokenizer_meta = parent.get("tokenizer") or {"kind": "byte"}
    tokenizer_path = tokenizer_meta.get("path")
    tokenizer_lineage = tokenizer_identity(
        tokenizer_meta.get("kind", "byte"),
        vocab_size=config.vocab_size,
        path=tokenizer_path if tokenizer_path is not None else None,
    )

    before_all = _evaluate_conversations(
        model, train_rows, tokenizer, max_seq_len=args.max_seq_len, batch_size=args.batch_size, device=args.device
    )
    before_eval = _evaluate_conversations(
        model, eval_rows, tokenizer, max_seq_len=args.max_seq_len, batch_size=args.batch_size, device=args.device
    )
    before_by_source = _evaluate_by_source(
        eval_groups, model, tokenizer, max_seq_len=args.max_seq_len, batch_size=args.batch_size, device=args.device
    )
    if args.disable_dynamo_wrapper and hasattr(torch, "_disable_dynamo"):
        # PyTorch 2.13 lazily imports the full torch._dynamo/sympy stack on the first AdamW
        # construction.  The continuation script is eager-only; making this opt-in keeps the
        # default behavior unchanged while allowing reproducible CPU canaries to start training.
        torch._disable_dynamo = lambda fn, *unused_args, **unused_kwargs: fn  # type: ignore[attr-defined]
        _disable_lazy_torch_dynamo()
    loss_history, _, _, training = sft(
        model,
        [],
        tokenizer,
        conversations=train_rows,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup=max(1, min(4, args.steps // 2)),
        device=args.device,
        max_seq_len=args.max_seq_len,
        seed=2027,
        log=print,
        return_metrics=True,
    )
    after_all = _evaluate_conversations(
        model, train_rows, tokenizer, max_seq_len=args.max_seq_len, batch_size=args.batch_size, device=args.device
    )
    after_eval = _evaluate_conversations(
        model, eval_rows, tokenizer, max_seq_len=args.max_seq_len, batch_size=args.batch_size, device=args.device
    )
    after_by_source = _evaluate_by_source(
        eval_groups, model, tokenizer, max_seq_len=args.max_seq_len, batch_size=args.batch_size, device=args.device
    )

    train_source_profiles = [
        _source_profile(label, path, rows, references[label])
        for label, path, rows in train_groups
    ]
    eval_source_profiles = [
        _source_profile(label, path, rows, references[label])
        for label, path, rows in eval_groups
    ]
    lineage = build_continuation_lineage(
        parent=parent,
        parent_checkpoint_sha256=parent_identity["sha256"],
        config={
            "stage": "sft_cross_surface_public_continuation",
            "backbone_init": args.backbone_init,
            "random_backbone_seed": args.random_backbone_seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_seq_len": args.max_seq_len,
            "seed": 2027,
        },
        model_config=config.__dict__,
        data_identity={
            "train_sources": train_source_profiles,
            "eval_sources": eval_source_profiles,
            "rows": {"train": len(train_rows), "eval": len(eval_rows)},
        },
        tokenizer=tokenizer_lineage,
        workspace=Path(__file__).resolve(),
    )

    child = dict(parent)
    child.update(
        {
            "state_dict": model.state_dict(),
            "stage": "sft_cross_surface_public_continuation",
            "parent_checkpoint_sha256": parent_identity["sha256"],
            "lineage": lineage,
            "cross_surface_training": {
                "train_sources": [
                    *train_source_profiles,
                ],
                "eval_sources": [
                    *eval_source_profiles,
                ],
                "steps": args.steps,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "max_seq_len": args.max_seq_len,
                "seed": 2027,
                "backbone_init": args.backbone_init,
                "random_backbone_seed": args.random_backbone_seed,
                "before_eval_by_source": before_by_source,
                "after_eval_by_source": after_by_source,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(child, args.output)
    transfer = analyze_weight_transfer(args.init, args.output)
    report = {
        "kind": "localagent_cross_surface_public_continuation_report",
        "schema_version": 1,
        "parent": parent_identity,
        "child": _identity(args.output),
        "train_sources": [
            _source_profile(label, path, rows, references[label])
            for label, path, rows in train_groups
        ],
        "eval_sources": [
            _source_profile(label, path, rows, references[label])
            for label, path, rows in eval_groups
        ],
        "rows": {"train": len(train_rows), "eval": len(eval_rows)},
        "split_contract": {
            "mode": "source_local_parent_and_slot_disjoint",
            "labels_checked": sorted(
                {label for label, _path, _rows in train_groups}
                | {label for label, _path, _rows in eval_groups}
            ),
            "cross_source_slot_reuse_allowed": True,
        },
        "hyperparameters": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_seq_len": args.max_seq_len,
            "seed": 2027,
            "backbone_init": args.backbone_init,
            "random_backbone_seed": args.random_backbone_seed,
            "device": args.device,
            "max_train_rows_per_source": args.max_train_rows,
            "max_eval_rows_per_source": args.max_eval_rows,
        },
        "before": {"train": before_all, "eval": before_eval, "eval_by_source": before_by_source},
        "after": {"train": after_all, "eval": after_eval, "eval_by_source": after_by_source},
        "loss_history": loss_history,
        "token_accounting": training,
        "weight_transfer": {
            "compatibility": transfer["compatibility"],
            "groups": transfer["groups"],
            "recommendation": transfer["recommendation"],
        },
        "claim_boundary": (
            "Public-train-only text/accessibility continuation across mobile, browser, and desktop "
            "projections with source-disjoint held-out metrics; no official benchmark score, native "
            "emulator/browser/desktop/MCP execution, screenshot grounding, or external-account claim."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
