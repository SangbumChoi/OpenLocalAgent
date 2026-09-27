"""Capability midtraining on a scheduled mixture of text, code, and agent conversations.

Midtraining is continued pretraining, not a second name for SFT.  It shifts a base model toward
the deployment domain while retaining a substantial general-text stream.  Text/code sources use
full next-token loss; canonical ``Conversation`` sources retain the assistant-only mask from
``render_conversation``.  Mixture weights can change linearly over the run.
Their configured unit is explicit: legacy whole-batch draws, measured input tokens, or measured
supervised loss tokens.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from localagent.data.conversation_artifact import assert_no_conversation_overlap
from localagent.data.prompt_contract import (
    LEGACY_CONVERSATION_PROMPT_CONTRACT,
    assert_prompt_contract_tokenizer,
    resolve_conversation_prompt_contract,
)
from localagent.data.render import (
    IGNORE,
    CatalogTokenCache,
    conversation_row_token_counts,
    render_conversation_rows,
    shifted_token_counts,
)
from localagent.data.schema import Conversation
from localagent.model.config import ModelConfig
from localagent.train.device import autocast_ctx
from localagent.train.loop import cosine_lr, pad_batch, router_loss_terms, set_lr, wsd_lr
from localagent.train.stage_data import (
    assert_checkpoint_compatible as assert_checkpoint_compatible,  # noqa: PLC0414
)
from localagent.train.stage_sampling import (
    next_midtrain_microbatch,
)
from localagent.train.stage_sampling import (
    sample_counted_batch as _sample_counted_batch,
)

_MIXTURE_STATE_VERSION = 1
_MIXTURE_UNITS = frozenset({"draws", "input_tokens", "loss_tokens"})


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_tokenizer_sha256(manifest: dict) -> str | None:
    """Read tokenizer artifact fingerprints emitted by current or earlier corpus manifests."""

    direct = manifest.get("tokenizer_sha256")
    if isinstance(direct, str):
        return direct
    tokenizer_meta = manifest.get("tokenizer")
    if isinstance(tokenizer_meta, dict) and isinstance(tokenizer_meta.get("sha256"), str):
        return tokenizer_meta["sha256"]
    training = manifest.get("tokenizer_training")
    if isinstance(training, dict):
        artifact = training.get("artifact")
        if isinstance(artifact, dict) and isinstance(artifact.get("sha256"), str):
            return artifact["sha256"]
    return None


def validate_packed_source(
    dataset,
    cfg: ModelConfig,
    *,
    source_name: str,
    configured_tokenizer_sha256: str | None,
) -> str | None:
    """Validate one packed source and return its tokenizer fingerprint when available."""

    if dataset.seq_len > cfg.max_seq_len:
        raise ValueError(
            f"source {source_name!r} seq_len {dataset.seq_len} exceeds "
            f"model max_seq_len {cfg.max_seq_len}"
        )
    source_vocab = int(dataset.manifest["vocab_size"])
    if source_vocab != cfg.vocab_size:
        raise ValueError(
            f"source {source_name!r} vocabulary {source_vocab} does not match "
            f"model vocabulary {cfg.vocab_size}"
        )
    fingerprint = _manifest_tokenizer_sha256(dataset.manifest)
    if fingerprint is not None:
        if configured_tokenizer_sha256 is None:
            raise ValueError(
                f"source {source_name!r} records a tokenizer fingerprint but the configured "
                "tokenizer has no file fingerprint"
            )
        if fingerprint != configured_tokenizer_sha256:
            raise ValueError(
                f"source {source_name!r} tokenizer fingerprint does not match configured tokenizer"
            )
    return fingerprint


@dataclass(frozen=True)
class _PackedSplitMembership:
    source_name: str
    split: str
    assignment_sha256: str
    assignment_artifact_sha256: str
    document_set_sha256: str
    record_pairs: frozenset[tuple[str, str]]
    identities: frozenset[str]
    content_sha256s: frozenset[str]

    def audit_metadata(self) -> dict[str, Any]:
        return {
            "source": self.source_name,
            "split": self.split,
            "documents": len(self.record_pairs),
            "unique_document_identities": len(self.identities),
            "assignment_sha256": self.assignment_sha256,
            "assignment_artifact_sha256": self.assignment_artifact_sha256,
            "document_set_sha256": self.document_set_sha256,
        }


def _packed_split_membership(dataset, *, source_name: str) -> _PackedSplitMembership:
    """Verify and materialize one content-bound packed split assignment."""

    from localagent.data.pretrain_corpus import (
        SPLIT_ASSIGNMENT_FORMAT,
        SPLIT_ASSIGNMENT_VERSION,
        load_frozen_split_assignment_manifest,
    )

    manifest_path = Path(dataset.root) / "manifest.json"
    try:
        assignment = load_frozen_split_assignment_manifest(manifest_path)
    except ValueError as error:
        raise ValueError(
            f"cannot prove packed held-out disjointness for {source_name!r}: {error}"
        ) from error

    assignment_digest = hashlib.sha256()
    assignment_first = True
    records = 0
    previous_key: tuple[str, str] | None = None
    identity_splits: dict[str, str] = {}
    split_identity_records: list[str] = []
    record_pairs: set[tuple[str, str]] = set()
    identities: set[str] = set()
    content_sha256s: set[str] = set()
    with assignment.path.open(encoding="utf-8") as handle:
        try:
            header = json.loads(next(handle))
        except (StopIteration, json.JSONDecodeError) as error:
            raise ValueError(
                f"cannot prove packed held-out disjointness for {source_name!r}: "
                "invalid split-assignment header"
            ) from error
        if header != {
            "format": SPLIT_ASSIGNMENT_FORMAT,
            "schema_version": SPLIT_ASSIGNMENT_VERSION,
        }:
            raise ValueError(
                f"cannot prove packed held-out disjointness for {source_name!r}: "
                "unsupported split-assignment header"
            )
        for line_no, line in enumerate(handle, start=2):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"cannot prove packed held-out disjointness for {source_name!r}: "
                    f"invalid assignment row {line_no}"
                ) from error
            document_id = row.get("document_id")
            document_sha256 = row.get("document_sha256")
            identity = row.get("identity_sha256")
            split = row.get("split")
            valid_hashes = all(
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
                for value in (document_sha256, identity)
            )
            if (
                not isinstance(document_id, str)
                or not valid_hashes
                or identity != hashlib.sha256(document_id.encode("utf-8")).hexdigest()
                or split not in {"train", "val"}
            ):
                raise ValueError(
                    f"cannot prove packed held-out disjointness for {source_name!r}: "
                    f"invalid assignment binding at row {line_no}"
                )
            key = (identity, document_sha256)
            if previous_key is not None and key <= previous_key:
                raise ValueError(
                    f"cannot prove packed held-out disjointness for {source_name!r}: "
                    "assignment rows are not unique and sorted"
                )
            previous_key = key
            previous_split = identity_splits.setdefault(identity, split)
            if previous_split != split:
                raise ValueError(
                    f"cannot prove packed held-out disjointness for {source_name!r}: "
                    "one document identity is assigned to multiple splits"
                )
            value = f"{identity}:{document_sha256}:{split}"
            if not assignment_first:
                assignment_digest.update(b"\n")
            assignment_digest.update(value.encode("ascii"))
            assignment_first = False
            records += 1
            if split == dataset.split:
                split_identity_records.append(identity)
                record_pairs.add((identity, document_sha256))
                identities.add(identity)
                content_sha256s.add(document_sha256)
    if records != assignment.records:
        raise ValueError(
            f"cannot prove packed held-out disjointness for {source_name!r}: "
            "assignment record count mismatch"
        )
    if assignment_digest.hexdigest() != assignment.assignment_sha256:
        raise ValueError(
            f"cannot prove packed held-out disjointness for {source_name!r}: "
            "assignment content fingerprint mismatch"
        )
    split_metadata = dataset.manifest["splits"][dataset.split]
    expected_documents = split_metadata.get("documents")
    expected_document_set = split_metadata.get("document_set_sha256")
    observed_document_set = hashlib.sha256(
        "\n".join(sorted(split_identity_records)).encode("ascii")
    ).hexdigest()
    if (
        isinstance(expected_documents, bool)
        or not isinstance(expected_documents, int)
        or expected_documents != len(record_pairs)
        or expected_document_set != observed_document_set
    ):
        raise ValueError(
            f"cannot prove packed held-out disjointness for {source_name!r}: "
            "split document fingerprint/count mismatch"
        )
    return _PackedSplitMembership(
        source_name=source_name,
        split=dataset.split,
        assignment_sha256=assignment.assignment_sha256,
        assignment_artifact_sha256=assignment.sha256,
        document_set_sha256=observed_document_set,
        record_pairs=frozenset(record_pairs),
        identities=frozenset(identities),
        content_sha256s=frozenset(content_sha256s),
    )


def _audit_packed_holdout_splits(
    train_sources: Sequence[tuple[str, object]],
    eval_sources: Sequence[tuple[str, object]],
) -> dict[str, Any] | None:
    """Fail closed unless every packed train/eval pair has verified zero document overlap."""

    if not eval_sources:
        return None
    train_memberships = [
        _packed_split_membership(dataset, source_name=name) for name, dataset in train_sources
    ]
    eval_memberships = [
        _packed_split_membership(dataset, source_name=name) for name, dataset in eval_sources
    ]
    pair_audits = []
    for train_membership in train_memberships:
        for eval_membership in eval_memberships:
            identity_overlap = train_membership.identities & eval_membership.identities
            content_overlap = train_membership.content_sha256s & eval_membership.content_sha256s
            if identity_overlap or content_overlap:
                raise ValueError(
                    "midtrain packed held-out contamination between "
                    f"{train_membership.source_name!r} and "
                    f"{eval_membership.source_name!r}: "
                    f"{len(identity_overlap)} document identity and "
                    f"{len(content_overlap)} content fingerprint overlap(s)"
                )
            pair_audits.append(
                {
                    "train_source": train_membership.source_name,
                    "eval_source": eval_membership.source_name,
                    "document_identity_overlap": 0,
                    "document_content_overlap": 0,
                }
            )
    return {
        "proof": "verified_content_bound_split_assignment_rows",
        "train": [membership.audit_metadata() for membership in train_memberships],
        "eval": [membership.audit_metadata() for membership in eval_memberships],
        "pairs": pair_audits,
    }


class ConversationDataset:
    """In-memory masked rows backed by the project's canonical interchange schema."""

    def __init__(
        self,
        conversations: Sequence[Conversation],
        tokenizer,
        max_seq_len: int,
        *,
        conversation_prompt_contract: str | None = None,
    ):
        self.conversation_prompt_contract = assert_prompt_contract_tokenizer(
            tokenizer,
            conversation_prompt_contract,
        )
        self.catalog_token_cache = CatalogTokenCache(tokenizer)
        self.rows = []
        for conv in conversations:
            self.rows.extend(
                render_conversation_rows(
                    conv,
                    tokenizer,
                    prompt_contract=self.conversation_prompt_contract,
                    max_seq_len=max_seq_len,
                    catalog_cache=self.catalog_token_cache,
                )
            )
        self._row_token_counts = tuple(shifted_token_counts(row) for row in self.rows)
        self.pad_id = tokenizer.pad_id
        if not self.rows:
            raise ValueError("conversation dataset is empty")

    def __len__(self) -> int:
        return len(self.rows)

    def sample_batch(self, batch_size: int, rng, device):
        rows = [self.rows[rng.randrange(len(self.rows))] for _ in range(batch_size)]
        return pad_batch(rows, self.pad_id, device)

    def sample_batch_with_counts(self, batch_size: int, rng, device):
        """Sample once and return exact pre-padding LM input/target counts."""

        rows = [self.rows[rng.randrange(len(self.rows))] for _ in range(batch_size)]
        counts = [shifted_token_counts(row) for row in rows]
        input_tokens = sum(value[0] for value in counts)
        loss_tokens = sum(value[1] for value in counts)
        x, y = pad_batch(rows, self.pad_id, device)
        return x, y, input_tokens, loss_tokens

    def sample_batch_token_counts(self, batch_size: int, rng) -> tuple[int, int]:
        """Sample the runtime row indices and count them without collating token tensors."""

        counts = [self._row_token_counts[rng.randrange(len(self.rows))] for _ in range(batch_size)]
        return (
            sum(value[0] for value in counts),
            sum(value[1] for value in counts),
        )


class ConversationTokenCountDataset:
    """Exact conversation row counts for planners, without retaining token rows."""

    def __init__(
        self,
        conversations: Sequence[Conversation],
        tokenizer,
        max_seq_len: int,
        *,
        conversation_prompt_contract: str | None = None,
    ):
        self.conversation_prompt_contract = assert_prompt_contract_tokenizer(
            tokenizer,
            conversation_prompt_contract,
        )
        self.catalog_token_cache = CatalogTokenCache(tokenizer)
        row_token_counts = conversation_row_token_counts(
            conversations,
            tokenizer,
            prompt_contract=self.conversation_prompt_contract,
            max_seq_len=max_seq_len,
            catalog_cache=self.catalog_token_cache,
        )
        if not row_token_counts:
            raise ValueError("conversation dataset is empty")
        self._row_token_counts = tuple(row_token_counts)

    def __len__(self) -> int:
        return len(self._row_token_counts)

    def sample_batch_token_counts(self, batch_size: int, rng) -> tuple[int, int]:
        """Consume runtime-equivalent row draws and return their exact counts."""

        counts = [
            self._row_token_counts[rng.randrange(len(self._row_token_counts))]
            for _ in range(batch_size)
        ]
        return (
            sum(value[0] for value in counts),
            sum(value[1] for value in counts),
        )


@dataclass(frozen=True)
class MixtureSource:
    name: str
    dataset: object
    start_weight: float
    end_weight: float


class ScheduledMixture:
    """Deterministic source sampler with linearly changing, normalized weights.

    ``unit="draws"`` preserves the historical weighted-random batch sampler. Token units use
    deficit-based weighted fair queuing: after every sampled batch, its measured token mass is
    distributed to per-source target entitlements according to the current schedule, and the next
    batch comes from the source furthest below its entitlement. The discrepancy is therefore
    explicit and bounded by coarse batch granularity rather than hidden behind row counts.
    """

    def __init__(self, sources: Sequence[MixtureSource], *, unit: str = "draws"):
        if not sources:
            raise ValueError("midtraining needs at least one source")
        if len({source.name for source in sources}) != len(sources):
            raise ValueError("midtraining source names must be unique")
        if any(source.start_weight < 0 or source.end_weight < 0 for source in sources):
            raise ValueError("mixture weights must be non-negative")
        if sum(source.start_weight for source in sources) <= 0:
            raise ValueError("start weights sum to zero")
        if sum(source.end_weight for source in sources) <= 0:
            raise ValueError("end weights sum to zero")
        if unit not in _MIXTURE_UNITS:
            raise ValueError(f"mixture unit must be one of {sorted(_MIXTURE_UNITS)}, got {unit!r}")
        self.sources = list(sources)
        self.unit = unit

    @property
    def source_names(self) -> list[str]:
        return [source.name for source in self.sources]

    def weights_at(self, progress: float) -> list[float]:
        progress = min(1.0, max(0.0, progress))
        start_total = sum(source.start_weight for source in self.sources)
        end_total = sum(source.end_weight for source in self.sources)
        start = [source.start_weight / start_total for source in self.sources]
        end = [source.end_weight / end_total for source in self.sources]
        return [
            start_weight + progress * (end_weight - start_weight)
            for start_weight, end_weight in zip(start, end)
        ]

    def initial_state(self) -> dict[str, Any]:
        """Return a JSON/checkpoint-safe scheduler state."""

        return {
            "schema_version": _MIXTURE_STATE_VERSION,
            "unit": self.unit,
            "source_names": self.source_names,
            "observations": 0,
            "target_units": {name: 0.0 for name in self.source_names},
            "served_units": {name: 0 for name in self.source_names},
        }

    def validate_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and copy scheduler state before a resume."""

        if state.get("schema_version") != _MIXTURE_STATE_VERSION:
            raise ValueError("resume checkpoint has an unsupported mixture state version")
        if state.get("unit") != self.unit:
            raise ValueError("resume checkpoint mixture unit does not match configured unit")
        if state.get("source_names") != self.source_names:
            raise ValueError("resume checkpoint mixture source order does not match configuration")
        observations = state.get("observations")
        target_units = state.get("target_units")
        served_units = state.get("served_units")
        if isinstance(observations, bool) or not isinstance(observations, int):
            raise TypeError("resume checkpoint mixture observations are invalid")
        if (
            observations < 0
            or not isinstance(target_units, Mapping)
            or not isinstance(served_units, Mapping)
        ):
            raise ValueError("resume checkpoint mixture state is invalid")
        if set(target_units) != set(self.source_names) or set(served_units) != set(
            self.source_names
        ):
            raise ValueError("resume checkpoint mixture state source set does not match")
        normalized_targets: dict[str, float] = {}
        normalized_served: dict[str, int] = {}
        for name in self.source_names:
            target = target_units[name]
            served = served_units[name]
            if (
                isinstance(target, bool)
                or not isinstance(target, (int, float))
                or not math.isfinite(float(target))
                or float(target) < 0
            ):
                raise ValueError("resume checkpoint mixture target units are invalid")
            if isinstance(served, bool) or not isinstance(served, int) or served < 0:
                raise ValueError("resume checkpoint mixture served units are invalid")
            normalized_targets[name] = float(target)
            normalized_served[name] = served
        target_total = sum(normalized_targets.values())
        served_total = sum(normalized_served.values())
        if not math.isclose(target_total, served_total, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("resume checkpoint mixture target/served totals disagree")
        return {
            "schema_version": _MIXTURE_STATE_VERSION,
            "unit": self.unit,
            "source_names": self.source_names,
            "observations": observations,
            "target_units": normalized_targets,
            "served_units": normalized_served,
        }

    def choose(
        self,
        progress: float,
        rng: random.Random,
        state: Mapping[str, Any] | None = None,
    ) -> MixtureSource:
        weights = self.weights_at(progress)
        if self.unit != "draws":
            if state is None:
                raise ValueError("token-faithful mixture sampling requires scheduler state")
            target_units = state["target_units"]
            served_units = state["served_units"]
            deficits = [
                float(target_units[source.name]) - int(served_units[source.name])
                for source in self.sources
            ]
            # A source whose schedule starts at zero must not win the all-zero initial tie. If it
            # is still owed prior entitlement, it remains eligible after its instantaneous weight
            # reaches zero, allowing the integrated token contract to settle honestly.
            eligible = [
                index
                for index, weight in enumerate(weights)
                if weight > 0 or deficits[index] > 1e-12
            ]
            largest = max(deficits[index] for index in eligible)
            tied = [
                index
                for index in eligible
                if math.isclose(deficits[index], largest, rel_tol=0.0, abs_tol=1e-12)
            ]
            return self.sources[tied[rng.randrange(len(tied))]]

        # Compatibility mode: preserve the established weighted-random draw sequence.
        needle = rng.random()
        cumulative = 0.0
        for source, weight in zip(self.sources, weights):
            cumulative += weight
            if needle <= cumulative:
                return source
        return self.sources[-1]

    def observe(
        self,
        state: dict[str, Any],
        source: MixtureSource,
        *,
        progress: float,
        input_tokens: int,
        loss_tokens: int,
    ) -> None:
        """Advance target entitlements with one actually sampled batch."""

        if source.name not in state["served_units"]:
            raise ValueError("sampled source is absent from mixture scheduler state")
        if input_tokens <= 0 or loss_tokens <= 0:
            raise ValueError("midtraining batches must contain input and supervised loss tokens")
        basis_units = {
            "draws": 1,
            "input_tokens": input_tokens,
            "loss_tokens": loss_tokens,
        }[self.unit]
        for name, weight in zip(self.source_names, self.weights_at(progress)):
            state["target_units"][name] += weight * basis_units
        state["served_units"][source.name] += basis_units
        state["observations"] += 1


def _empty_token_accounting(source_names: Sequence[str]) -> dict[str, Any]:
    return {
        "input_tokens": 0,
        "loss_tokens": 0,
        "sources": {name: {"input_tokens": 0, "loss_tokens": 0} for name in source_names},
    }


def _checkpoint_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"resume checkpoint {label} must be a non-negative integer")
    return value


def _validated_token_accounting(
    value: Any,
    *,
    source_names: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("resume checkpoint has no valid token accounting")
    raw_sources = value.get("sources")
    if not isinstance(raw_sources, Mapping) or set(raw_sources) != set(source_names):
        raise ValueError("resume checkpoint token accounting source set does not match")

    sources: dict[str, dict[str, int]] = {}
    for name in source_names:
        raw_source = raw_sources[name]
        if not isinstance(raw_source, Mapping):
            raise TypeError(f"resume checkpoint token accounting for source {name!r} is invalid")
        sources[name] = {
            key: _checkpoint_nonnegative_int(
                raw_source.get(key),
                label=f"token_accounting.sources[{name!r}].{key}",
            )
            for key in ("input_tokens", "loss_tokens")
        }

    totals = {
        key: _checkpoint_nonnegative_int(
            value.get(key),
            label=f"token_accounting.{key}",
        )
        for key in ("input_tokens", "loss_tokens")
    }
    for key, total in totals.items():
        source_total = sum(source[key] for source in sources.values())
        if total != source_total:
            raise ValueError(
                f"resume checkpoint token_accounting.{key} disagrees with per-source sum"
            )
    return {**totals, "sources": sources}


def _validated_loss_history(value: Any, *, completed_steps: int) -> list[float]:
    if not isinstance(value, list) or len(value) != completed_steps:
        raise ValueError("resume checkpoint loss_history length disagrees with completed steps")
    history: list[float] = []
    for loss in value:
        if (
            isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
        ):
            raise ValueError("resume checkpoint loss_history contains an invalid value")
        history.append(float(loss))
    return history


def _reconstruct_legacy_draw_state(
    mixture: ScheduledMixture,
    *,
    completed_steps: int,
    total_steps: int,
    accum_steps: int,
    draws: Mapping[str, int],
) -> dict[str, Any]:
    """Reconstruct state omitted by checkpoints from the historical draw sampler."""

    if mixture.unit != "draws":
        raise ValueError(
            "token-faithful resume requires checkpointed mixture state; "
            "an old row-draw checkpoint cannot prove token entitlements"
        )
    state = mixture.initial_state()
    for step in range(completed_steps):
        progress = step / max(1, total_steps - 1)
        weights = mixture.weights_at(progress)
        for _ in range(accum_steps):
            for name, weight in zip(mixture.source_names, weights):
                state["target_units"][name] += weight
            state["observations"] += 1
    state["served_units"] = {name: int(draws[name]) for name in mixture.source_names}
    if state["observations"] != sum(state["served_units"].values()):
        raise ValueError("legacy resume checkpoint draw count disagrees with completed steps")
    return mixture.validate_state(state)


def _validate_mixture_accounting_state(
    mixture: ScheduledMixture,
    state: Mapping[str, Any],
    *,
    completed_steps: int,
    accum_steps: int,
    draws: Mapping[str, int],
    token_accounting: Mapping[str, Any],
) -> None:
    expected_observations = completed_steps * accum_steps
    if int(state["observations"]) != expected_observations:
        raise ValueError(
            "resume checkpoint mixture observations disagree with completed optimizer steps"
        )
    if int(state["observations"]) != sum(draws.values()):
        raise ValueError("resume checkpoint mixture observations disagree with source draws")
    if mixture.unit == "draws":
        expected = draws
    else:
        token_key = mixture.unit
        expected = {
            name: int(token_accounting["sources"][name][token_key]) for name in mixture.source_names
        }
    if any(int(state["served_units"][name]) != int(expected[name]) for name in expected):
        raise ValueError("resume checkpoint mixture served units disagree with measured accounting")


def _safe_share(value: float, total: float) -> float | None:
    return float(value) / float(total) if total else None


def _mixture_accounting_report(
    mixture: ScheduledMixture,
    state: Mapping[str, Any],
    *,
    draws: Mapping[str, int],
    token_accounting: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose scheduled entitlements and three independently measured realized shares."""

    draw_total = sum(draws.values())
    input_total = int(token_accounting["input_tokens"])
    loss_total = int(token_accounting["loss_tokens"])
    target_total = sum(float(value) for value in state["target_units"].values())
    served_total = sum(int(value) for value in state["served_units"].values())
    sources: dict[str, Any] = {}
    start_weights = mixture.weights_at(0.0)
    end_weights = mixture.weights_at(1.0)
    for index, name in enumerate(mixture.source_names):
        source_tokens = token_accounting["sources"][name]
        target_units = float(state["target_units"][name])
        served_units = int(state["served_units"][name])
        target_share = _safe_share(target_units, target_total)
        realized_share = _safe_share(served_units, served_total)
        sources[name] = {
            "start_weight": start_weights[index],
            "end_weight": end_weights[index],
            "draws": int(draws[name]),
            "draw_share": _safe_share(draws[name], draw_total),
            "input_tokens": int(source_tokens["input_tokens"]),
            "input_token_share": _safe_share(source_tokens["input_tokens"], input_total),
            "loss_tokens": int(source_tokens["loss_tokens"]),
            "loss_token_share": _safe_share(source_tokens["loss_tokens"], loss_total),
            "scheduled_target_basis_units": target_units,
            "scheduled_target_basis_share": target_share,
            "realized_basis_units": served_units,
            "realized_basis_share": realized_share,
            "basis_share_error": (
                None
                if target_share is None or realized_share is None
                else realized_share - target_share
            ),
            "basis_deficit_units": target_units - served_units,
        }
    return {
        "schema_version": _MIXTURE_STATE_VERSION,
        "unit": mixture.unit,
        "selection": (
            "weighted_random_batches"
            if mixture.unit == "draws"
            else "largest_integrated_token_deficit"
        ),
        "loss_normalization": (
            "equal_microbatch_means"
            if mixture.unit == "draws"
            else "supervised_tokens_across_accumulation"
        ),
        "schedule": "linear_per_optimizer_step",
        "observations": int(state["observations"]),
        "totals": {
            "draws": draw_total,
            "input_tokens": input_total,
            "loss_tokens": loss_total,
            "scheduled_target_basis_units": target_total,
            "realized_basis_units": served_total,
        },
        "sources": sources,
    }


@torch.no_grad()
def _evaluate_sources(
    model,
    sources: Sequence[MixtureSource],
    *,
    batches_per_source: int,
    batch_size: int,
    seed: int,
    device: str,
    amp_dtype=torch.float32,
) -> dict[str, Any]:
    """Deterministically measure teacher-forced loss/accuracy on explicit held-out sources."""

    if not sources:
        raise ValueError("midtrain held-out evaluation needs at least one source")
    if len({source.name for source in sources}) != len(sources):
        raise ValueError("midtrain held-out source names must be unique")
    if batches_per_source < 1 or batch_size < 1:
        raise ValueError("held-out batches_per_source and batch_size must be positive")
    was_training = model.training
    model.to(device).eval()
    device_obj = torch.device(device)
    source_metrics: dict[str, Any] = {}
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    for source_index, source in enumerate(sources):
        rng = random.Random(seed + source_index)
        source_loss = 0.0
        source_correct = 0
        source_tokens = 0
        for _ in range(batches_per_source):
            x, y, _, reported_loss_tokens = _sample_counted_batch(source, batch_size, rng, device)
            with autocast_ctx(device_obj, amp_dtype):
                logits, _ = model(x)
            flat_logits = logits.float().reshape(-1, logits.shape[-1])
            flat_targets = y.reshape(-1)
            mask = flat_targets != IGNORE
            observed_loss_tokens = int(mask.sum())
            if observed_loss_tokens != reported_loss_tokens:
                raise ValueError(
                    f"held-out source {source.name!r} token accounting disagrees with mask"
                )
            if observed_loss_tokens == 0:
                raise ValueError(f"held-out source {source.name!r} has no supervised tokens")
            source_loss += float(
                F.cross_entropy(
                    flat_logits,
                    flat_targets,
                    ignore_index=IGNORE,
                    reduction="sum",
                )
            )
            predictions = flat_logits.argmax(dim=-1)
            source_correct += int(((predictions == flat_targets) & mask).sum())
            source_tokens += observed_loss_tokens
        source_metrics[source.name] = {
            "batches": batches_per_source,
            "loss_tokens": source_tokens,
            "mean_loss": source_loss / source_tokens,
            "token_accuracy": source_correct / source_tokens,
        }
        total_loss += source_loss
        total_correct += source_correct
        total_tokens += source_tokens
    model.train(was_training)
    return {
        "sources": source_metrics,
        "loss_tokens": total_tokens,
        "mean_loss": total_loss / total_tokens,
        "token_accuracy": total_correct / total_tokens,
    }


def midtrain(
    model,
    mixture: ScheduledMixture,
    *,
    steps: int,
    batch_size: int,
    accum_steps: int = 1,
    lr: float = 1e-4,
    warmup: int = 100,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    lr_schedule: str = "wsd",
    decay_frac: float = 0.2,
    device: str = "cpu",
    seed: int = 0,
    log=print,
    checkpoint_path: str | Path | None = None,
    amp_dtype=torch.float32,
    checkpoint_every: int = 0,
    resume_from: str | Path | None = None,
    lineage: Mapping[str, Any] | None = None,
    tokenizer_metadata: Mapping[str, Any] | None = None,
    data_metadata: Mapping[str, Any] | None = None,
    execution: Mapping[str, Any] | None = None,
    eval_sources: Sequence[MixtureSource] | None = None,
    eval_batches: int = 0,
    eval_batch_size: int | None = None,
    eval_seed: int | None = None,
    conversation_prompt_contract: str | None = None,
    return_metrics: bool = False,
):
    """Continue training over a domain mixture.

    ``mixture.unit`` defines whether scheduled weights apply to historical whole-batch draws or
    measured input/supervised tokens. Token modes use fair-deficit selection and normalize each
    optimizer accumulation over its supervised tokens.

    The default return remains ``(history, source_draws)``. With ``return_metrics=True``, the
    second item is a metrics mapping containing draws and realized token accounting.
    """

    if steps < 1 or batch_size < 1 or accum_steps < 1:
        raise ValueError("steps, batch_size, and accum_steps must be positive")
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be non-negative")
    if lr_schedule not in {"cosine", "wsd"}:
        raise ValueError("lr_schedule must be 'cosine' or 'wsd'")
    prompt_contract = resolve_conversation_prompt_contract(conversation_prompt_contract)
    torch.manual_seed(seed)
    model.to(device).train()
    device_obj = torch.device(device)
    use_grad_scaler = device_obj.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay
    )
    rng = random.Random(seed)
    history: list[float] = []
    lm_loss_history: list[float] = []
    router_aux_loss_history: list[float] = []
    router_weighted_loss_history: list[float] = []
    draws = {source.name: 0 for source in mixture.sources}
    source_names = mixture.source_names
    token_accounting = _empty_token_accounting(source_names)
    mixture_state = mixture.initial_state()
    start_step = 0
    heldout_eval = None
    eval_contract = None
    if eval_sources:
        resolved_eval_batch_size = eval_batch_size or batch_size
        resolved_eval_seed = seed + 10_000 if eval_seed is None else eval_seed
        eval_contract = {
            "kind": "deterministic_teacher_forced_next_token",
            "sources": [source.name for source in eval_sources],
            "batches_per_source": eval_batches,
            "batch_size": resolved_eval_batch_size,
            "seed": resolved_eval_seed,
            "same_draws_pre_post": True,
        }
        if prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT:
            eval_contract["conversation_prompt_contract"] = prompt_contract

    if resume_from is not None:
        from localagent.train.stage_data import assert_resume_lineage

        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=True)
        recorded_lineage = checkpoint.get("lineage")
        if lineage is not None:
            assert_resume_lineage(checkpoint, lineage)
        elif recorded_lineage is not None:
            raise ValueError(
                "resume checkpoint records lineage but no expected lineage was provided"
            )
        recorded_execution = checkpoint.get("execution")
        if execution is not None:
            if not isinstance(recorded_execution, Mapping):
                raise ValueError(
                    "resume checkpoint has no execution identity for an exact continuation"
                )
            if dict(recorded_execution) != dict(execution):
                raise ValueError("resume checkpoint execution identity mismatch")
        elif recorded_execution is not None:
            raise ValueError("resume checkpoint records execution identity but none was provided")
        model.load_state_dict(checkpoint["state_dict"])
        recorded_optimizer = checkpoint.get("optimizer")
        if not isinstance(recorded_optimizer, Mapping):
            raise ValueError("resume checkpoint has no optimizer state")
        optimizer.load_state_dict(recorded_optimizer)
        recorded_scaler = checkpoint.get("grad_scaler")
        if use_grad_scaler:
            if not isinstance(recorded_scaler, Mapping):
                raise ValueError("resume checkpoint has no required gradient-scaler state")
            scaler.load_state_dict(recorded_scaler)
        elif recorded_scaler is not None:
            raise ValueError("resume checkpoint has unexpected gradient-scaler state")

        checkpoint_step = _checkpoint_nonnegative_int(
            checkpoint.get("step"),
            label="step",
        )
        start_step = checkpoint_step + 1
        history = _validated_loss_history(
            checkpoint.get("loss_history"),
            completed_steps=start_step,
        )
        loss_components = checkpoint.get("loss_components")
        if loss_components is None:
            if model.cfg.sparse_ffn:
                raise ValueError(
                    "sparse midtrain resume checkpoint is missing router loss components"
                )
            lm_loss_history = list(history)
            router_aux_loss_history = [0.0] * len(history)
            router_weighted_loss_history = [0.0] * len(history)
        else:
            if not isinstance(loss_components, Mapping):
                raise ValueError("midtrain resume loss_components must be a mapping")
            component_names = {
                "lm_loss": lm_loss_history,
                "router_aux": router_aux_loss_history,
                "router_weighted": router_weighted_loss_history,
            }
            for name, destination in component_names.items():
                values = loss_components.get(name)
                if not isinstance(values, list) or len(values) != len(history):
                    raise ValueError(
                        f"midtrain resume loss_components[{name!r}] is inconsistent"
                    )
                destination.extend(float(value) for value in values)
        recorded_draws = checkpoint.get("source_draws")
        if not isinstance(recorded_draws, Mapping) or set(recorded_draws) != set(draws):
            raise ValueError("resume checkpoint source set does not match midtrain mixture")
        draws = {
            name: _checkpoint_nonnegative_int(
                recorded_draws[name],
                label=f"source_draws[{name!r}]",
            )
            for name in source_names
        }
        token_accounting = _validated_token_accounting(
            checkpoint.get("token_accounting"),
            source_names=source_names,
        )
        recorded_mixture_state = checkpoint.get("mixture_state")
        if isinstance(recorded_mixture_state, Mapping):
            mixture_state = mixture.validate_state(recorded_mixture_state)
        else:
            mixture_state = _reconstruct_legacy_draw_state(
                mixture,
                completed_steps=start_step,
                total_steps=steps,
                accum_steps=accum_steps,
                draws=draws,
            )
        _validate_mixture_accounting_state(
            mixture,
            mixture_state,
            completed_steps=start_step,
            accum_steps=accum_steps,
            draws=draws,
            token_accounting=token_accounting,
        )
        recorded_python_rng = checkpoint.get("rng_state")
        if recorded_python_rng is None:
            raise ValueError("resume checkpoint has no Python RNG state")
        try:
            rng.setstate(recorded_python_rng)
        except (TypeError, ValueError) as error:
            raise ValueError("resume checkpoint Python RNG state is invalid") from error
        recorded_torch_rng = checkpoint.get("torch_rng_state")
        if not isinstance(recorded_torch_rng, torch.Tensor):
            raise ValueError("resume checkpoint has no valid Torch RNG state")
        torch.set_rng_state(recorded_torch_rng.cpu())
        if device_obj.type == "cuda":
            recorded_cuda_rng = checkpoint.get("cuda_rng_state_all")
            if not isinstance(recorded_cuda_rng, list):
                raise ValueError("resume checkpoint has no CUDA RNG state")
            torch.cuda.set_rng_state_all(recorded_cuda_rng)
        elif device_obj.type == "mps":
            recorded_mps_rng = checkpoint.get("mps_rng_state")
            if not isinstance(recorded_mps_rng, torch.Tensor):
                raise ValueError("resume checkpoint has no MPS RNG state")
            torch.mps.set_rng_state(recorded_mps_rng.cpu())
        elif device_obj.type == "xpu":
            recorded_xpu_rng = checkpoint.get("xpu_rng_state_all")
            if not isinstance(recorded_xpu_rng, list):
                raise ValueError("resume checkpoint has no XPU RNG state")
            torch.xpu.set_rng_state_all(recorded_xpu_rng)
        if start_step > steps:
            raise ValueError(
                f"resume checkpoint is already at step {start_step - 1}, beyond total steps {steps}"
            )

    if eval_sources:
        if resume_from is not None:
            recorded_heldout = checkpoint.get("heldout_eval")
            if not isinstance(recorded_heldout, Mapping):
                raise ValueError(
                    "resume checkpoint has no heldout_eval baseline; refusing to relabel "
                    "the resumed model as the original pre-midtrain baseline"
                )
            if recorded_heldout.get("contract") != eval_contract:
                raise ValueError("resume checkpoint heldout_eval contract mismatch")
            if not isinstance(recorded_heldout.get("pre"), Mapping):
                raise ValueError("resume checkpoint has no valid heldout_eval pre metrics")
            heldout_eval = copy.deepcopy(dict(recorded_heldout))
            # The resumed checkpoint may be the final periodic save written just before post-eval.
            # Retain its original baseline and recompute only the final-model side.
            heldout_eval["post"] = None
            heldout_eval["delta"] = None
        else:
            heldout_pre = _evaluate_sources(
                model,
                eval_sources,
                batches_per_source=eval_batches,
                batch_size=eval_contract["batch_size"],
                seed=eval_contract["seed"],
                device=device,
                amp_dtype=amp_dtype,
            )
            heldout_eval = {
                "contract": eval_contract,
                "pre": heldout_pre,
                "post": None,
                "delta": None,
            }

    def save(step: int) -> None:
        if checkpoint_path is None:
            return
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cfg": model.cfg.__dict__,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "grad_scaler": scaler.state_dict() if use_grad_scaler else None,
            "step": step,
            "loss_history": history,
            "loss_components": {
                "lm_loss": lm_loss_history,
                "router_aux": router_aux_loss_history,
                "router_weighted": router_weighted_loss_history,
            },
            "source_draws": draws,
            "token_accounting": token_accounting,
            "mixture_state": mixture_state,
            "mixture_accounting": _mixture_accounting_report(
                mixture,
                mixture_state,
                draws=draws,
                token_accounting=token_accounting,
            ),
            "rng_state": rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if device_obj.type == "cuda" else None
            ),
            "mps_rng_state": (torch.mps.get_rng_state() if device_obj.type == "mps" else None),
            "xpu_rng_state_all": (
                torch.xpu.get_rng_state_all() if device_obj.type == "xpu" else None
            ),
            "stage": "midtrain",
            "lineage": lineage,
            "heldout_eval": heldout_eval,
        }
        if prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT:
            payload["conversation_prompt_contract"] = prompt_contract
        if tokenizer_metadata is not None:
            payload["tokenizer"] = dict(tokenizer_metadata)
        if data_metadata is not None:
            payload["data"] = dict(data_metadata)
        if execution is not None:
            payload["execution"] = dict(execution)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    for step in range(start_step, steps):
        current_lr = (
            wsd_lr(step, steps, lr, warmup, decay_frac, min_ratio=0.0)
            if lr_schedule == "wsd"
            else cosine_lr(step, steps, lr, warmup, 0.1)
        )
        set_lr(optimizer, current_lr)
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_lm_loss = 0.0
        step_router_aux = 0.0
        step_router_weighted = 0.0
        sampled_batches = []
        for _ in range(accum_steps):
            sampled = next_midtrain_microbatch(
                mixture,
                mixture_state,
                rng,
                step=step,
                total_steps=steps,
                batch_size=batch_size,
                device=device,
            )
            source = sampled.source
            draws[source.name] += 1
            input_tokens = sampled.input_tokens
            loss_tokens = sampled.loss_tokens
            token_accounting["input_tokens"] += input_tokens
            token_accounting["loss_tokens"] += loss_tokens
            token_accounting["sources"][source.name]["input_tokens"] += input_tokens
            token_accounting["sources"][source.name]["loss_tokens"] += loss_tokens
            sampled_batches.append((sampled.x, sampled.y, loss_tokens))
        step_loss_tokens = sum(batch[2] for batch in sampled_batches)
        for x, y, loss_tokens in sampled_batches:
            with autocast_ctx(device_obj, amp_dtype):
                _, lm_loss = model(x, targets=y)
                loss, router_aux, router_weighted = router_loss_terms(model, lm_loss)
            loss_weight = (
                1.0 / accum_steps if mixture.unit == "draws" else loss_tokens / step_loss_tokens
            )
            scaler.scale(loss * loss_weight).backward()
            step_loss += float(loss.detach()) * loss_weight
            step_lm_loss += float(lm_loss.detach()) * loss_weight
            step_router_aux += float(router_aux.detach()) * loss_weight
            step_router_weighted += float(router_weighted.detach()) * loss_weight
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        history.append(step_loss)
        lm_loss_history.append(step_lm_loss)
        router_aux_loss_history.append(step_router_aux)
        router_weighted_loss_history.append(step_router_weighted)
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            message = (
                f"  [midtrain] step {step:4d}/{steps}  loss {step_loss:.3f}  "
                f"source_draws={draws}"
            )
            if model.cfg.sparse_ffn:
                message += (
                    f"  lm {step_lm_loss:.3f}  router_aux {step_router_aux:.3f}  "
                    f"router_weighted {step_router_weighted:.3f}"
                )
            log(message)
        if checkpoint_every and (step + 1) % checkpoint_every == 0:
            save(step)

    if eval_sources:
        heldout_post = _evaluate_sources(
            model,
            eval_sources,
            batches_per_source=eval_batches,
            batch_size=heldout_eval["contract"]["batch_size"],
            seed=heldout_eval["contract"]["seed"],
            device=device,
            amp_dtype=amp_dtype,
        )
        heldout_eval["post"] = heldout_post
        heldout_eval["delta"] = {
            "mean_loss": heldout_post["mean_loss"] - heldout_eval["pre"]["mean_loss"],
            "token_accuracy": (
                heldout_post["token_accuracy"] - heldout_eval["pre"]["token_accuracy"]
            ),
        }
    save(steps - 1)
    metrics = {
        "steps_completed": len(history),
        "source_draws": draws,
        "token_accounting": token_accounting,
        "mixture_accounting": _mixture_accounting_report(
            mixture,
            mixture_state,
            draws=draws,
            token_accounting=token_accounting,
        ),
        "heldout_eval": heldout_eval,
        "loss_components": {
            "optimization": history,
            "lm_loss": lm_loss_history,
            "router_aux": router_aux_loss_history,
            "router_weighted": router_weighted_loss_history,
        },
    }
    if prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT:
        metrics["conversation_prompt_contract"] = prompt_contract
    return (history, metrics) if return_metrics else (history, draws)


def run(config_path: str, *, resume: bool | None = None) -> None:
    """Run midtraining from a declarative source mixture."""

    import yaml

    from localagent.data.pretrain_corpus import PackedShardDataset
    from localagent.model import LocalAgentLM
    from localagent.model.tokenizer import load_tokenizer
    from localagent.train.device import execution_metadata, resolve_device, resolve_dtype
    from localagent.train.stage_data import (
        build_stage_lineage,
        canonical_sha256,
        load_conversation_source,
        load_stage_parent_checkpoint,
        tokenizer_identity,
    )

    config = yaml.safe_load(Path(config_path).read_text())
    if config.get("stage", "midtrain") != "midtrain":
        raise ValueError(f"expected stage='midtrain', got {config.get('stage')!r}")
    data_cfg = config["data"]
    conversation_prompt_contract = resolve_conversation_prompt_contract(
        data_cfg.get("conversation_prompt_contract")
    )
    strict_conversation_artifacts = data_cfg.get("strict_conversation_artifacts", False)
    if not isinstance(strict_conversation_artifacts, bool):
        raise TypeError("data.strict_conversation_artifacts must be boolean")

    cfg = ModelConfig.from_yaml(config["model_config"])
    cfg.assert_within_budget()
    tok_cfg = data_cfg.get("tokenizer", {"kind": "byte"})
    tokenizer = load_tokenizer(tok_cfg.get("kind", "byte"), tok_cfg.get("path"))
    if tokenizer.vocab_size != cfg.vocab_size:
        raise ValueError("tokenizer vocabulary does not match model config")
    assert_prompt_contract_tokenizer(tokenizer, conversation_prompt_contract)
    tokenizer_path = tok_cfg.get("path")
    tokenizer_lineage = tokenizer_identity(
        str(tok_cfg.get("kind", "byte")),
        vocab_size=tokenizer.vocab_size,
        path=tokenizer_path,
    )
    init_from = Path(config["init_from"])
    checkpoint, parent_checkpoint_sha256 = load_stage_parent_checkpoint(
        init_from,
        stage="midtrain",
        requested_model_config=cfg,
        expected_tokenizer_sha256=str(tokenizer_lineage["sha256"]),
    )
    runtime = config.get("runtime", {})
    seed = int(runtime.get("seed", 0))
    requested_device = runtime.get("device", "auto")
    requested_dtype = runtime.get("dtype", "auto")
    device = resolve_device(requested_device)
    dtype = resolve_dtype(device, requested_dtype)
    execution = execution_metadata(
        requested_device=requested_device,
        resolved_device=device,
        requested_dtype=requested_dtype,
        resolved_dtype=dtype,
    )
    torch.manual_seed(seed)
    model = LocalAgentLM(cfg)
    state = checkpoint.get("state_dict", checkpoint.get("model"))
    if state is None:
        raise ValueError("init_from checkpoint has no state_dict/model")
    model.load_state_dict(state)

    configured_tokenizer_sha256 = (
        _sha256_file(tokenizer_path) if tokenizer_path is not None else None
    )
    mixture_cfg = data_cfg.get("mixture", {})
    if not isinstance(mixture_cfg, Mapping):
        raise TypeError("midtrain data.mixture must be a mapping")
    mixture_unit = str(mixture_cfg.get("unit", "draws"))
    sources: list[MixtureSource] = []
    source_identities: list[dict[str, Any]] = []
    source_metadata: list[dict[str, Any]] = []
    train_source_keys: set[tuple[str, str, str | None]] = set()
    train_artifact_fingerprints: set[str] = set()
    train_conversations_for_audit: list[Conversation] = []
    packed_train_sources: list[tuple[str, object]] = []
    shard_tokenizer_fingerprints: set[str] = set()
    for source_cfg in data_cfg["sources"]:
        source_type = source_cfg["type"]
        source_key = (
            source_type,
            str(Path(source_cfg["path"]).resolve()),
            source_cfg.get("split", "train") if source_type == "shards" else None,
        )
        if source_key in train_source_keys:
            raise ValueError("midtrain training sources contain a duplicate artifact/split")
        train_source_keys.add(source_key)
        if source_type == "shards":
            dataset = PackedShardDataset(source_cfg["path"], split=source_cfg.get("split", "train"))
            packed_train_sources.append((source_cfg["name"], dataset))
            fingerprint = validate_packed_source(
                dataset,
                cfg,
                source_name=source_cfg["name"],
                configured_tokenizer_sha256=configured_tokenizer_sha256,
            )
            if fingerprint is not None:
                shard_tokenizer_fingerprints.add(fingerprint)
            artifact_identity = {
                "manifest_sha256": canonical_sha256(dataset.manifest),
                "split": dataset.split,
            }
        elif source_type == "conversations":
            loaded_source = load_conversation_source(
                source_cfg,
                require_verified=strict_conversation_artifacts,
                expected_split="train",
            )
            source_conversations = list(loaded_source.conversations)
            dataset = ConversationDataset(
                source_conversations,
                tokenizer,
                cfg.max_seq_len,
                conversation_prompt_contract=conversation_prompt_contract,
            )
            train_conversations_for_audit.extend(source_conversations)
            artifact_identity = dict(loaded_source.identity)
        else:
            raise ValueError(f"unknown midtrain source type {source_type!r}")
        artifact_fingerprint = canonical_sha256(
            {"type": source_type, "artifact": artifact_identity}
        )
        if artifact_fingerprint in train_artifact_fingerprints:
            raise ValueError("midtrain training sources contain duplicate content identities")
        train_artifact_fingerprints.add(artifact_fingerprint)
        start_weight = float(source_cfg["weight"])
        sources.append(
            MixtureSource(
                name=source_cfg["name"],
                dataset=dataset,
                start_weight=start_weight,
                end_weight=float(source_cfg.get("end_weight", start_weight)),
            )
        )
        source_identities.append(
            {
                "name": source_cfg["name"],
                "type": source_type,
                "split": source_cfg.get("split"),
                "start_weight": start_weight,
                "end_weight": float(source_cfg.get("end_weight", start_weight)),
                "artifact": artifact_identity,
            }
        )
        source_metadata.append(
            {
                "name": source_cfg["name"],
                "type": source_type,
                "path": str(source_cfg["path"]),
                "split": source_cfg.get("split"),
            }
        )
    if len(shard_tokenizer_fingerprints) > 1:
        raise ValueError("midtrain shard sources use different tokenizer fingerprints")

    eval_sources: list[MixtureSource] = []
    eval_source_identities: list[dict[str, Any]] = []
    eval_source_metadata: list[dict[str, Any]] = []
    eval_source_keys: set[tuple[str, str, str | None]] = set()
    eval_artifact_fingerprints: set[str] = set()
    eval_conversations_for_audit: list[Conversation] = []
    packed_eval_sources: list[tuple[str, object]] = []
    for source_cfg in data_cfg.get("eval_sources", []):
        source_type = source_cfg["type"]
        source_key = (
            source_type,
            str(Path(source_cfg["path"]).resolve()),
            source_cfg.get("split", "val") if source_type == "shards" else None,
        )
        if source_key in train_source_keys:
            raise ValueError("midtrain held-out source exactly overlaps a training source")
        if source_key in eval_source_keys:
            raise ValueError("midtrain held-out sources contain a duplicate artifact/split")
        eval_source_keys.add(source_key)
        if source_type == "shards":
            eval_dataset = PackedShardDataset(
                source_cfg["path"], split=source_cfg.get("split", "val")
            )
            packed_eval_sources.append((source_cfg["name"], eval_dataset))
            fingerprint = validate_packed_source(
                eval_dataset,
                cfg,
                source_name=source_cfg["name"],
                configured_tokenizer_sha256=configured_tokenizer_sha256,
            )
            if fingerprint is not None:
                shard_tokenizer_fingerprints.add(fingerprint)
            artifact_identity = {
                "manifest_sha256": canonical_sha256(eval_dataset.manifest),
                "split": eval_dataset.split,
            }
        elif source_type == "conversations":
            loaded_source = load_conversation_source(
                source_cfg,
                require_verified=strict_conversation_artifacts,
                expected_split="eval",
            )
            eval_conversations = list(loaded_source.conversations)
            eval_dataset = ConversationDataset(
                eval_conversations,
                tokenizer,
                cfg.max_seq_len,
                conversation_prompt_contract=conversation_prompt_contract,
            )
            eval_conversations_for_audit.extend(eval_conversations)
            artifact_identity = dict(loaded_source.identity)
        else:
            raise ValueError(f"unknown midtrain eval source type {source_type!r}")
        artifact_fingerprint = canonical_sha256(
            {"type": source_type, "artifact": artifact_identity}
        )
        if artifact_fingerprint in train_artifact_fingerprints:
            raise ValueError("midtrain held-out source content overlaps a training source")
        if artifact_fingerprint in eval_artifact_fingerprints:
            raise ValueError("midtrain held-out sources contain duplicate content identities")
        eval_artifact_fingerprints.add(artifact_fingerprint)
        eval_sources.append(
            MixtureSource(
                name=source_cfg["name"],
                dataset=eval_dataset,
                start_weight=1.0,
                end_weight=1.0,
            )
        )
        eval_source_identities.append(
            {
                "name": source_cfg["name"],
                "type": source_type,
                "split": source_cfg.get("split"),
                "artifact": artifact_identity,
            }
        )
        eval_source_metadata.append(
            {
                "name": source_cfg["name"],
                "type": source_type,
                "path": str(source_cfg["path"]),
                "split": source_cfg.get("split"),
            }
        )
    conversation_overlap_audit = assert_no_conversation_overlap(
        train_conversations_for_audit,
        eval_conversations_for_audit,
        left_label="midtrain training",
        right_label="held-out",
        conversation_prompt_contract=conversation_prompt_contract,
    )
    if len(shard_tokenizer_fingerprints) > 1:
        raise ValueError("midtrain train/eval shard sources use different tokenizer fingerprints")
    packed_holdout_audit = _audit_packed_holdout_splits(
        packed_train_sources,
        packed_eval_sources,
    )

    schedule = config.get("schedule", {})
    batch = config.get("batch", {})
    optim = config.get("optim", {})
    log_cfg = config.get("log", {})
    evaluation_cfg = config.get("evaluation", {})
    out_dir = Path(log_cfg.get("out_dir", "runs/midtrain"))
    checkpoint_path = out_dir / "latest.pt"
    configured_resume = runtime.get("resume", False)
    if not isinstance(configured_resume, bool):
        raise TypeError("runtime.resume must be boolean")
    resume_requested = configured_resume if resume is None else resume
    if not isinstance(resume_requested, bool):
        raise TypeError("resume override must be boolean or None")
    if resume is True and not checkpoint_path.exists():
        raise FileNotFoundError(
            f"midtrain resume requested but checkpoint does not exist: {checkpoint_path}"
        )
    resume_from = checkpoint_path if resume_requested and checkpoint_path.exists() else None
    lineage = build_stage_lineage(
        stage="midtrain",
        config=config,
        model_config=cfg.__dict__,
        data_identity={
            **(
                {"conversation_prompt_contract": conversation_prompt_contract}
                if conversation_prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT
                else {}
            ),
            "mixture": {"unit": mixture_unit},
            "sources": source_identities,
            "eval_sources": eval_source_identities,
            "conversation_overlap_audit": conversation_overlap_audit.as_dict(),
            "packed_holdout_audit": packed_holdout_audit,
        },
        tokenizer=tokenizer_lineage,
        workspace=Path(__file__).resolve(),
        parent_checkpoint_sha256=parent_checkpoint_sha256,
    )
    loss_history, metrics = midtrain(
        model,
        ScheduledMixture(sources, unit=mixture_unit),
        steps=int(schedule.get("total_steps", 4_000)),
        batch_size=int(batch.get("micro_batch_size", 8)),
        accum_steps=int(batch.get("grad_accum_steps", 1)),
        lr=float(optim.get("lr", 1e-4)),
        warmup=int(schedule.get("warmup_steps", 100)),
        weight_decay=float(optim.get("weight_decay", 0.1)),
        grad_clip=float(optim.get("grad_clip", 1.0)),
        lr_schedule=str(schedule.get("type", "wsd")),
        decay_frac=float(schedule.get("decay_frac", 0.2)),
        device=device,
        seed=seed,
        checkpoint_path=checkpoint_path,
        amp_dtype=dtype,
        checkpoint_every=int(log_cfg.get("ckpt_every", 0)),
        resume_from=resume_from,
        lineage=lineage,
        tokenizer_metadata={
            "kind": str(tok_cfg.get("kind", "byte")),
            "path": tokenizer_path,
            "sha256": tokenizer_lineage["sha256"],
        },
        data_metadata={
            **(
                {"conversation_prompt_contract": conversation_prompt_contract}
                if conversation_prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT
                else {}
            ),
            "mixture": {"unit": mixture_unit},
            "sources": source_metadata,
            "eval_sources": eval_source_metadata,
            "heldout_conversation_overlap": 0,
            "heldout_rendered_prompt_overlap": 0,
            "conversation_overlap_audit": conversation_overlap_audit.as_dict(),
            "packed_holdout_audit": packed_holdout_audit,
        },
        execution=execution,
        eval_sources=eval_sources,
        eval_batches=(int(evaluation_cfg.get("batches_per_source", 8)) if eval_sources else 0),
        eval_batch_size=(
            int(evaluation_cfg.get("batch_size", batch.get("micro_batch_size", 8)))
            if eval_sources
            else None
        ),
        eval_seed=(int(evaluation_cfg.get("seed", seed + 10_000)) if eval_sources else None),
        conversation_prompt_contract=conversation_prompt_contract,
        return_metrics=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "stage": "midtrain",
                "checkpoint": str(checkpoint_path),
                "loss_last": loss_history[-1] if loss_history else None,
                "loss_steps": len(loss_history),
                **metrics,
                "lineage": lineage,
                "execution": execution,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def mixture_report(config_path: str) -> str:
    """Return a compact, inspectable source summary without loading model weights."""

    import yaml

    config = yaml.safe_load(Path(config_path).read_text())
    return json.dumps(
        {
            "mixture": config["data"].get("mixture", {"unit": "draws"}),
            "sources": config["data"]["sources"],
        },
        indent=2,
    )
